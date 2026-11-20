from __future__ import annotations

import random
import logging
import time
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from core.domain import CorruptionResult, EvaluationResult, OcelLog, Prediction
from core.metrics import evaluate_rankings, expected_rank_metrics

logger = logging.getLogger(__name__)

PAIR_FEATURES = 5


def _pair_vector(total: int, linked: int, probability: float, o2o: int, m: int) -> list[float]:
    """[log(1+co-occurrence count), share of context objects co-occurring with the candidate,
    mean conditional probability, log(1+O2O links to the context), share of context objects
    O2O-linked to the candidate]."""
    return [math.log1p(total), linked / m, probability / m, math.log1p(o2o), o2o / m]

@dataclass(frozen=True)
class GnnConfig:
    hidden_dim: int = 128
    epochs: int = 400
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    top_k: int = 10
    negatives_per_positive: int = 1
    seed: int = 42
    sage_aggregation: str = "sum"
    sampled_softmax_temperature: float = 0.1
    same_type_negative_fraction: float = 0.5
    cooccurrence_candidates: bool = False
    checkpoint_path: str | None = None
    validation_interval: int = 50
    model_dir: str = "models"
    # Revision (7 Nov 2026), both off by default = the submitted architecture:
    # pair_features: add pairwise structural features of (event context, candidate) to the
    #   score (co-occurrence counts from the training events and O2O adjacency), as in
    #   link-prediction models that combine node embeddings with pairwise heuristics.
    # edge_dropout: in every training epoch, additionally hide this share of E2O edges from
    #   message passing, so the model learns to cope with incomplete context.
    pair_features: bool = False
    edge_dropout: float = 0.0
    # Revision (14 Nov 2026): a sixth pairwise feature, temporal proximity between the event
    #   and the candidate's nearest observed event: 1 / (1 + log(1 + hours)), 0 if none.
    time_feature: bool = False
    # (14 Nov) seventh feature: relative time rank, 1/(1+log(1+#observed event-object pairs closer in time
    #   to the event than the candidate's nearest event)), the query event's own pairs left out
    time_rank: bool = False
    # (15 Nov) share of the same-type negatives drawn among objects whose events are close in time
    #   to the query event (temporally hard negatives; requires time_feature). 0 = off.
    time_negatives: float = 0.0
    # Ablation: score = g(pairwise features) only, no GNN representations (implies pair_features).
    features_only: bool = False


class GnnModelAdapter:
    """PyTorch Geometric implementation of the heterogeneous GraphSAGE GNN."""

    def __init__(self, config: GnnConfig):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch_geometric.data import HeteroData
            from torch_geometric.nn import HeteroConv, SAGEConv
        except ImportError as exc:
            raise RuntimeError(
                "The GNN predictor requires torch and torch-geometric. "
                "Install them in the active Python environment."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.F = F
        self.HeteroData = HeteroData
        self.HeteroConv = HeteroConv
        self.SAGEConv = SAGEConv
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fit_seconds = 0.0
        self.inference_seconds = 0.0
        self._fitted = None  # (model, event_map, object_map, object_type_tensor) once trained/loaded
        self._pair_tables = None  # (co-occurrence rows, occurrence counts, O2O neighbours)
        self._time_index = None  # temporal proximity tables (time_feature)

    def fit(
        self,
        corruption: CorruptionResult,
    ) -> tuple[object, dict[str, int], dict[str, int], object]:
        torch = self.torch
        if self.config.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        log = corruption.training
        event_map = {event.id: index for index, event in enumerate(log.events)}
        candidate_ids = corruption.candidate_object_ids
        model_objects = [obj for obj in log.objects if obj.id in candidate_ids]
        object_map = {obj.id: index for index, obj in enumerate(model_objects)}
        inverse_event_map = {index: identifier for identifier, index in event_map.items()}
        inverse_object_map = {index: identifier for identifier, index in object_map.items()}
        event_types = {
            event_type: index
            for index, event_type in enumerate(dict.fromkeys(event.type for event in log.events))
        }
        object_types = {
            object_type: index
            for index, object_type in enumerate(
                dict.fromkeys(obj.type for obj in model_objects)
            )
        }

        event_type_tensor = torch.tensor(
            [event_types[event.type] for event in log.events],
            dtype=torch.long,
            device=self.device,
        )
        object_type_tensor = torch.tensor(
            [object_types[obj.type] for obj in model_objects],
            dtype=torch.long,
            device=self.device,
        )
        _, relation_names = self._build_graph(log, event_map, object_map)
        training_log = corruption.statistics_training
        positives_by_event = self._training_pairs_by_event(
            training_log, event_map, object_map
        )
        observed_by_event = self._event_objects(training_log)
        if self.config.pair_features:
            self._build_pair_tables(training_log, object_map)
        if self.config.time_rank and not self.config.time_feature:
            raise ValueError("time_rank requires time_feature")
        if self.config.time_feature:
            if not self.config.pair_features:
                raise ValueError("time_feature requires pair_features")
            self._build_time_index(corruption, object_map)
        build_training_graph = self._training_graph_builder(log, event_map, object_map)
        dropout_generator = self.torch.Generator().manual_seed(self.config.seed + 1)
        logger.info(
            "GNN options: pair_features=%s edge_dropout=%.2f",
            self.config.pair_features, self.config.edge_dropout,
        )
        logger.info(
            "GNN setup: device=%s events=%d objects=%d observed_pairs=%d "
            "held_out_pairs=%d negatives_per_positive=%d epochs=%d hidden_dim=%d",
            self.device, len(event_map), len(object_map),
            sum(len(pairs) for pairs in positives_by_event.values()),
            len(corruption.removed_event_object_links),
            self.config.negatives_per_positive,
            self.config.epochs, self.config.hidden_dim,
        )
        model = self._build_model(
            len(event_types), len(object_map), len(object_types),
            event_type_tensor, object_type_tensor,
            relation_names,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-5
        )
        best_loss = float("inf")
        best_validation_mrr = float("-inf")
        best_epoch = 0
        best_state = None
        type_indices = defaultdict(list)
        for index, type_index in enumerate(object_type_tensor.tolist()):
            type_indices[type_index].append(index)
        all_object_indices = list(range(len(object_map)))
        object_types_np = object_type_tensor.detach().cpu().numpy()
        if self.config.time_negatives > 0 and not self.config.time_feature:
            raise ValueError("time_negatives requires time_feature")

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            sampled_events = random.choices(
                list(positives_by_event),
                k=min(self.config.batch_size, len(positives_by_event)),
            )
            sampled = [
                random.choice(positives_by_event[event_index])
                for event_index in sampled_events
            ]
            removed_queries = {
                (
                    inverse_event_map[event_index],
                    inverse_object_map[object_index],
                )
                for event_index, object_index in sampled
            }
            query_data = build_training_graph(sampled, dropout_generator)
            embeddings = model(query_data)
            pos_events, pos_objects, neg_objects = [], [], []
            pair_feature_rows = []
            for event_index, object_index in sampled:
                object_type_index = object_type_tensor[object_index].item()
                observed_indices = {
                    object_map[identifier]
                    for identifier in observed_by_event.get(
                        inverse_event_map[event_index], ()
                    )
                    if identifier in object_map
                }
                same_count = round(
                    self.config.negatives_per_positive
                    * self.config.same_type_negative_fraction
                )
                cross_count = self.config.negatives_per_positive - same_count
                excluded = {object_index, *observed_indices}
                near = []
                if self.config.time_negatives > 0:
                    near = self._time_near_negatives(
                        inverse_event_map[event_index], object_type_index,
                        round(same_count * self.config.time_negatives), excluded, object_types_np,
                    )
                same_type = near + self._sample_excluding_tensor(
                    type_indices[object_type_index],
                    same_count - len(near),
                    excluded | set(near),
                )
                cross_type = self._sample_excluding_tensor(
                    all_object_indices,
                    cross_count,
                    excluded,
                    forbidden_type=object_type_index,
                    object_type_tensor=object_type_tensor,
                )
                pos_events.append(event_index)
                pos_objects.append(object_index)
                neg_objects.extend(same_type + cross_type)
                if self.config.pair_features:
                    context = [index for index in observed_indices if index != object_index]
                    pair_feature_rows.append(self._pair_feature_rows(
                        context, [object_index, *same_type, *cross_type],
                        leave_out_positive=object_index,
                        query_event=inverse_event_map[event_index],
                    ))
            if not pos_events:
                continue
            event_embeddings = self.F.normalize(
                embeddings["event"][torch.tensor(pos_events, device=self.device)], dim=-1
            )
            positive_embeddings = self.F.normalize(
                embeddings["object"][torch.tensor(pos_objects, device=self.device)], dim=-1
            )
            positive_scores = model.score_pairs(
                event_embeddings, positive_embeddings
            )
            negative_embeddings = self.F.normalize(
                embeddings["object"][
                    torch.tensor(neg_objects, device=self.device)
                ].reshape(
                    len(pos_events), self.config.negatives_per_positive, -1
                ),
                dim=-1,
            )
            negative_scores = model.score_pairs(
                event_embeddings.unsqueeze(1).expand_as(negative_embeddings),
                negative_embeddings,
            )
            logits = torch.cat(
                [positive_scores.unsqueeze(1), negative_scores], dim=1
            ) / self.config.sampled_softmax_temperature
            if self.config.pair_features:
                features = torch.tensor(
                    pair_feature_rows, dtype=torch.float, device=self.device
                )
                logits = logits + model.pair_head(features).squeeze(-1)
                if self.config.features_only:
                    logits = model.pair_head(features).squeeze(-1)
            loss = self.F.cross_entropy(
                logits, torch.zeros(len(pos_events), dtype=torch.long, device=self.device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            validation_mrr = None
            should_validate = (
                corruption.validation is not None
                and corruption.validation_removed_event_object_links
                and (
                    epoch == 1
                    or epoch % self.config.validation_interval == 0
                    or epoch == self.config.epochs
                )
            )
            if should_validate:
                validation_mrr = self._validation_mrr(
                    model,
                    corruption,
                    event_map,
                    object_map,
                    relation_names,
                )
                logger.info(
                    "GNN validation check after epoch %d/%d mrr=%.4f",
                    epoch, self.config.epochs, validation_mrr,
                )
            if validation_mrr is not None and validation_mrr > best_validation_mrr:
                best_validation_mrr = validation_mrr
                best_loss = loss.item()
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            elif (
                corruption.validation is None
                or not corruption.validation_removed_event_object_links
            ) and loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if epoch == 1 or epoch % max(1, self.config.epochs // 10) == 0:
                logger.info(
                    "GNN epoch %d/%d loss=%.5f positive=%.4f negative=%.4f",
                    epoch, self.config.epochs, loss.item(),
                    positive_scores.mean().item(),
                    negative_scores.mean().item(),
                )
        if best_state is None:
            raise RuntimeError("GNN training did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        checkpoint_path = self._save_checkpoint(
            model,
            best_epoch,
            best_loss,
            best_validation_mrr,
            event_map,
            object_map,
            event_types,
            object_types,
            relation_names,
            event_type_tensor,
            object_type_tensor,
        )
        logger.info(
            "Saved best GNN checkpoint: path=%s epoch=%d loss=%.5f validation_mrr=%.4f",
            checkpoint_path, best_epoch, best_loss, best_validation_mrr,
        )
        return model, event_map, object_map, object_type_tensor

    def _save_checkpoint(
        self,
        model: object,
        best_epoch: int,
        best_loss: float,
        best_validation_mrr: float,
        event_map: dict[str, int],
        object_map: dict[str, int],
        event_types: dict[str, int],
        object_types: dict[str, int],
        relation_names: tuple[str, ...],
        event_type_tensor: object,
        object_type_tensor: object,
    ) -> Path:
        checkpoint_dir = Path(self.config.model_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"gnn_best_{timestamp}.pt"
        self.torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(self.config),
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "best_validation_mrr": best_validation_mrr,
                "event_map": event_map,
                "object_map": object_map,
                "event_types": event_types,
                "object_types": object_types,
                "relation_names": relation_names,
                "event_type_tensor": event_type_tensor.detach().cpu(),
                "object_type_tensor": object_type_tensor.detach().cpu(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load(
        self, checkpoint_path: str
    ) -> tuple[object, dict[str, int], dict[str, int], object]:
        """Rebuilds a trained model purely from a saved checkpoint, without
        retraining or requiring the original corruption/log data. Only valid
        for inference against the same dataset and split the checkpoint was
        trained on, since event/object identities are keyed by string id.
        """
        checkpoint = self.torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        event_map = checkpoint["event_map"]
        object_map = checkpoint["object_map"]
        event_types = checkpoint["event_types"]
        object_types = checkpoint["object_types"]
        relation_names = checkpoint["relation_names"]
        event_type_tensor = checkpoint["event_type_tensor"].to(self.device)
        object_type_tensor = checkpoint["object_type_tensor"].to(self.device)
        model = self._build_model(
            len(event_types), len(object_map), len(object_types),
            event_type_tensor, object_type_tensor,
            relation_names,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(
            "Loaded GNN checkpoint: path=%s epoch=%s validation_mrr=%s",
            checkpoint_path,
            checkpoint.get("best_epoch", "unknown"),
            checkpoint.get("best_validation_mrr", "unknown"),
        )
        return model, event_map, object_map, object_type_tensor

    def _validation_mrr(
        self,
        model: object,
        corruption: CorruptionResult,
        event_map: dict[str, int],
        object_map: dict[str, int],
        relation_names: tuple[str, ...],
    ) -> float:
        validation_log = corruption.validation
        validation_links = corruption.validation_removed_event_object_links
        if validation_log is None or not validation_links:
            raise ValueError("Validation data is required for validation scoring")
        # Hidden validation links are already absent from validation_log.
        validation_data, _ = self._build_graph(validation_log, event_map, object_map)
        observed = self._event_objects(validation_log)
        test_events = sorted({link.event_id for link in validation_links})
        candidate_neighbors = self._candidate_neighbors(
            corruption.training or validation_log
        )
        model.eval()
        with self.torch.no_grad():
            embeddings = model(validation_data)
            event_embeddings = self.F.normalize(embeddings["event"], dim=-1)
            object_embeddings = self.F.normalize(embeddings["object"], dim=-1)
            scores = model.score_all(
                event_embeddings[
                    [event_map[event_id] for event_id in test_events]
                ],
                object_embeddings,
            )
            if self.config.features_only:
                scores.zero_()
        if self.config.pair_features:
            self._add_pair_scores(model, scores, test_events, observed, object_map)
        reciprocal_ranks = []
        targets = defaultdict(set)
        for link in validation_links:
            targets[link.event_id].add(link.object_id)
        for row, event_id in enumerate(test_events):
            event_scores = scores[row].clone()
            for object_id in observed.get(event_id, ()):
                event_scores[object_map[object_id]] = -self.torch.inf
            if self.config.cooccurrence_candidates:
                candidates = {
                    object_map[object_id]
                    for observed_id in observed.get(event_id, ())
                    for object_id in candidate_neighbors.get(observed_id, ())
                    if object_id in object_map
                }
                if not candidates:
                    continue
                mask = self.torch.zeros(
                    len(object_map), dtype=self.torch.bool, device=self.device
                )
                mask[list(candidates)] = True
                event_scores[~mask] = -self.torch.inf
            for target in targets[event_id]:
                filtered = event_scores.clone()
                for other in targets[event_id] - {target}:
                    filtered[object_map[other]] = -self.torch.inf
                target_score = filtered[object_map[target]]
                better = int((filtered > target_score).sum().item())
                ties = int((filtered == target_score).sum().item())
                reciprocal_ranks.append(expected_rank_metrics(better, ties)[0])
        return (
            sum(reciprocal_ranks) / len(validation_links)
            if reciprocal_ranks else 0.0
        )

    def predict(
        self, corruption: CorruptionResult
    ) -> tuple[Prediction, ...]:
        # Train (or load) once; later calls, e.g. scoring the validation events for the
        # decision rule, reuse the same model.
        if self._fitted is None:
            fit_started = time.perf_counter()
            if self.config.checkpoint_path:
                self._fitted = self.load(self.config.checkpoint_path)
            else:
                self._fitted = self.fit(corruption)
            self.fit_seconds = time.perf_counter() - fit_started
        model, event_map, object_map, _ = self._fitted
        inference_started = time.perf_counter()
        torch = self.torch
        inverse_objects = {index: object_id for object_id, index in object_map.items()}
        observed = self._event_objects(corruption.corrupted)
        candidate_neighbors = self._candidate_neighbors(
            corruption.training or corruption.corrupted
        )
        test_events = {link.event_id for link in corruption.removed_event_object_links}
        logger.info(
            "GNN candidate policy: filter=%s universe=%d",
            "co-occurrence" if self.config.cooccurrence_candidates else "all",
            len(object_map),
        )
        # Hidden test links are already absent from corruption.corrupted.
        evaluation_data, _ = self._build_graph(
            corruption.corrupted,
            event_map,
            object_map,
        )
        model.eval()
        with torch.no_grad():
            embeddings = model(evaluation_data)
            event_embeddings = self.F.normalize(embeddings["event"], dim=-1)
            object_embeddings = self.F.normalize(embeddings["object"], dim=-1)
            test_event_indices = sorted(event_map[event_id] for event_id in test_events)
            scores = model.score_all(
                event_embeddings[test_event_indices], object_embeddings
            )
            if self.config.features_only:
                scores.zero_()
            if self.config.pair_features:
                if self._pair_tables is None:  # model loaded from a checkpoint
                    self._build_pair_tables(corruption.statistics_training, object_map)
                if self.config.time_feature and self._time_index is None:
                    self._build_time_index(corruption, object_map)
                inverse_events = {index: identifier for identifier, index in event_map.items()}
                self._add_pair_scores(
                    model, scores,
                    [inverse_events[index] for index in test_event_indices],
                    observed, object_map,
                )
            scores_by_event = {
                event_index: scores[position]
                for position, event_index in enumerate(test_event_indices)
            }

        predictions = []
        logger.info("Scoring %d held-out events with top_k=%d", len(test_events), self.config.top_k)
        included_targets = 0
        candidate_sizes = []
        for event_id in sorted(test_events):
            event_scores = scores_by_event[event_map[event_id]].clone()
            for object_id in observed.get(event_id, ()):
                event_scores[object_map[object_id]] = -torch.inf
            if self.config.cooccurrence_candidates:
                candidates = {
                    object_map[object_id]
                    for observed_id in observed.get(event_id, ())
                    for object_id in candidate_neighbors.get(observed_id, ())
                    if object_id in object_map
                }
                target_id = next(
                    link.object_id
                    for link in corruption.removed_event_object_links
                    if link.event_id == event_id
                )
                candidate_sizes.append(len(candidates))
                included_targets += int(target_id in candidates)
                if not candidates:
                    logger.warning(
                        "No co-occurrence candidates for event=%s; counting as retrieval failure",
                        event_id,
                    )
                    continue
                candidate_mask = torch.zeros(
                    len(object_map), dtype=torch.bool, device=self.device
                )
                candidate_mask[list(candidates)] = True
                event_scores[~candidate_mask] = -torch.inf
            top_values, top_indices = torch.topk(
                event_scores, k=min(self.config.top_k, event_scores.numel())
            )
            # One device-to-host copy per event instead of one per candidate.
            for rank, (value, index) in enumerate(
                zip(top_values.tolist(), top_indices.tolist()), 1
            ):
                if not math.isfinite(value):
                    break  # observed or filtered objects are not candidates
                predictions.append(
                    Prediction(event_id, inverse_objects[index], float(value), rank)
                )
        if self.config.cooccurrence_candidates and test_events:
            logger.info(
                "GNN candidate retrieval: recall=%.4f average_size=%.1f",
                included_targets / len(test_events),
                sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0,
            )
        self.inference_seconds = time.perf_counter() - inference_started
        return tuple(predictions)

    # ---- pairwise structural features (revision) ---------------------------------------
    @property
    def pair_dim(self) -> int:
        return PAIR_FEATURES + (1 if self.config.time_feature else 0) + (1 if self.config.time_rank else 0)

    def _build_time_index(self, corruption: CorruptionResult, object_map: dict[str, int]) -> None:
        """Event times and, per object, the times of its events in the OBSERVED log: the full
        log without the hidden test and validation links (all events, also one-object events)."""
        import numpy as np
        import pandas as pd
        log = corruption.original or corruption.corrupted
        events = list(log.events)
        external = getattr(self, "event_times", None)  # {event id: timestamp}, set by the caller
        raw = [(external.get(e.id) if external else None)
               or (e.attributes.get("ocel_time") if e.attributes else None) for e in events]
        if any(v is None for v in raw):
            raise ValueError("time_feature needs a timestamp for every event")
        hours = pd.to_datetime(pd.Series(raw), utc=True, format="mixed").astype("int64").to_numpy() / 3.6e12
        event_index = {e.id: i for i, e in enumerate(events)}
        hidden = {(l.event_id, l.object_id) for l in corruption.removed_event_object_links}
        hidden |= {(l.event_id, l.object_id) for l in (corruption.validation_removed_event_object_links or ())}
        pairs = sorted(
            (object_map[l.object_id], event_index[l.event_id])
            for l in log.event_object_links
            if l.object_id in object_map and (l.event_id, l.object_id) not in hidden
        )
        po = np.array([p[0] for p in pairs], dtype=np.int64)
        pe = np.array([p[1] for p in pairs], dtype=np.int64)
        pt = hours[pe]
        has = np.zeros(len(object_map), dtype=bool)
        has[po] = True
        starts = np.searchsorted(po, np.arange(len(object_map)))
        ends = np.searchsorted(po, np.arange(len(object_map)), side="right")
        counts = ends - starts
        width = max(1, int(counts.max()) if len(counts) else 1)
        mat_t = np.full((len(object_map), width), np.inf)
        mat_e = np.full((len(object_map), width), -1, dtype=np.int64)
        column = np.arange(len(po)) - starts[po]
        mat_t[po, column] = pt
        mat_e[po, column] = pe
        self._time_index = {"hours": hours, "event_index": event_index, "po": po, "pe": pe,
                            "pt": pt, "has": has, "starts": starts, "ends": ends,
                            "mat_t": mat_t, "mat_e": mat_e, "sorted_t": np.sort(pt),
                            "links_at": np.bincount(pe, minlength=len(events)),
                            "time_order_t": pt[np.argsort(pt, kind="stable")],
                            "time_order_o": po[np.argsort(pt, kind="stable")]}
        logger.info("Time feature: %d observed (object, event) pairs, %d objects with events",
                    len(po), int(has.sum()))

    @staticmethod
    def _time_value(hours_gap):
        import numpy as np
        return np.where(np.isfinite(hours_gap), 1.0 / (1.0 + np.log1p(hours_gap)), 0.0)

    def _time_features_one(self, event_id: str, candidates: list[int]) -> list[float]:
        """Training: temporal feature per candidate, leaving the query event itself out."""
        import numpy as np
        ti = self._time_index
        e = ti["event_index"][event_id]
        index = np.asarray(candidates, dtype=np.int64)
        gaps = np.abs(ti["mat_t"][index] - ti["hours"][e])
        gaps[ti["mat_e"][index] == e] = np.inf
        gap = gaps.min(axis=1)
        if self.config.time_rank:
            return np.stack([self._time_value(gap), self._rank_value(e, gap)], axis=1).tolist()
        return self._time_value(gap)[:, None].tolist()

    def _time_near_negatives(self, event_id: str, type_index: int, count: int, excluded: set,
                             object_types) -> list[int]:
        """`count` objects of the given type whose observed events are among the closest in time
        to the query event (pool of the 4*count nearest distinct objects, sampled uniformly)."""
        import numpy as np
        if count <= 0:
            return []
        ti = self._time_index
        t = ti["hours"][ti["event_index"][event_id]]
        times, objs = ti["time_order_t"], ti["time_order_o"]
        centre = int(np.searchsorted(times, t))
        left, right = centre - 1, centre
        pool, seen = [], set(excluded)
        steps = 0
        while len(pool) < 4 * count and (left >= 0 or right < len(times)) and steps < 20000:
            steps += 1
            if right >= len(times) or (left >= 0 and t - times[left] <= times[right] - t):
                o = int(objs[left]); left -= 1
            else:
                o = int(objs[right]); right += 1
            if o not in seen and object_types[o] == type_index:
                seen.add(o)
                pool.append(o)
        return random.sample(pool, min(count, len(pool)))

    def _rank_value(self, e: int, gaps):
        """Relative rank: observed pairs strictly closer in time than `gaps`, minus the event's own pairs."""
        import numpy as np
        ti = self._time_index
        t = ti["hours"][e]
        finite = np.isfinite(gaps)
        g = np.where(finite, gaps, 0.0)
        lo = np.searchsorted(ti["sorted_t"], t - g, side="right")
        hi = np.searchsorted(ti["sorted_t"], t + g, side="left")
        closer = np.maximum(hi - lo - np.where(g > 0, ti["links_at"][e], 0), 0)
        return np.where(finite, 1.0 / (1.0 + np.log1p(closer)), 0.0)

    def _time_features_all(self, event_id: str):
        """Scoring: temporal feature of every object for one event (hidden links are absent)."""
        import numpy as np
        ti = self._time_index
        t = ti["hours"][ti["event_index"][event_id]]
        gap = np.full(len(ti["has"]), np.inf)
        idx = np.where(ti["has"])[0]
        if len(idx):
            gap[idx] = np.minimum.reduceat(np.abs(ti["pt"] - t), ti["starts"][idx])
        if self.config.time_rank:
            return np.stack([self._time_value(gap), self._rank_value(ti["event_index"][event_id], gap)], axis=1)
        return self._time_value(gap)[:, None]

    def _build_pair_tables(self, training_log: OcelLog, object_map: dict[str, int]) -> None:
        """Co-occurrence counts between objects over the TRAINING events, how many training
        events each object occurs in, and O2O neighbours (O2O relations are observed context)."""
        size = len(object_map)
        cooccurrence = [dict() for _ in range(size)]
        occurrences = [0] * size
        for objects in self._event_objects(training_log).values():
            indices = [object_map[o] for o in objects if o in object_map]
            for a in indices:
                occurrences[a] += 1
                row = cooccurrence[a]
                for b in indices:
                    if b != a:
                        row[b] = row.get(b, 0) + 1
        neighbours = [set() for _ in range(size)]
        for link in training_log.object_object_links:
            if link.source_id in object_map and link.target_id in object_map:
                a, b = object_map[link.source_id], object_map[link.target_id]
                neighbours[a].add(b)
                neighbours[b].add(a)
        self._pair_tables = (cooccurrence, occurrences, neighbours)

    def _pair_feature_rows(
        self, context: list[int], candidates: list[int], leave_out_positive: int | None = None,
        query_event: str | None = None,
    ) -> list[list[float]]:
        """Features of (context, candidate) pairs for training queries. The query event is a
        training event, so its own contribution to the counts is removed (leave-one-out):
        otherwise the positive would always look like it co-occurs with the context."""
        cooccurrence, occurrences, neighbours = self._pair_tables
        m = len(context)
        rows = []
        for candidate in candidates:
            if m == 0:
                rows.append([0.0] * PAIR_FEATURES)
                continue
            total = linked = 0
            probability = 0.0
            o2o = 0
            for c in context:
                count = cooccurrence[c].get(candidate, 0)
                n = occurrences[c]
                if leave_out_positive is not None:
                    n -= 1
                    if candidate == leave_out_positive:
                        count -= 1
                if count > 0:
                    total += count
                    linked += 1
                    probability += count / n if n > 0 else 0.0
                if candidate in neighbours[c]:
                    o2o += 1
            rows.append(_pair_vector(total, linked, probability, o2o, m))
        if self.config.time_feature:
            if query_event is None:
                raise ValueError("time_feature needs the query event")
            for row, values in zip(rows, self._time_features_one(query_event, candidates)):
                row.extend(values)
        return rows

    def _add_pair_scores(
        self, model, scores, event_ids: list[str], observed: dict[str, set[str]],
        object_map: dict[str, int],
    ) -> None:
        """Adds temperature * (head(features) - head(0)) to the cosine scores, in place.
        Candidates without any co-occurrence or O2O link have all-zero features and keep
        their cosine score; the shift by head(0) does not change any ranking."""
        torch = self.torch
        cooccurrence, occurrences, neighbours = self._pair_tables
        tau = self.config.sampled_softmax_temperature
        with torch.no_grad():
            zero = model.pair_head(torch.zeros(1, self.pair_dim, device=self.device)).item()
            if self.config.time_feature:
                self._add_pair_scores_dense(model, scores, event_ids, observed, object_map, zero)
                return
            for row, event_id in enumerate(event_ids):
                context = [object_map[o] for o in observed.get(event_id, ()) if o in object_map]
                m = len(context)
                if m == 0:
                    continue
                acc: dict[int, list] = {}
                for c in context:
                    n = occurrences[c]
                    for candidate, count in cooccurrence[c].items():
                        a = acc.setdefault(candidate, [0, 0, 0.0, 0])
                        a[0] += count
                        a[1] += 1
                        a[2] += count / n if n > 0 else 0.0
                    for candidate in neighbours[c]:
                        acc.setdefault(candidate, [0, 0, 0.0, 0])[3] += 1
                if not acc:
                    continue
                index = torch.tensor(list(acc), dtype=torch.long, device=self.device)
                features = torch.tensor(
                    [_pair_vector(a[0], a[1], a[2], a[3], m) for a in acc.values()],
                    dtype=torch.float, device=self.device,
                )
                scores[row, index] += tau * (model.pair_head(features).squeeze(-1) - zero)

    def _add_pair_scores_dense(self, model, scores, event_ids, observed, object_map, zero) -> None:
        """time_feature: every candidate has a (temporal) feature, so g runs on all objects."""
        import numpy as np
        torch = self.torch
        cooccurrence, occurrences, neighbours = self._pair_tables
        tau = self.config.sampled_softmax_temperature
        size = scores.shape[1]
        for row, event_id in enumerate(event_ids):
            features = np.zeros((size, self.pair_dim), dtype=np.float32)
            features[:, PAIR_FEATURES:] = self._time_features_all(event_id)
            context = [object_map[o] for o in observed.get(event_id, ()) if o in object_map]
            m = len(context)
            if m:
                acc: dict[int, list] = {}
                for c in context:
                    n = occurrences[c]
                    for candidate, count in cooccurrence[c].items():
                        a = acc.setdefault(candidate, [0, 0, 0.0, 0])
                        a[0] += count
                        a[1] += 1
                        a[2] += count / n if n > 0 else 0.0
                    for candidate in neighbours[c]:
                        acc.setdefault(candidate, [0, 0, 0.0, 0])[3] += 1
                if acc:
                    index = np.fromiter(acc, dtype=np.int64)
                    features[index, :PAIR_FEATURES] = np.array(
                        [_pair_vector(a[0], a[1], a[2], a[3], m) for a in acc.values()],
                        dtype=np.float32)
            values = model.pair_head(torch.from_numpy(features).to(self.device)).squeeze(-1)
            scores[row] += tau * (values - zero)

    def _training_graph_builder(self, log: OcelLog, event_map, object_map):
        """Fast per-epoch training graph: the same graph as _build_graph(log, ..., removed),
        built by masking a precomputed edge tensor instead of re-reading every link."""
        torch = self.torch
        pairs = [(event_map[l.event_id], object_map[l.object_id]) for l in log.event_object_links]
        positions: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, pair in enumerate(pairs):
            positions[pair].append(position)
        base = self._edge_index(pairs)
        object_edges = [
            (object_map[l.source_id], object_map[l.target_id])
            for l in log.object_object_links
            if l.source_id in object_map and l.target_id in object_map
        ]
        o2o = self._edge_index(object_edges) if object_edges else None
        edge_count = len(pairs)
        dropout = self.config.edge_dropout

        def build(sampled, generator):
            mask = torch.ones(edge_count, dtype=torch.bool)
            removed = [p for pair in sampled for p in positions.get(tuple(pair), ())]
            if removed:
                mask[torch.tensor(removed, dtype=torch.long)] = False
            if dropout > 0:
                mask &= torch.rand(edge_count, generator=generator) >= dropout
            edges = base[:, mask.to(base.device)]
            graph = self.HeteroData()
            graph["event"].num_nodes = len(event_map)
            graph["object"].num_nodes = len(object_map)
            graph["event", "has", "object"].edge_index = edges
            graph["object", "in", "event"].edge_index = edges.flip(0)
            if o2o is not None:
                graph["object", "related", "object"].edge_index = o2o
            return graph.to(self.device)

        return build

    @staticmethod
    def _candidate_neighbors(log: OcelLog) -> dict[str, set[str]]:
        neighbors: dict[str, set[str]] = defaultdict(set)
        for event_objects in GnnModelAdapter._event_objects(log).values():
            for source in event_objects:
                neighbors[source].update(
                    target for target in event_objects if target != source
                )
        return neighbors

    def evaluate(
        self, corruption: CorruptionResult, predictions: tuple[Prediction, ...]
    ) -> EvaluationResult:
        return evaluate_rankings("gnn", corruption, predictions)

    def _build_model(
        self,
        event_type_count: int,
        object_count: int,
        object_type_count: int,
        event_type_tensor: object,
        object_type_tensor: object,
        relation_names: tuple[str, ...],
    ) -> object:
        torch = self.torch
        nn = self.nn
        F = self.F
        HeteroConv = self.HeteroConv
        SAGEConv = self.SAGEConv
        hidden_dim = self.config.hidden_dim
        pair_dim = self.pair_dim

        class HeteroGnn(nn.Module):
            def __init__(inner_self):
                super().__init__()
                inner_self.event_type = nn.Embedding(event_type_count, hidden_dim)
                inner_self.object_type = nn.Embedding(object_type_count, hidden_dim)
                inner_self.object_id = nn.Embedding(object_count, hidden_dim)

                def make_conv():
                    relations = {}
                    for relation_name in relation_names:
                        source, relation, target = relation_name.split("|")
                        relations[(source, relation, target)] = SAGEConv(
                            (-1, -1), hidden_dim, aggr=self.config.sage_aggregation
                        )
                    return HeteroConv(relations, aggr="sum")

                inner_self.conv1 = make_conv()
                inner_self.conv2 = make_conv()
                inner_self.conv3 = make_conv()
                inner_self.dropout = nn.Dropout(0.3)
                if self.config.pair_features:
                    inner_self.pair_head = nn.Sequential(
                        nn.Linear(pair_dim, 16), nn.ReLU(), nn.Linear(16, 1)
                    )
                    # start as the plain model: the head's output is constant at first
                    nn.init.zeros_(inner_self.pair_head[2].weight)
            def forward(inner_self, graph):
                object_values = (
                    inner_self.object_id.weight
                    + inner_self.object_type(object_type_tensor)
                )
                event_values = inner_self.event_type(event_type_tensor)
                values = {
                    "event": event_values,
                    "object": object_values,
                }
                for conv in (inner_self.conv1, inner_self.conv2):
                    values = conv(values, graph.edge_index_dict)
                    values = {
                        key: inner_self.dropout(F.relu(value))
                        for key, value in values.items()
                    }
                values = inner_self.conv3(values, graph.edge_index_dict)
                return {key: F.relu(value) for key, value in values.items()}

            def score_pairs(inner_self, event_values, object_values):
                event_values = F.normalize(event_values, dim=-1)
                object_values = F.normalize(object_values, dim=-1)
                return (event_values * object_values).sum(dim=-1)

            def score_all(inner_self, event_values, object_values, chunk_size=256):
                scores = []
                for start in range(0, object_values.size(0), chunk_size):
                    objects = object_values[start:start + chunk_size]
                    events = event_values.unsqueeze(1).expand(
                        -1, objects.size(0), -1
                    )
                    candidates = objects.unsqueeze(0).expand(
                        event_values.size(0), -1, -1
                    )
                    scores.append(
                        inner_self.score_pairs(events, candidates).reshape(
                            event_values.size(0), -1
                        )
                    )
                return torch.cat(scores, dim=1)

        return HeteroGnn()

    def _build_graph(
        self,
        log: OcelLog,
        event_map: dict[str, int],
        object_map: dict[str, int],
        removed_queries: set[tuple[str, str]] | None = None,
    ) -> tuple[object, tuple[str, ...]]:
        torch = self.torch
        graph = self.HeteroData()
        graph["event"].num_nodes = len(event_map)
        graph["object"].num_nodes = len(object_map)
        event_edges: list[tuple[int, int]] = []
        for link in log.event_object_links:
            if removed_queries and (link.event_id, link.object_id) in removed_queries:
                continue
            event_edges.append((event_map[link.event_id], object_map[link.object_id]))
        # Object-to-object edges are observed context and are never masked.
        # (Masking the O2O edges of the query objects leaked the targets' identity.)
        object_edges: list[tuple[int, int]] = []
        for link in log.object_object_links:
            if link.source_id not in object_map or link.target_id not in object_map:
                continue
            object_edges.append(
                (object_map[link.source_id], object_map[link.target_id])
            )

        graph["event", "has", "object"].edge_index = self._edge_index(event_edges)
        graph["object", "in", "event"].edge_index = self._edge_index(
            [(target, source) for source, target in event_edges]
        )
        relation_names = ["event|has|object", "object|in|event"]
        # Logs without object-to-object links get no "related" relation
        # (an empty edge list cannot be turned into a 2 x E edge index).
        if object_edges:
            graph["object", "related", "object"].edge_index = self._edge_index(object_edges)
            relation_names.append("object|related|object")
        return graph.to(self.device), tuple(relation_names)

    def _edge_index(self, edges: list[tuple[int, int]]) -> object:
        return self.torch.tensor(
            edges, dtype=self.torch.long, device=self.device
        ).T.contiguous()

    @staticmethod
    def _sample_excluding_tensor(
        pool: object,
        count: int,
        excluded: set[int],
        forbidden_type: int | None = None,
        object_type_tensor: object | None = None,
    ) -> list[int]:
        if count == 0:
            return []
        values = pool if isinstance(pool, list) else pool.tolist()
        selected: list[int] = []
        selected_set = set(excluded)
        attempts = 0
        max_attempts = max(100, count * 30)
        while len(selected) < count and attempts < max_attempts:
            candidate = values[random.randrange(len(values))]
            candidate_type = (
                object_type_tensor[candidate].item()
                if forbidden_type is not None
                else None
            )
            if (
                candidate not in selected_set
                and candidate_type != forbidden_type
            ):
                selected.append(candidate)
                selected_set.add(candidate)
            attempts += 1
        if len(selected) < count:
            available = [
                candidate
                for candidate in values
                if candidate not in selected_set
                and (
                    forbidden_type is None
                    or object_type_tensor[candidate].item() != forbidden_type
                )
            ]
            if len(available) < count - len(selected):
                raise ValueError("Insufficient candidates for negative sampling")
            selected.extend(random.sample(available, count - len(selected)))
        return selected

    @staticmethod
    def _relation_name(qualifier: str | None, direction: str) -> str:
        value = qualifier or "unspecified"
        normalized = "".join(
            character if character.isalnum() else "_"
            for character in value.lower()
        ).strip("_")
        return f"{direction}_{normalized or 'unspecified'}"

    @staticmethod
    def _training_pairs_by_event(
        log: OcelLog, event_map: dict[str, int], object_map: dict[str, int]
    ) -> dict[int, list[tuple[int, int]]]:
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for link in log.event_object_links:
            if link.event_id in event_map and link.object_id in object_map:
                grouped[event_map[link.event_id]].append(
                    (event_map[link.event_id], object_map[link.object_id])
                )
        return grouped

    @staticmethod
    def _event_objects(log: OcelLog) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        for link in log.event_object_links:
            grouped[link.event_id].add(link.object_id)
        return grouped
