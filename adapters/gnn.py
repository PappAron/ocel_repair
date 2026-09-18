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

logger = logging.getLogger(__name__)

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
        training_log = log
        positives_by_event = self._training_pairs_by_event(
            training_log, event_map, object_map
        )
        observed_by_event = self._event_objects(training_log)
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
            query_data, _ = self._build_graph(
                log, event_map, object_map, removed_queries
            )
            embeddings = model(query_data)
            pos_events, pos_objects, neg_objects = [], [], []
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
                same_type = self._sample_excluding_tensor(
                    type_indices[object_type_index],
                    same_count,
                    excluded,
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
        validation_queries = {
            (link.event_id, link.object_id) for link in validation_links
        }
        validation_data, _ = self._build_graph(
            validation_log, event_map, object_map, validation_queries
        )
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
        ranks = []
        targets = {link.event_id: link.object_id for link in validation_links}
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
            target_index = object_map[targets[event_id]]
            rank = int(
                (event_scores > event_scores[target_index]).sum().item()
            ) + 1
            ranks.append(rank)
        return (
            sum(1.0 / rank for rank in ranks) / len(validation_links)
            if ranks else 0.0
        )

    def predict(
        self, corruption: CorruptionResult
    ) -> tuple[Prediction, ...]:
        fit_started = time.perf_counter()
        if self.config.checkpoint_path:
            model, event_map, object_map, _ = self.load(self.config.checkpoint_path)
        else:
            model, event_map, object_map, _ = self.fit(corruption)
        self.fit_seconds = time.perf_counter() - fit_started
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
        evaluation_queries = {
            (link.event_id, link.object_id)
            for link in corruption.removed_event_object_links
        }
        evaluation_data, _ = self._build_graph(
            corruption.corrupted,
            event_map,
            object_map,
            evaluation_queries,
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
            ranked = torch.argsort(event_scores, descending=True)[: self.config.top_k]
            for rank, index in enumerate(ranked.tolist(), 1):
                predictions.append(
                    Prediction(
                        event_id,
                        inverse_objects[index],
                        float(event_scores[index].item()),
                        rank,
                    )
                )
        if self.config.cooccurrence_candidates and test_events:
            logger.info(
                "GNN candidate retrieval: recall=%.4f average_size=%.1f",
                included_targets / len(test_events),
                sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0,
            )
        self.inference_seconds = time.perf_counter() - inference_started
        return tuple(predictions)

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
        truth = {
            (link.event_id, link.object_id)
            for link in corruption.removed_event_object_links
        }
        by_event: dict[str, list[Prediction]] = defaultdict(list)
        for prediction in predictions:
            by_event[prediction.event_id].append(prediction)
        ranks = [
            prediction.rank
            for event_id, object_id in truth
            for prediction in by_event[event_id]
            if prediction.object_id == object_id
        ]
        correct = len(ranks)
        hits = {
            k: sum(rank <= k for rank in ranks) / len(truth) if truth else 0.0
            for k in (1, 3, 5, 10)
        }
        precision = correct / len(predictions) if predictions else 0.0
        recall = correct / len(truth) if truth else 0.0
        return EvaluationResult(
            "gnn",
            len(truth),
            hits,
            sum(1 / rank for rank in ranks) / len(truth) if truth else 0.0,
            precision,
            recall,
        )

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
        object_edges: list[tuple[int, int]] = []
        masked_object_indices = {
            object_map[object_id]
            for _, object_id in (removed_queries or ())
            if object_id in object_map
        }
        for link in log.object_object_links:
            if link.source_id not in object_map or link.target_id not in object_map:
                continue
            if (
                object_map[link.source_id] in masked_object_indices
                or object_map[link.target_id] in masked_object_indices
            ):
                continue
            object_edges.append(
                (object_map[link.source_id], object_map[link.target_id])
            )

        graph["event", "has", "object"].edge_index = self._edge_index(event_edges)
        graph["object", "in", "event"].edge_index = self._edge_index(
            [(target, source) for source, target in event_edges]
        )
        graph["object", "related", "object"].edge_index = self._edge_index(object_edges)
        relation_names = (
            "event|has|object",
            "object|in|event",
            "object|related|object",
        )
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
