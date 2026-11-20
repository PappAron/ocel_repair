"""Scaling experiments for the heterogeneous GraphSAGE model (8 Nov 2026).

Kept separate from adapters/gnn.py: with the default settings, ScalableGnnConfig
gives exactly the submitted model (object_inputs="id", full-graph training, all
objects ranked). Three remedies from claude-notes/gnn-design-scaling.md:

1. object_inputs: what an object node starts with before message passing.
   "id"          learned vector per object ID + object-type vector (submitted model;
                 parameters grow with #objects, transductive)
   "type"        object-type vector only (fixed parameter count, inductive)
   "type+degree" type vector + a linear map of log(1 + degree) per relation, computed
                 from the graph being scored (fixed parameter count, inductive).
                 Events get log(1 + #observed objects) the same way.
2. sampled_training: each step passes only a sampled neighbourhood (fan-out per
   layer) around the step's query events and candidate objects through the model,
   instead of the whole graph. Validation and test scoring stay full-graph.
3. CandidateFilteredPredictor: two-stage repair. Candidates of an event are the
   objects within `hops` object-to-object steps of its observed objects (a step =
   sharing an event in the observed log, or an O2O link); only these are ranked.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass

from adapters.gnn import GnnConfig, GnnModelAdapter, PAIR_FEATURES
from core.domain import CorruptionResult, Prediction
from core.metrics import evaluate_rankings

logger = logging.getLogger(__name__)

OBJECT_INPUTS = ("id", "type", "type+degree")


@dataclass(frozen=True)
class ScalableGnnConfig(GnnConfig):
    object_inputs: str = "id"
    sampled_training: bool = False
    fanouts: tuple[int, ...] = (15, 10, 5)  # per layer, input layer first


class ScalableGnnAdapter(GnnModelAdapter):
    def __init__(self, config: ScalableGnnConfig):
        if config.object_inputs not in OBJECT_INPUTS:
            raise ValueError(f"object_inputs must be one of {OBJECT_INPUTS}")
        if config.sampled_training and getattr(config, 'time_feature', False):
            raise ValueError('time_feature is not implemented for sampled training')
        if config.sampled_training and len(config.fanouts) != 3:
            raise ValueError("fanouts needs one value per GNN layer (3)")
        super().__init__(config)
        # (event type indices, object type indices) of the log being scored; set only when
        # a trained model is applied to another log (inductive transfer)
        self.node_type_override = None

    # ---- graphs carry node ids and degree features ------------------------------------
    @property
    def _uses_degree(self) -> bool:
        return self.config.object_inputs == "type+degree"

    def _degree_features(self, edge_index_dict, num_events, num_objects):
        """log(1 + degree) per relation and direction, from the edges of this graph."""
        torch = self.torch
        has = edge_index_dict[("event", "has", "object")]
        event_x = torch.log1p(
            torch.bincount(has[0], minlength=num_events).float()
        ).unsqueeze(1)
        object_cols = [torch.bincount(has[1], minlength=num_objects).float()]
        related = edge_index_dict.get(("object", "related", "object"))
        if related is not None:
            object_cols.append(torch.bincount(related[0], minlength=num_objects).float())
            object_cols.append(torch.bincount(related[1], minlength=num_objects).float())
        else:
            object_cols += [torch.zeros(num_objects, device=has.device), torch.zeros(num_objects, device=has.device)]
        object_x = torch.log1p(torch.stack(object_cols, dim=1))
        return event_x.to(self.device), object_x.to(self.device)

    def _decorate(self, graph):
        torch = self.torch
        ne, no = graph["event"].num_nodes, graph["object"].num_nodes
        graph["event"].n_id = torch.arange(ne, device=self.device)
        graph["object"].n_id = torch.arange(no, device=self.device)
        if self.node_type_override is not None:
            graph["event"].t, graph["object"].t = self.node_type_override
        if self._uses_degree:
            graph["event"].x, graph["object"].x = self._degree_features(
                graph.edge_index_dict, ne, no
            )
        return graph

    def _build_graph(self, log, event_map, object_map, removed_queries=None):
        graph, relations = super()._build_graph(log, event_map, object_map, removed_queries)
        return self._decorate(graph), relations

    def _training_graph_builder(self, log, event_map, object_map):
        build = super()._training_graph_builder(log, event_map, object_map)
        return lambda sampled, generator: self._decorate(build(sampled, generator))

    # ---- model ------------------------------------------------------------------------
    def _build_model(self, event_type_count, object_count, object_type_count,
                     event_type_tensor, object_type_tensor, relation_names):
        nn, F = self.nn, self.F
        HeteroConv, SAGEConv = self.HeteroConv, self.SAGEConv
        hidden_dim = self.config.hidden_dim
        pair_dim = self.pair_dim
        config = self.config

        class HeteroGnn(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.event_type = nn.Embedding(event_type_count, hidden_dim)
                inner.object_type = nn.Embedding(object_type_count, hidden_dim)
                if config.object_inputs == "id":
                    inner.object_id = nn.Embedding(object_count, hidden_dim)
                if config.object_inputs == "type+degree":
                    inner.event_degree = nn.Linear(1, hidden_dim)
                    inner.object_degree = nn.Linear(3, hidden_dim)

                def make_conv():
                    relations = {}
                    for name in relation_names:
                        source, relation, target = name.split("|")
                        relations[(source, relation, target)] = SAGEConv(
                            (-1, -1), hidden_dim, aggr=config.sage_aggregation
                        )
                    return HeteroConv(relations, aggr="sum")

                inner.conv1, inner.conv2, inner.conv3 = make_conv(), make_conv(), make_conv()
                inner.dropout = nn.Dropout(0.3)
                if config.pair_features:
                    inner.pair_head = nn.Sequential(
                        nn.Linear(pair_dim, 16), nn.ReLU(), nn.Linear(16, 1)
                    )
                    nn.init.zeros_(inner.pair_head[2].weight)

            def forward(inner, graph):
                event_ids = graph["event"].n_id
                object_ids = graph["object"].n_id
                object_t = getattr(graph["object"], "t", None)
                event_t = getattr(graph["event"], "t", None)
                object_values = inner.object_type(
                    object_type_tensor[object_ids] if object_t is None else object_t)
                event_values = inner.event_type(
                    event_type_tensor[event_ids] if event_t is None else event_t)
                if config.object_inputs == "id":
                    object_values = object_values + inner.object_id(object_ids)
                if config.object_inputs == "type+degree":
                    object_values = object_values + inner.object_degree(graph["object"].x)
                    event_values = event_values + inner.event_degree(graph["event"].x)
                values = {"event": event_values, "object": object_values}
                for conv in (inner.conv1, inner.conv2):
                    values = conv(values, graph.edge_index_dict)
                    values = {k: inner.dropout(F.relu(v)) for k, v in values.items()}
                values = inner.conv3(values, graph.edge_index_dict)
                return {k: F.relu(v) for k, v in values.items()}

            def score_pairs(inner, event_values, object_values):
                event_values = F.normalize(event_values, dim=-1)
                object_values = F.normalize(object_values, dim=-1)
                return (event_values * object_values).sum(dim=-1)

            def score_all(inner, event_values, object_values, chunk_size=256):
                event_values = F.normalize(event_values, dim=-1)
                object_values = F.normalize(object_values, dim=-1)
                return event_values @ object_values.T

        return HeteroGnn()

    # ---- training ---------------------------------------------------------------------
    def fit(self, corruption: CorruptionResult):
        if not self.config.sampled_training:
            return super().fit(corruption)
        return self._fit_sampled(corruption)

    def _fit_sampled(self, corruption: CorruptionResult):
        """Same queries, negatives, loss, validation and checkpointing as
        GnnModelAdapter.fit, but every step runs on a sampled subgraph."""
        torch = self.torch
        config = self.config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        log = corruption.training
        event_map = {event.id: i for i, event in enumerate(log.events)}
        candidate_ids = corruption.candidate_object_ids
        model_objects = [obj for obj in log.objects if obj.id in candidate_ids]
        object_map = {obj.id: i for i, obj in enumerate(model_objects)}
        inverse_event_map = {i: e for e, i in event_map.items()}
        event_types = {t: i for i, t in enumerate(dict.fromkeys(e.type for e in log.events))}
        object_types = {t: i for i, t in enumerate(dict.fromkeys(o.type for o in model_objects))}
        event_type_tensor = torch.tensor([event_types[e.type] for e in log.events], device=self.device)
        object_type_tensor = torch.tensor([object_types[o.type] for o in model_objects], device=self.device)
        _, relation_names = super()._build_graph(log, event_map, object_map)
        positives_by_event = self._training_pairs_by_event(corruption.statistics_training, event_map, object_map)
        observed_by_event = self._event_objects(corruption.statistics_training)
        if config.pair_features:
            self._build_pair_tables(corruption.statistics_training, object_map)
        sampler = NeighbourSampler(self, log, event_map, object_map, config.fanouts)
        model = self._build_model(
            len(event_types), len(object_map), len(object_types),
            event_type_tensor, object_type_tensor, relation_names,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
        type_indices = defaultdict(list)
        for index, type_index in enumerate(object_type_tensor.tolist()):
            type_indices[type_index].append(index)
        all_object_indices = list(range(len(object_map)))
        best = (float("-inf"), float("inf"), 0, None)  # mrr, loss, epoch, state
        logger.info("GNN sampled training: fanouts=%s object_inputs=%s",
                    config.fanouts, config.object_inputs)
        subgraph_sizes = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            sampled_events = random.choices(
                list(positives_by_event), k=min(config.batch_size, len(positives_by_event)))
            sampled = [random.choice(positives_by_event[e]) for e in sampled_events]
            pos_events, pos_objects, neg_objects, pair_rows = [], [], [], []
            same_count = round(config.negatives_per_positive * config.same_type_negative_fraction)
            cross_count = config.negatives_per_positive - same_count
            for event_index, object_index in sampled:
                type_index = object_type_tensor[object_index].item()
                observed = {object_map[o] for o in observed_by_event.get(inverse_event_map[event_index], ())
                            if o in object_map}
                excluded = {object_index, *observed}
                same = self._sample_excluding_tensor(type_indices[type_index], same_count, excluded)
                cross = self._sample_excluding_tensor(all_object_indices, cross_count, excluded,
                                                      forbidden_type=type_index,
                                                      object_type_tensor=object_type_tensor)
                pos_events.append(event_index)
                pos_objects.append(object_index)
                neg_objects.extend(same + cross)
                if config.pair_features:
                    context = [i for i in observed if i != object_index]
                    pair_rows.append(self._pair_feature_rows(
                        context, [object_index, *same, *cross], leave_out_positive=object_index))
            graph, local_event, local_object = sampler.sample(
                torch.tensor(pos_events), torch.tensor(pos_objects + neg_objects), sampled)
            # the sampler works on the CPU; the model may be on the GPU
            graph = graph.to(self.device)
            local_event, local_object = local_event.to(self.device), local_object.to(self.device)
            subgraph_sizes.append(graph["event"].num_nodes + graph["object"].num_nodes)
            embeddings = model(graph)
            n = len(pos_events)
            event_emb = embeddings["event"][local_event]
            object_emb = embeddings["object"][local_object]
            positive_scores = model.score_pairs(event_emb, object_emb[:n])
            negative_emb = object_emb[n:].reshape(n, config.negatives_per_positive, -1)
            negative_scores = model.score_pairs(event_emb.unsqueeze(1).expand_as(negative_emb), negative_emb)
            logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], 1) / config.sampled_softmax_temperature
            if config.pair_features:
                logits = logits + model.pair_head(torch.tensor(pair_rows, dtype=torch.float, device=self.device)).squeeze(-1)
            loss = self.F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if corruption.validation is not None and (
                epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.epochs):
                mrr = self._validation_mrr(model, corruption, event_map, object_map, relation_names)
                logger.info("GNN validation check after epoch %d/%d mrr=%.4f (mean subgraph nodes %.0f)",
                            epoch, config.epochs, mrr, sum(subgraph_sizes) / len(subgraph_sizes))
                if mrr > best[0]:
                    best = (mrr, loss.item(), epoch,
                            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[3])
        path = self._save_checkpoint(model, best[2], best[1], best[0], event_map, object_map,
                                     event_types, object_types, relation_names,
                                     event_type_tensor, object_type_tensor)
        logger.info("Saved best GNN checkpoint: path=%s epoch=%d validation_mrr=%.4f",
                    path, best[2], best[0])
        return model, event_map, object_map, object_type_tensor


class NeighbourSampler:
    """GraphSAGE-style sampling. Starting from the seed events and objects, each layer
    adds up to `fanout` incoming neighbours per node and relation (all of them when a
    node has fewer). The query links of the step are removed from the sampled edges,
    and degree features use the full training graph minus those query links."""

    def __init__(self, adapter: ScalableGnnAdapter, log, event_map, object_map, fanouts):
        torch = adapter.torch
        self.adapter, self.torch, self.fanouts = adapter, torch, fanouts
        self.ne, self.no = len(event_map), len(object_map)
        pairs = torch.tensor(
            [(event_map[l.event_id], object_map[l.object_id]) for l in log.event_object_links],
            dtype=torch.long).T
        o2o = [(object_map[l.source_id], object_map[l.target_id]) for l in log.object_object_links
               if l.source_id in object_map and l.target_id in object_map]
        self.has_o2o = bool(o2o)
        o2o = torch.tensor(o2o, dtype=torch.long).T if o2o else torch.empty(2, 0, dtype=torch.long)
        # incoming-neighbour lists (CSR by target node) for each relation
        self.csr = {
            "has": self._csr(pairs[0], pairs[1], self.no),      # event -> object
            "in": self._csr(pairs[1], pairs[0], self.ne),       # object -> event
            "related": self._csr(o2o[0], o2o[1], self.no),      # object -> object
        }
        self.event_degree = torch.bincount(pairs[0], minlength=self.ne).float()
        self.object_degree = torch.stack([
            torch.bincount(pairs[1], minlength=self.no).float(),
            torch.bincount(o2o[0], minlength=self.no).float(),
            torch.bincount(o2o[1], minlength=self.no).float(),
        ], 1)

    def _csr(self, source, target, num_targets):
        torch = self.torch
        order = torch.argsort(target, stable=True)
        counts = torch.bincount(target, minlength=num_targets)
        rowptr = torch.zeros(num_targets + 1, dtype=torch.long)
        rowptr[1:] = torch.cumsum(counts, 0)
        return rowptr, source[order]

    def _sample(self, relation, nodes, fanout):
        """Returns (source, target) of up to `fanout` incoming edges per node."""
        torch = self.torch
        rowptr, col = self.csr[relation]
        degree = rowptr[nodes + 1] - rowptr[nodes]
        take = torch.clamp(degree, max=fanout)
        if take.sum() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        target = torch.repeat_interleave(nodes, take)
        node_degree = torch.repeat_interleave(degree, take)
        start = torch.repeat_interleave(rowptr[nodes], take)
        group_start = torch.repeat_interleave(torch.cumsum(take, 0) - take, take)
        position = torch.arange(len(target)) - group_start
        random_offset = (torch.rand(len(target)) * node_degree).long()
        offset = torch.where(node_degree <= fanout, position, random_offset)
        return col[start + offset], target

    def sample(self, seed_events, seed_objects, query_pairs):
        torch = self.torch
        query = torch.tensor(query_pairs, dtype=torch.long).unique(dim=0)
        query_keys = set((query[:, 0] * self.no + query[:, 1]).tolist())
        events, objects = seed_events.unique(), seed_objects.unique()
        frontier_e, frontier_o = events, objects
        edges = {"has": [], "in": [], "related": []}
        for fanout in self.fanouts:
            src_has, dst_has = self._sample("has", frontier_o, fanout)   # events -> objects
            src_in, dst_in = self._sample("in", frontier_e, fanout)      # objects -> events
            edges["has"].append(torch.stack([src_has, dst_has]))
            edges["in"].append(torch.stack([src_in, dst_in]))
            new_e, new_o = [src_has], [src_in]
            if self.has_o2o:
                src_rel, dst_rel = self._sample("related", frontier_o, fanout)
                edges["related"].append(torch.stack([src_rel, dst_rel]))
                new_o.append(src_rel)
            new_e = torch.cat(new_e).unique()
            new_o = torch.cat(new_o).unique()
            frontier_e = new_e[~torch.isin(new_e, events)]
            frontier_o = new_o[~torch.isin(new_o, objects)]
            events = torch.cat([events, frontier_e]).unique()
            objects = torch.cat([objects, frontier_o]).unique()
        # merge, drop duplicates and the query links (event, object) in both directions
        has = torch.cat(edges["has"], 1).unique(dim=1)
        has = has[:, [k not in query_keys for k in (has[0] * self.no + has[1]).tolist()]]
        inn = torch.cat(edges["in"], 1).unique(dim=1)
        inn = inn[:, [k not in query_keys for k in (inn[1] * self.no + inn[0]).tolist()]]
        graph = self.adapter.HeteroData()
        graph["event"].num_nodes, graph["object"].num_nodes = len(events), len(objects)
        graph["event"].n_id, graph["object"].n_id = events, objects
        local = lambda ids, pool: torch.searchsorted(pool, ids)
        graph["event", "has", "object"].edge_index = torch.stack([local(has[0], events), local(has[1], objects)])
        graph["object", "in", "event"].edge_index = torch.stack([local(inn[0], objects), local(inn[1], events)])
        if self.has_o2o:
            rel = torch.cat(edges["related"], 1).unique(dim=1)
            graph["object", "related", "object"].edge_index = torch.stack(
                [local(rel[0], objects), local(rel[1], objects)])
        if self.adapter._uses_degree:
            event_degree = self.event_degree.clone()
            object_degree = self.object_degree.clone()
            event_degree.index_add_(0, query[:, 0], -torch.ones(len(query)))
            object_degree[:, 0].index_add_(0, query[:, 1], -torch.ones(len(query)))
            graph["event"].x = torch.log1p(event_degree[events]).unsqueeze(1)
            graph["object"].x = torch.log1p(object_degree[objects])
        return graph, local(seed_events, events), local(seed_objects, objects)


# ---- two-stage repair ---------------------------------------------------------------
def khop_candidates(log, hops: int, use_o2o: bool = True) -> dict[str, set[str]]:
    """Objects within `hops` object-to-object steps of each event's observed objects
    (a step = sharing an event in `log`, or an O2O link), observed objects excluded."""
    event_objects = defaultdict(set)
    for link in log.event_object_links:
        event_objects[link.event_id].add(link.object_id)
    neighbours = defaultdict(set)
    for objects in event_objects.values():
        for o in objects:
            neighbours[o] |= objects
    if use_o2o:
        for link in log.object_object_links:
            neighbours[link.source_id].add(link.target_id)
            neighbours[link.target_id].add(link.source_id)
    cache = {}

    def reach(o):
        if o not in cache:
            seen, frontier = {o}, {o}
            for _ in range(hops):
                frontier = {n for f in frontier for n in neighbours[f]} - seen
                seen |= frontier
            cache[o] = seen
        return cache[o]

    return {e: set().union(*(reach(o) for o in objs)) - objs for e, objs in event_objects.items()}


class CandidateFilteredPredictor:
    """Reranks only the k-hop candidates of a base predictor's full ranking. The base
    predictor's trained model is reused (no extra training)."""

    def __init__(self, base, hops: int, use_o2o: bool = True, name: str | None = None):
        self.base, self.hops, self.use_o2o = base, hops, use_o2o
        self.name = name or f"two-stage-{hops}hop"
        self._fit_seconds = self._inference_seconds = 0.0
        self.stats = {}

    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        test_events = {l.event_id for l in corruption.removed_event_object_links}
        candidates = khop_candidates(corruption.corrupted, self.hops, self.use_o2o)
        if self.base is None:
            # reference: every candidate gets the same score (= random order among candidates)
            predictions = tuple(Prediction(e, o, 0.0, 1) for e in sorted(test_events)
                                for o in sorted(candidates.get(e, ())))
        else:
            predictions = self.base.predict(corruption)
        started = time.perf_counter()
        truth = defaultdict(set)
        for l in corruption.removed_event_object_links:
            truth[l.event_id].add(l.object_id)
        sizes = [len(candidates.get(e, ())) for e in test_events]
        covered = sum(len(t & candidates.get(e, set())) for e, t in truth.items())
        universe = len(corruption.candidate_object_ids)
        stats = {
            "hops": self.hops, "use_o2o": self.use_o2o,
            "recall": covered / sum(len(t) for t in truth.values()),
            "mean_candidates": sum(sizes) / len(sizes), "median_candidates": sorted(sizes)[len(sizes) // 2],
            "max_candidates": max(sizes), "universe": universe,
        }
        if not self.stats:  # keep the test-split statistics (validation is scored second)
            self.stats = stats
        logger.info("Two-stage %s: candidate recall=%.4f mean size=%.1f (universe %d)",
                    self.name, stats["recall"], stats["mean_candidates"], universe)
        kept, rank = [], defaultdict(int)
        for p in predictions:
            if p.event_id in test_events and p.object_id in candidates.get(p.event_id, ()):
                rank[p.event_id] += 1
                kept.append(Prediction(p.event_id, p.object_id, p.score, rank[p.event_id]))
        base_fit, base_inference = self.base.timing() if self.base is not None else (0.0, 0.0)
        self._fit_seconds = base_fit
        self._inference_seconds = base_inference + time.perf_counter() - started
        return tuple(kept)

    def timing(self):
        return self._fit_seconds, self._inference_seconds

    def evaluate(self, corruption, predictions):
        return evaluate_rankings(self.name, corruption, predictions)


class ScalableGnnPredictor:
    """GnnPredictor with a ScalableGnnAdapter."""

    def __init__(self, config: ScalableGnnConfig):
        from adapters.predictors import GnnPredictor
        self._inner = GnnPredictor.__new__(GnnPredictor)
        self._inner.model = ScalableGnnAdapter(config)
        self._inner._fit_seconds = self._inner._inference_seconds = 0.0
        self.model = self._inner.model

    def train(self, corruption):
        self._inner.train(corruption)

    def predict(self, corruption):
        return self._inner.predict(corruption)

    def timing(self):
        return self._inner.timing()

    def evaluate(self, corruption, predictions):
        return self._inner.evaluate(corruption, predictions)

    def parameter_count(self) -> int | None:
        fitted = self.model._fitted
        return sum(p.numel() for p in fitted[0].parameters()) if fitted else None
