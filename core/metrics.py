"""Ranking metrics shared by all predictors.

All metrics are expected values under random tie-breaking, in the filtered
setting: when an event has several hidden objects, the other hidden objects are
removed from the ranking while one of them is scored.
"""
from collections import defaultdict

import numpy as np

from core.domain import CorruptionResult, EvaluationResult, Prediction

KS = (1, 3, 5, 10)


def expected_rank_metrics(better: int, ties: int, ks=KS):
    """Expected reciprocal rank and P(rank <= k) for a target that has `better`
    candidates scored strictly higher and `ties` candidates (itself included)
    with exactly its score, ordered uniformly at random among the ties."""
    ties = max(1, int(ties))
    better = int(better)
    rr = sum(1.0 / (better + j) for j in range(1, ties + 1)) / ties
    hits = {k: 0.0 if better >= k else min(1.0, (k - better) / ties) for k in ks}
    return rr, hits


def per_query_metrics(corruption: CorruptionResult, predictions: tuple[Prediction, ...]):
    """Per hidden link: (event_id, object_id, expected reciprocal rank, P(hit@1), P(hit@3),
    P(hit@10)), same filtered / random-tie convention as evaluate_rankings. Used for paired
    statistics (bootstrap, McNemar-style comparisons) between methods on the same gaps."""
    truth_by_event: dict[str, set[str]] = defaultdict(set)
    for link in corruption.removed_event_object_links:
        truth_by_event[link.event_id].add(link.object_id)
    scores_by_event: dict[str, dict[str, float]] = defaultdict(dict)
    for prediction in predictions:
        scores_by_event[prediction.event_id][prediction.object_id] = prediction.score
    rows = []
    for event_id in sorted(truth_by_event):
        targets = truth_by_event[event_id]
        scores = scores_by_event.get(event_id, {})
        ids = list(scores)
        values = np.asarray([scores[i] for i in ids], dtype=float)
        for target in sorted(targets):
            if target not in scores:
                rows.append((event_id, target, 0.0, 0.0, 0.0, 0.0))
                continue
            keep = np.array([(i == target) or (i not in targets) for i in ids], dtype=bool)
            t = scores[target]
            better = int(np.sum(values[keep] > t))
            ties = int(np.sum(values[keep] == t))
            rr, hits = expected_rank_metrics(better, ties, (1, 3, 10))
            rows.append((event_id, target, rr, hits[1], hits[3], hits[10]))
    return tuple(rows)


def evaluate_rankings(
    model_name: str,
    corruption: CorruptionResult,
    predictions: tuple[Prediction, ...],
    ks=KS,
) -> EvaluationResult:
    """Scores predictions against the hidden links. `predictions` must contain a
    score for every candidate of an event (not only a top-k list); a hidden
    object without a score counts as a miss."""
    truth_by_event: dict[str, set[str]] = defaultdict(set)
    for link in corruption.removed_event_object_links:
        truth_by_event[link.event_id].add(link.object_id)
    ids_by_event: dict[str, list[str]] = defaultdict(list)
    scores_by_event: dict[str, list[float]] = defaultdict(list)
    for prediction in predictions:
        ids_by_event[prediction.event_id].append(prediction.object_id)
        scores_by_event[prediction.event_id].append(prediction.score)
    rr_sum, n = 0.0, 0
    hits_sum = {k: 0.0 for k in ks}
    for event_id, targets in truth_by_event.items():
        ids = ids_by_event.get(event_id, [])
        scores = np.asarray(scores_by_event.get(event_id, []), dtype=float)
        position = {object_id: i for i, object_id in enumerate(ids)}
        for target in targets:
            n += 1
            if target not in position:
                continue
            keep = np.ones(len(ids), dtype=bool)
            for other in targets - {target}:
                if other in position:
                    keep[position[other]] = False
            target_score = scores[position[target]]
            better = int(np.sum(scores[keep] > target_score))
            ties = int(np.sum(scores[keep] == target_score))
            rr, hits = expected_rank_metrics(better, ties, ks)
            rr_sum += rr
            for k in ks:
                hits_sum[k] += hits[k]
    return EvaluationResult(
        model_name,
        n,
        {k: hits_sum[k] / n if n else 0.0 for k in ks},
        rr_sum / n if n else 0.0,
    )
