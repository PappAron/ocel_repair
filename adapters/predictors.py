import math
import logging
from collections import Counter, defaultdict
from itertools import combinations

from core.domain import CorruptionResult, EvaluationResult, Prediction
from core.ports import PredictorPort
from adapters.gnn import GnnConfig, GnnModelAdapter

logger = logging.getLogger(__name__)

class CoOccurrenceBaseline(PredictorPort):
    def __init__(self, top_k: int = 10, same_type_candidates: bool = False):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.same_type_candidates = same_type_candidates

    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        counts = Counter()
        pairs = Counter()
        training_log = corruption.training or corruption.corrupted
        for event_id in self._event_objects(training_log).values():
            objects = sorted(event_id)
            for object_id in objects:
                counts[object_id] += 1
            for left, right in combinations(objects, 2):
                pairs[left, right] += 1
                pairs[right, left] += 1

        conditional: dict[str, dict[str, float]] = defaultdict(dict)
        for (left, right), count in pairs.items():
            conditional[left][right] = count / counts[left]
        logger.info(
            "Co-occurrence model fitted: training_events=%d observed_objects=%d directed_pairs=%d",
            len(self._event_objects(training_log)), len(counts), len(conditional),
        )

        predictions: list[Prediction] = []
        test_events = {
            link.event_id for link in corruption.removed_event_object_links
        }
        candidate_ids = corruption.candidate_object_ids
        object_types = {obj.id: obj.type for obj in corruption.original.objects}
        logger.info(
            "Co-occurrence candidate universe aligned: %d event-participating objects",
            len(candidate_ids),
        )
        for event_id, observed in self._event_objects(corruption.corrupted).items():
            if event_id not in test_events:
                continue
            candidates = candidate_ids - observed
            if self.same_type_candidates:
                target_link = next(
                    link
                    for link in corruption.removed_event_object_links
                    if link.event_id == event_id
                )
                target_type = object_types[target_link.object_id]
                candidates = {
                    candidate
                    for candidate in candidates
                    if object_types[candidate] == target_type
                }
            scored = []
            for candidate in candidates:
                score = sum(
                    math.log(conditional[object_id].get(candidate, 1e-9))
                    for object_id in observed
                )
                scored.append((candidate, score))
            for rank, (candidate, score) in enumerate(
                sorted(scored, key=lambda item: (-item[1], item[0]))[: self.top_k], 1
            ):
                predictions.append(Prediction(event_id, candidate, score, rank))
        return tuple(predictions)

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
        hits = {
            k: sum(rank <= k for rank in ranks) / len(truth) if truth else 0.0
            for k in (1, 3, 5, 10)
        }
        correct = len(ranks)
        predicted = len(predictions)
        recall = correct / len(truth) if truth else 0.0
        precision = correct / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return EvaluationResult(
            "co-occurrence",
            len(truth),
            hits,
            sum(1 / rank for rank in ranks) / len(truth) if truth else 0.0,
            precision,
            recall,
            f1,
        )

    @staticmethod
    def _event_objects(log):
        grouped: dict[str, set[str]] = defaultdict(set)
        for link in log.event_object_links:
            grouped[link.event_id].add(link.object_id)
        return grouped

class GnnPredictor(PredictorPort):
    def __init__(self, config: GnnConfig | None = None):
        self.model = GnnModelAdapter(config or GnnConfig())

    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        return self.model.predict(corruption)

    def evaluate(
        self, corruption: CorruptionResult, predictions: tuple[Prediction, ...]
    ) -> EvaluationResult:
        return self.model.evaluate(corruption, predictions)