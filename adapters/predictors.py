import math
import logging
import time
from collections import Counter, defaultdict
from itertools import combinations

from core.domain import CorruptionResult, EvaluationResult, Prediction
from core.ports import PredictorPort
from core.metrics import evaluate_rankings
from adapters.gnn import GnnConfig, GnnModelAdapter

logger = logging.getLogger(__name__)

class CoOccurrenceBaseline(PredictorPort):
    def __init__(self, top_k: int = 10):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self._fit_seconds = 0.0
        self._inference_seconds = 0.0

    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        started = time.perf_counter()
        counts = Counter()
        pairs = Counter()
        training_log = corruption.statistics_training or corruption.corrupted
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
        self._fit_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()

        predictions: list[Prediction] = []
        test_events = {
            link.event_id for link in corruption.removed_event_object_links
        }
        candidate_ids = corruption.candidate_object_ids
        logger.info(
            "Co-occurrence candidate universe aligned: %d event-participating objects",
            len(candidate_ids),
        )
        for event_id, observed in self._event_objects(corruption.corrupted).items():
            if event_id not in test_events:
                continue
            candidates = candidate_ids - observed
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
        self._inference_seconds = time.perf_counter() - inference_started
        return tuple(predictions)

    def timing(self) -> tuple[float, float]:
        return self._fit_seconds, self._inference_seconds

    def evaluate(
        self, corruption: CorruptionResult, predictions: tuple[Prediction, ...]
    ) -> EvaluationResult:
        return evaluate_rankings("co-occurrence", corruption, predictions)

    @staticmethod
    def _event_objects(log):
        grouped: dict[str, set[str]] = defaultdict(set)
        for link in log.event_object_links:
            grouped[link.event_id].add(link.object_id)
        return grouped

class GnnPredictor(PredictorPort):
    def __init__(self, config: GnnConfig | None = None):
        self.model = GnnModelAdapter(config or GnnConfig())
        self._fit_seconds = 0.0
        self._inference_seconds = 0.0

    def train(self, corruption: CorruptionResult) -> None:
        self.model.fit(corruption)

    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        predictions = self.model.predict(corruption)
        self._fit_seconds = self.model.fit_seconds
        self._inference_seconds = self.model.inference_seconds
        return predictions

    def timing(self) -> tuple[float, float]:
        return self._fit_seconds, self._inference_seconds

    def evaluate(
        self, corruption: CorruptionResult, predictions: tuple[Prediction, ...]
    ) -> EvaluationResult:
        return self.model.evaluate(corruption, predictions)