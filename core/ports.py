from abc import ABC, abstractmethod
from pathlib import Path

from core.domain import CorruptionResult, EvaluationResult, OcelLog, Prediction

class DataLoaderPort(ABC):
    @abstractmethod
    def load(self, source_path: str | Path) -> OcelLog:
        pass

class GapSimulatorPort(ABC):
    @abstractmethod
    def introduce_gaps(self, ocel_log: OcelLog) -> CorruptionResult:
        pass


class PredictorPort(ABC):
    @abstractmethod
    def predict(self, corruption: CorruptionResult) -> tuple[Prediction, ...]:
        pass

    @abstractmethod
    def evaluate(
        self,
        corruption: CorruptionResult,
        predictions: tuple[Prediction, ...],
    ) -> EvaluationResult:
        pass

    @abstractmethod
    def timing(self) -> tuple[float, float]:
        pass

class ResultDisplayPort(ABC):
    @abstractmethod
    def render(self, results: object) -> None:
        pass

    @abstractmethod
    def render_comparison(self, results: dict[str, EvaluationResult]) -> None:
        pass

    @abstractmethod
    def render_comparison_examples(
        self,
        corruption: CorruptionResult,
        predictions: dict[str, tuple[Prediction, ...]],
        limit: int = 10,
    ) -> None:
        pass