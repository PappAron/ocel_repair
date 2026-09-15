import logging
import time
from dataclasses import replace

from core.ports import DataLoaderPort, GapSimulatorPort, PredictorPort, ResultDisplayPort

logger = logging.getLogger(__name__)

class OcelPipeline:
    def __init__(
        self,
        loader: DataLoaderPort,
        predictor: PredictorPort,
        display: ResultDisplayPort,
        simulator: GapSimulatorPort,
    ):
        self.loader = loader
        self.simulator = simulator
        self.predictor = predictor
        self.display = display

    def run(self, source_path: str):
        started = time.perf_counter()
        logger.info("Starting pipeline: source=%s predictor=%s", source_path, type(self.predictor).__name__)
        logger.info("Loading OCEL data")
        raw_data = self.loader.load(source_path)
        logger.info(
            "Loaded events=%d objects=%d event_object_links=%d object_object_links=%d",
            len(raw_data.events), len(raw_data.objects),
            len(raw_data.event_object_links), len(raw_data.object_object_links),
        )
        logger.info("Introducing reproducible event-object gaps")
        corruption = self.simulator.introduce_gaps(raw_data)
        logger.info(
            "Corruption complete: training_links=%d corrupted_links=%d removed_links=%d",
            len(corruption.training.event_object_links if corruption.training else ()),
            len(corruption.corrupted.event_object_links),
            len(corruption.removed_event_object_links),
        )
        logger.info("Generating predictions")
        predictions = self.predictor.predict(corruption)
        logger.info("Generated %d predictions", len(predictions))
        results = replace(
            self.predictor.evaluate(corruption, predictions),
            runtime_seconds=time.perf_counter() - started,
        )
        logger.info(
            "Evaluation complete: model=%s mrr=%.4f f1=%.4f elapsed=%.2fs",
            results.model, results.mrr, results.f1, time.perf_counter() - started,
        )
        self.display.render(results)
        return results

    def compare(
        self,
        source_path: str,
        predictors: dict[str, PredictorPort],
        visualization_path: str | None = None,
        visualization_samples: int = 5,
        visualization_top_n: int = 5,
    ):
        started = time.perf_counter()
        logger.info("Starting model comparison: source=%s models=%s", source_path, list(predictors))
        raw_data = self.loader.load(source_path)
        logger.info(
            "Loaded events=%d objects=%d event_object_links=%d object_object_links=%d",
            len(raw_data.events), len(raw_data.objects),
            len(raw_data.event_object_links), len(raw_data.object_object_links),
        )
        corruption = self.simulator.introduce_gaps(raw_data)
        results = {}
        predictions_by_model = {}
        for name, predictor in predictors.items():
            model_started = time.perf_counter()
            logger.info("Comparison model started: %s", name)
            predictions = predictor.predict(corruption)
            predictions_by_model[name] = predictions
            result = predictor.evaluate(corruption, predictions)
            result = replace(
                result,
                runtime_seconds=time.perf_counter() - model_started,
            )
            results[name] = result
            logger.info(
                "Comparison model complete: model=%s runtime=%.2fs mrr=%.4f f1=%.4f",
                name, result.runtime_seconds, result.mrr, result.f1,
            )
        logger.info("Model comparison complete: elapsed=%.2fs", time.perf_counter() - started)
        self.display.render_comparison(results)
        if visualization_path:
            self.display.render_samples(
                corruption,
                predictions_by_model,
                visualization_path,
                visualization_samples,
                visualization_top_n,
            )
        return results