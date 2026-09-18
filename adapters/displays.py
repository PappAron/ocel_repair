from collections import defaultdict

from core.ports import ResultDisplayPort
from core.domain import CorruptionResult, Prediction
from dataclasses import asdict

class ConsoleDisplay(ResultDisplayPort):
    def render(self, results: object) -> None:
        print("\n--- RESULTS ---")
        values = asdict(results) if hasattr(results, "__dataclass_fields__") else results
        for key, value in values.items():
            print(f"{key}: {value}")

    def render_comparison(self, results: dict[str, object]) -> None:
        print("\n--- MODEL COMPARISON ---")
        headers = (
            "Model", "Fit (s)", "Inference (s)", "Total (s)",
            "Hits@1", "Hits@3", "Hits@5", "Hits@10", "MRR",
        )
        print(" | ".join(headers))
        print("-" * 96)
        for name, result in results.items():
            print(
                f"{name} | {result.fit_seconds:.2f} | "
                f"{result.inference_seconds:.2f} | {result.runtime_seconds:.2f} | "
                f"{result.hits_at_k.get(1, 0.0):.4f} | "
                f"{result.hits_at_k.get(3, 0.0):.4f} | "
                f"{result.hits_at_k.get(5, 0.0):.4f} | "
                f"{result.hits_at_k.get(10, 0.0):.4f} | "
                f"{result.mrr:.4f}"
            )

    def render_comparison_examples(
        self,
        corruption: CorruptionResult,
        predictions: dict[str, tuple[Prediction, ...]],
        limit: int = 10,
    ) -> None:
        observed = defaultdict(set)
        for link in corruption.corrupted.event_object_links:
            observed[link.event_id].add(link.object_id)
        targets = {
            link.event_id: link.object_id
            for link in corruption.removed_event_object_links
        }
        by_model_event = {}
        for model_name, model_predictions in predictions.items():
            grouped = defaultdict(list)
            for prediction in model_predictions:
                grouped[prediction.event_id].append(prediction)
            by_model_event[model_name] = grouped

        print("\n--- EVALUATION EXAMPLES ---")
        for event_id in sorted(targets)[:limit]:
            print(f"\nEvent {event_id}")
            print(f"  Observed: {', '.join(sorted(observed[event_id])) or '(none)'}")
            print(f"  Missing:  {targets[event_id]}")
            for model_name, grouped in by_model_event.items():
                ranked = grouped.get(event_id, [])[:5]
                values = ", ".join(
                    f"{prediction.object_id} (r{prediction.rank})"
                    for prediction in ranked
                ) or "(no predictions)"
                print(f"  {model_name}: {values}")
