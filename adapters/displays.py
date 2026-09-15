from typing import Any, Dict
from core.ports import ResultDisplayPort
from dataclasses import asdict

class ConsoleDisplay(ResultDisplayPort):
    def render(self, results: object) -> None:
        print("\n--- RESULTS ---")
        values = asdict(results) if hasattr(results, "__dataclass_fields__") else results
        for key, value in values.items():
            print(f"{key}: {value}")

    def render_comparison(self, results: dict[str, object]) -> None:
        print("\n--- MODEL COMPARISON ---")
        headers = ("Model", "Runtime (s)", "Hits@1", "Hits@3", "Hits@5", "Hits@10", "MRR", "F1")
        print(" | ".join(headers))
        print("-" * 96)
        for name, result in results.items():
            print(
                f"{name} | {result.runtime_seconds:.2f} | "
                f"{result.hits_at_k.get(1, 0.0):.4f} | "
                f"{result.hits_at_k.get(3, 0.0):.4f} | "
                f"{result.hits_at_k.get(5, 0.0):.4f} | "
                f"{result.hits_at_k.get(10, 0.0):.4f} | "
                f"{result.mrr:.4f} | {result.f1:.4f}"
            )

    def render_samples(
        self,
        corruption,
        predictions,
        output_path: str,
        sample_count: int,
        top_n: int,
    ) -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path

        links_by_event = {}
        for link in corruption.original.event_object_links:
            links_by_event.setdefault(link.event_id, []).append(link.object_id)
        removed_by_event = {
            link.event_id: link.object_id
            for link in corruption.removed_event_object_links
        }
        observed_by_event = {}
        for link in corruption.corrupted.event_object_links:
            observed_by_event.setdefault(link.event_id, []).append(link.object_id)
        object_types = {obj.id: obj.type for obj in corruption.original.objects}
        prediction_maps = {}
        for model_name, model_predictions in predictions.items():
            prediction_maps[model_name] = {}
            for prediction in model_predictions:
                prediction_maps[model_name].setdefault(prediction.event_id, []).append(
                    prediction
                )

        event_ids = list(removed_by_event)[:sample_count]
        if not event_ids:
            raise ValueError("No hidden links are available for visualization")
        rows = []
        for event_id in event_ids:
            all_objects = links_by_event[event_id]
            actual = removed_by_event[event_id]
            observed = observed_by_event.get(event_id, [])
            rows.append(
                (
                    event_id,
                    " | ".join(
                        f"{obj} ({object_types.get(obj, '?')})" for obj in all_objects
                    ),
                    " | ".join(observed) or "-",
                    f"{actual} ({object_types.get(actual, '?')})",
                    *[
                        " | ".join(
                            f"{item.object_id} (r{item.rank})"
                            for item in sorted(
                                prediction_maps.get(model_name, {}).get(event_id, []),
                                key=lambda item: item.rank,
                            )[:top_n]
                        )
                        or "-"
                        for model_name in ("co-occurrence", "gnn")
                    ],
                )
            )

        print("\n--- SAMPLE RECONSTRUCTIONS ---")
        for row in rows:
            print(f"\nEvent {row[0]}")
            print(f"  All objects:      {row[1]}")
            print(f"  Observed objects: {row[2]}")
            print(f"  Actual missing:   {row[3]}")
            print(f"  Co-occurrence:    {row[4]}")
            print(f"  GNN:              {row[5]}")

        figure, axis = plt.subplots(
            figsize=(18, max(3.5, 1.1 * len(rows) + 1.5))
        )
        axis.axis("off")
        table = axis.table(
            cellText=[list(row) for row in rows],
            colLabels=[
                "Event",
                "All objects",
                "Observed",
                "Actual missing",
                "Co-occurrence top predictions",
                "GNN top predictions",
            ],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.0)
        figure.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"\nSaved sample visualization: {output_path}")