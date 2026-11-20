"""Decision rule for automatic repair (reviewer points D11, E6, E8).

A ranking says which object is most plausible; it does not say whether a link should be
inserted at all. The rule here inserts, for each event with a gap, only its top-ranked object,
and only if the model is confident enough. Confidence is the score margin between the best and
the second-best candidate (0 when the best score is tied). The threshold is chosen on the
validation events for a target precision and then applied unchanged to the test events.

Reported per target precision: the chosen threshold, validation precision/coverage, and test
precision, recall (correctly inserted hidden links / all hidden links) and coverage (share of
gap events that receive an automatic insertion). Ties at the top are scored as expected values
under random tie-breaking, as in core/metrics.py.
"""
import math
from collections import defaultdict

from core.domain import EventObjectLink, Prediction

TARGET_PRECISIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MIN_SUPPORT = 10


def event_decisions(
    links: tuple[EventObjectLink, ...], predictions: tuple[Prediction, ...]
) -> list[dict]:
    """One record per event with hidden links: confidence margin, expected correctness of
    inserting its top-ranked object, and its number of hidden links."""
    hidden: dict[str, set[str]] = defaultdict(set)
    for link in links:
        hidden[link.event_id].add(link.object_id)
    scored: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for prediction in predictions:
        if prediction.event_id in hidden and math.isfinite(prediction.score):
            scored[prediction.event_id].append((prediction.score, prediction.object_id))
    records = []
    for event_id in sorted(hidden):
        targets = hidden[event_id]
        candidates = scored.get(event_id, [])
        if not candidates:
            records.append({"event_id": event_id, "margin": -math.inf, "correct": 0.0,
                            "hidden": len(targets)})
            continue
        top = max(score for score, _ in candidates)
        tied = [object_id for score, object_id in candidates if score == top]
        lower = [score for score, _ in candidates if score < top]
        if len(tied) > 1:
            margin = 0.0
        else:
            margin = top - max(lower) if lower else math.inf
        correct = sum(object_id in targets for object_id in tied) / len(tied)
        records.append({"event_id": event_id, "margin": margin, "correct": correct,
                        "hidden": len(targets)})
    return records


def _apply(records: list[dict], threshold: float) -> dict:
    inserted = [r for r in records if r["margin"] >= threshold]
    correct = sum(r["correct"] for r in inserted)
    hidden = sum(r["hidden"] for r in records)
    return {
        "inserted": len(inserted),
        "precision": correct / len(inserted) if inserted else None,
        "recall": correct / hidden if hidden else 0.0,
        "coverage": len(inserted) / len(records) if records else 0.0,
    }


def choose_threshold(records: list[dict], target: float, min_support: int = MIN_SUPPORT) -> float:
    """Lowest margin threshold whose validation precision reaches `target`, with at least
    `min_support` insertions; +inf (never insert) if no threshold qualifies."""
    best = math.inf
    for threshold in sorted({r["margin"] for r in records if r["margin"] > -math.inf}):
        stats = _apply(records, threshold)
        if stats["inserted"] >= min_support and stats["precision"] >= target:
            return threshold
    return best


def decision_report(
    validation_links: tuple[EventObjectLink, ...],
    validation_predictions: tuple[Prediction, ...],
    test_links: tuple[EventObjectLink, ...],
    test_predictions: tuple[Prediction, ...],
    targets=TARGET_PRECISIONS,
) -> dict:
    validation = event_decisions(validation_links, validation_predictions)
    test = event_decisions(test_links, test_predictions)
    report = {
        "confidence": "score margin top-1 minus top-2 (0 if top-1 is tied)",
        "rule": "insert the top-ranked object if margin >= threshold; threshold chosen on validation",
        "validation_events": len(validation),
        "test_events": len(test),
        "always_insert": {"validation": _apply(validation, -math.inf),
                          "test": _apply(test, -math.inf)},
        "targets": {},
        # per-event records, so precision-coverage curves can be drawn afterwards
        "records": {
            "validation": [[r["margin"] if math.isfinite(r["margin"]) else None,
                            r["correct"], r["hidden"]] for r in validation],
            "test": [[r["margin"] if math.isfinite(r["margin"]) else None,
                      r["correct"], r["hidden"]] for r in test],
        },
    }
    for target in targets:
        threshold = choose_threshold(validation, target)
        report["targets"][str(target)] = {
            "threshold": None if math.isinf(threshold) else threshold,
            "validation": _apply(validation, threshold),
            "test": _apply(test, threshold),
        }
    return report


def format_report(name: str, report: dict) -> str:
    def pct(value):
        return "  n/a" if value is None else f"{value:5.3f}"
    lines = [f"Decision rule ({name}): {report['confidence']}"]
    lines.append("  target  threshold   val-prec val-cov | test-prec test-rec test-cov inserted")
    rows = [("always", None, report["always_insert"])]
    rows += [(t, v["threshold"], v) for t, v in report["targets"].items()]
    for label, threshold, stats in rows:
        th = "      -  " if threshold is None and label == "always" else (
            "  never  " if threshold is None else f"{threshold:9.4f}")
        val, test = stats["validation"], stats["test"]
        lines.append(
            f"  {label:>6} {th}   {pct(val['precision'])}   {val['coverage']:5.3f} | "
            f"{pct(test['precision'])}     {test['recall']:5.3f}    {test['coverage']:5.3f} {test['inserted']:8d}"
        )
    return "\n".join(lines)
