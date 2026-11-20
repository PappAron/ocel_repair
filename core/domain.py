from dataclasses import dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Object:
    id: str
    type: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventObjectLink:
    event_id: str
    object_id: str
    qualifier: str | None = None


@dataclass(frozen=True)
class ObjectObjectLink:
    source_id: str
    target_id: str
    qualifier: str | None = None


@dataclass(frozen=True)
class OcelLog:
    events: tuple[Event, ...]
    objects: tuple[Object, ...]
    event_object_links: tuple[EventObjectLink, ...]
    object_object_links: tuple[ObjectObjectLink, ...]

    def without_event_object_links(
        self, removed: set[tuple[str, str, str | None]]
    ) -> "OcelLog":
        links = tuple(
            link
            for link in self.event_object_links
            if (link.event_id, link.object_id, link.qualifier) not in removed
        )
        return replace(self, event_object_links=links)


@dataclass(frozen=True)
class CorruptionResult:
    original: OcelLog
    corrupted: OcelLog
    removed_event_object_links: tuple[EventObjectLink, ...]
    training: OcelLog | None = None
    validation: OcelLog | None = None
    validation_removed_event_object_links: tuple[EventObjectLink, ...] = ()
    # Events whose links were added to the graphs as context only (one-object events, see
    # --one-object-context). They are never damaged, queried or counted in statistics.
    context_event_ids: frozenset = frozenset()

    @property
    def statistics_training(self):
        """The training log without context-only events: source of training queries and
        co-occurrence statistics (identical to `training` unless context events were added)."""
        if self.training is None or not self.context_event_ids:
            return self.training
        context = self.context_event_ids
        return OcelLog(
            self.training.events,
            self.training.objects,
            tuple(l for l in self.training.event_object_links if l.event_id not in context),
            self.training.object_object_links,
        )

    @property
    def candidate_object_ids(self) -> frozenset[str]:
        # All objects of the log's object table. (Previously: objects with an
        # event-object link in the ORIGINAL log, which used the ground truth.)
        return frozenset(obj.id for obj in self.original.objects)


@dataclass(frozen=True)
class Prediction:
    event_id: str
    object_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    evaluated_links: int
    hits_at_k: Mapping[int, float]
    mrr: float
    runtime_seconds: float = 0.0
    fit_seconds: float = 0.0
    inference_seconds: float = 0.0
    decision: Mapping | None = None
    per_query: tuple | None = None  # (event_id, object_id, reciprocal rank, hits@1, hits@3, hits@10)
