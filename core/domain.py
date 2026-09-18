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

    @property
    def candidate_object_ids(self) -> frozenset[str]:
        return frozenset(link.object_id for link in self.original.event_object_links)


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
    precision: float
    recall: float
    runtime_seconds: float = 0.0
    fit_seconds: float = 0.0
    inference_seconds: float = 0.0
