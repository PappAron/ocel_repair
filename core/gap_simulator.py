import random
import logging

from core.domain import CorruptionResult, EventObjectLink, OcelLog
from core.ports import GapSimulatorPort

logger = logging.getLogger(__name__)

class RandomEventObjectGapSimulator(GapSimulatorPort):
    def __init__(
        self,
        drop_fraction: float = 0.3,
        test_fraction: float = 0.3,
        seed: int = 42,
    ):
        if not 0 <= drop_fraction < 1:
            raise ValueError("drop_fraction must be between 0 and 1")
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        self.drop_fraction = drop_fraction
        self.test_fraction = test_fraction
        self.seed = seed

    def introduce_gaps(self, ocel_log: OcelLog) -> CorruptionResult:
        rng = random.Random(self.seed)
        by_event: dict[str, list[EventObjectLink]] = {}
        for link in ocel_log.event_object_links:
            by_event.setdefault(link.event_id, []).append(link)

        event_ids = sorted(by_event)
        test_count = max(1, int(len(event_ids) * self.test_fraction))
        test_events = set(rng.sample(event_ids, test_count))
        logger.info(
            "Gap configuration: test_fraction=%.2f drop_fraction=%.2f seed=%d test_events=%d",
            self.test_fraction, self.drop_fraction, self.seed, len(test_events),
        )
        removed: list[EventObjectLink] = []
        for event_id in test_events:
            links = by_event[event_id]
            if len(links) <= 1:
                continue
            count = max(1, int(len(links) * self.drop_fraction))
            removed.extend(rng.sample(links, count))

        removed_keys = {
            (link.event_id, link.object_id, link.qualifier) for link in removed
        }
        training_links = tuple(
            link for link in ocel_log.event_object_links if link.event_id not in test_events
        )
        training = OcelLog(
            ocel_log.events,
            ocel_log.objects,
            training_links,
            ocel_log.object_object_links,
        )
        return CorruptionResult(
            original=ocel_log,
            corrupted=ocel_log.without_event_object_links(removed_keys),
            removed_event_object_links=tuple(removed),
            training=training,
        )


class SingleTargetGapSimulator(RandomEventObjectGapSimulator):
    """Single-target protocol: one hidden object link per test event."""

    def __init__(self, test_fraction: float = 0.3, seed: int = 42):
        super().__init__(drop_fraction=0.0, test_fraction=test_fraction, seed=seed)

    def introduce_gaps(self, ocel_log: OcelLog) -> CorruptionResult:
        rng = random.Random(self.seed)
        by_event: dict[str, list[EventObjectLink]] = {}
        for link in ocel_log.event_object_links:
            by_event.setdefault(link.event_id, []).append(link)

        eligible_events = sorted(
            event_id for event_id, links in by_event.items() if len(links) >= 2
        )
        test_count = max(1, int(len(eligible_events) * self.test_fraction))
        test_events = set(rng.sample(eligible_events, test_count))
        removed = [rng.choice(by_event[event_id]) for event_id in sorted(test_events)]
        removed_keys = {
            (link.event_id, link.object_id, link.qualifier) for link in removed
        }
        training = OcelLog(
            ocel_log.events,
            ocel_log.objects,
            tuple(link for link in ocel_log.event_object_links if link.event_id not in test_events),
            ocel_log.object_object_links,
        )
        logger.info(
            "Single-target gap configuration: test_fraction=%.2f seed=%d "
            "test_events=%d removed_links=%d",
            self.test_fraction, self.seed, len(test_events), len(removed),
        )
        return CorruptionResult(
            original=ocel_log,
            corrupted=ocel_log.without_event_object_links(removed_keys),
            removed_event_object_links=tuple(removed),
            training=training,
        )


class SampledGapSimulator(GapSimulatorPort):
    """Splits events into training/validation/test pools and draws fixed-size
    validation and test samples from the held-out pool.

    For each sampled test/validation event with n participating objects, the
    simulator hides k = max(1, floor(n * drop_fraction)) of them (at least
    one, and scaling with event size). Training positives come only from the
    70% training event split, while the graph retains all non-hidden
    event-object links.
    """

    def __init__(
        self,
        test_fraction: float = 0.3,
        sample_size: int = 200,
        split_seed: int = 42,
        sample_seed: int = 0,
        target_seed: int = 42,
        validation: bool = True,
        drop_fraction: float = 0.3,
    ):
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not 0 <= drop_fraction < 1:
            raise ValueError("drop_fraction must be between 0 (inclusive) and 1 (exclusive)")
        self.test_fraction = test_fraction
        self.sample_size = sample_size
        self.split_seed = split_seed
        self.sample_seed = sample_seed
        self.target_seed = target_seed
        self.validation = validation
        self.drop_fraction = drop_fraction

    def _select_targets(
        self,
        target_rng: random.Random,
        event_ids: list[str],
        by_event: dict[str, list[EventObjectLink]],
    ) -> list[EventObjectLink]:
        removed: list[EventObjectLink] = []
        for event_id in event_ids:
            links = by_event[event_id]
            k = max(1, int(len(links) * self.drop_fraction))
            removed.extend(target_rng.sample(links, k))
        return removed

    def introduce_gaps(self, ocel_log: OcelLog) -> CorruptionResult:
        import pandas as pd

        by_event: dict[str, list[EventObjectLink]] = {}
        for link in ocel_log.event_object_links:
            by_event.setdefault(link.event_id, []).append(link)
        eligible = sorted(
            event_id for event_id, links in by_event.items() if len(links) >= 2
        )
        event_frame = pd.DataFrame({"event_id": eligible})
        train_frame = event_frame.sample(frac=1 - self.test_fraction, random_state=self.split_seed)
        test_frame = event_frame.drop(train_frame.index)
        if not self.validation:
            sampled_test = test_frame.sample(
                n=min(self.sample_size, len(test_frame)),
                random_state=self.sample_seed,
            )
            target_rng = random.Random(self.target_seed)
            removed = self._select_targets(
                target_rng, sampled_test["event_id"].tolist(), by_event
            )
            removed_keys = {
                (link.event_id, link.object_id, link.qualifier) for link in removed
            }
            train_ids = set(train_frame["event_id"].tolist())
            training = OcelLog(
                ocel_log.events,
                ocel_log.objects,
                tuple(
                    link
                    for link in ocel_log.event_object_links
                    if link.event_id in train_ids
                ),
                ocel_log.object_object_links,
            )
            logger.info(
                "Sampled gap protocol: train_events=%d test_events=%d "
                "test_queries=%d validation=disabled",
                len(train_frame), len(sampled_test), len(removed),
            )
            return CorruptionResult(
                original=ocel_log,
                corrupted=ocel_log.without_event_object_links(removed_keys),
                removed_event_object_links=tuple(removed),
                training=training,
            )
        sampled_test = test_frame.sample(
            n=min(self.sample_size, len(test_frame)),
            random_state=self.sample_seed,
        )
        validation_frame = test_frame.drop(sampled_test.index)
        sampled_validation = validation_frame.sample(
            n=min(self.sample_size, len(validation_frame)),
            random_state=self.sample_seed + 1,
        )
        final_test_frame = sampled_test

        target_rng = random.Random(self.target_seed)
        removed = self._select_targets(
            target_rng, sampled_test["event_id"].tolist(), by_event
        )
        validation_removed = self._select_targets(
            target_rng, sampled_validation["event_id"].tolist(), by_event
        )
        removed_keys = {
            (link.event_id, link.object_id, link.qualifier) for link in removed
        }
        validation_keys = {
            (link.event_id, link.object_id, link.qualifier)
            for link in validation_removed
        }
        train_ids = set(train_frame["event_id"].tolist())
        training = OcelLog(
            ocel_log.events,
            ocel_log.objects,
            tuple(link for link in ocel_log.event_object_links if link.event_id in train_ids),
            ocel_log.object_object_links,
        )
        validation_ids = train_ids | set(sampled_validation["event_id"].tolist())
        test_ids = train_ids | set(final_test_frame["event_id"].tolist())
        validation = OcelLog(
            ocel_log.events,
            ocel_log.objects,
            tuple(
                link
                for link in ocel_log.event_object_links
                if link.event_id in validation_ids
            ),
            ocel_log.object_object_links,
        ).without_event_object_links(validation_keys)
        test_context = OcelLog(
            ocel_log.events,
            ocel_log.objects,
            tuple(
                link
                for link in ocel_log.event_object_links
                if link.event_id in test_ids
            ),
            ocel_log.object_object_links,
        ).without_event_object_links(removed_keys)
        logger.info(
            "Sampled gap protocol: train_events=%d validation_events=%d "
            "test_events=%d validation_queries=%d test_queries=%d",
            len(train_frame), len(sampled_validation), len(sampled_test),
            len(validation_removed), len(removed),
        )
        return CorruptionResult(
            original=ocel_log,
            corrupted=test_context,
            removed_event_object_links=tuple(removed),
            training=training,
            validation=validation,
            validation_removed_event_object_links=tuple(validation_removed),
        )