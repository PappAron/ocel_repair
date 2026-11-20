import random
import logging

from dataclasses import replace

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
        context_drop: float = 0.0,
        o2o_drop: float = 0.0,
        one_object_context: bool = False,
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
        if not 0 <= context_drop < 1:
            raise ValueError("context_drop must be between 0 (inclusive) and 1 (exclusive)")
        if not 0 <= o2o_drop <= 1:
            raise ValueError("o2o_drop must be between 0 and 1")
        self.context_drop = context_drop
        self.o2o_drop = o2o_drop
        self.one_object_context = one_object_context

    def _degrade(self, ocel_log: OcelLog, train_ids: set[str]) -> OcelLog:
        """Sensitivity analyses (C3, E11). These links are not queries: they are simply
        missing from the log the models see.

        context_drop: every event-object link of a TRAINING event is removed independently
        with this probability (log-wide incompleteness), but each event keeps at least one
        object. Test and validation events are untouched, so the queries stay the same.
        o2o_drop: this share of all object-object links is removed uniformly at random.
        """
        if self.context_drop == 0 and self.o2o_drop == 0:
            return ocel_log
        rng = random.Random(self.target_seed + 7919)
        dropped: set[tuple[str, str, str | None]] = set()
        if self.context_drop > 0:
            by_event: dict[str, list[EventObjectLink]] = {}
            for link in ocel_log.event_object_links:
                if link.event_id in train_ids:
                    by_event.setdefault(link.event_id, []).append(link)
            for event_id in sorted(by_event):
                links = by_event[event_id]
                gone = [link for link in links if rng.random() < self.context_drop]
                if len(gone) == len(links):
                    gone.remove(rng.choice(gone))  # keep at least one object per event
                dropped.update((l.event_id, l.object_id, l.qualifier) for l in gone)
        o2o = ocel_log.object_object_links
        if self.o2o_drop > 0 and o2o:
            keep = len(o2o) - round(len(o2o) * self.o2o_drop)
            o2o = tuple(sorted(rng.sample(range(len(o2o)), keep)))
            o2o = tuple(ocel_log.object_object_links[i] for i in o2o)
        logger.info(
            "Sensitivity degradation: context_drop=%.2f removed_training_e2o=%d of %d | "
            "o2o_drop=%.2f kept_o2o=%d of %d",
            self.context_drop, len(dropped),
            sum(1 for l in ocel_log.event_object_links if l.event_id in train_ids),
            self.o2o_drop, len(o2o), len(ocel_log.object_object_links),
        )
        return replace(
            ocel_log.without_event_object_links(dropped), object_object_links=o2o
        )

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
        ocel_log = self._degrade(ocel_log, set(train_frame["event_id"].tolist()))
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
        context_ids = frozenset()
        if self.one_object_context:
            # Diagnostic: one-object events (never eligible) join every graph as context.
            context_ids = frozenset(e for e, links in by_event.items() if len(links) == 1)
            context_links = tuple(
                link for link in ocel_log.event_object_links if link.event_id in context_ids
            )
            training = replace(training, event_object_links=training.event_object_links + context_links)
            validation = replace(validation, event_object_links=validation.event_object_links + context_links)
            test_context = replace(test_context, event_object_links=test_context.event_object_links + context_links)
            logger.info("One-object context: %d events, %d links added to all graphs",
                        len(context_ids), len(context_links))
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
            context_event_ids=context_ids,
        )

class LogWideGapSimulator(GapSimulatorPort):
    """Log-wide protocol (Section 6.4 of the paper): one missing rate p for the whole log.

    Every E2O link of the log is removed independently with probability p (O2O relations are
    kept unless o2o_drop is set). Events are split 70/30 into training and held-out events; from
    the held-out events that lost at least one link, two disjoint samples of `sample_size` events
    are drawn as validation and test events, and ALL their removed links are the queries.
    The model sees only the remaining links of the training events and of the evaluated events;
    the removed links of training events are lost (as in a real damaged log). Evaluated events may
    have lost all their links; such gaps cannot be reached from relations and count as failures
    of relation-based methods (the temporal feature can still score them).
    """

    def __init__(self, missing_rate: float, sample_size: int = 200, seed: int = 0,
                 test_fraction: float = 0.3, o2o_drop: float = 0.0):
        if not 0 < missing_rate < 1:
            raise ValueError("missing_rate must be in (0, 1)")
        self.p, self.sample_size, self.seed = missing_rate, sample_size, seed
        self.test_fraction, self.o2o_drop = test_fraction, o2o_drop

    def introduce_gaps(self, ocel_log: OcelLog) -> CorruptionResult:
        rng = random.Random(self.seed)
        links = list(ocel_log.event_object_links)
        removed = [l for l in links if rng.random() < self.p]
        removed_keys = {(l.event_id, l.object_id, l.qualifier) for l in removed}
        o2o = ocel_log.object_object_links
        if self.o2o_drop > 0:
            keep = max(0, round(len(o2o) * (1 - self.o2o_drop)))
            o2o = tuple(sorted(rng.sample(list(o2o), keep), key=lambda l: (l.source_id, l.target_id)))
        damaged = replace(ocel_log.without_event_object_links(removed_keys), object_object_links=o2o)
        events = sorted({l.event_id for l in links})
        rng.shuffle(events)
        cut = int(round(len(events) * (1 - self.test_fraction)))
        train_ids, held = set(events[:cut]), events[cut:]
        removed_by_event = {}
        for l in removed:
            removed_by_event.setdefault(l.event_id, []).append(l)
        damaged_held = [e for e in held if e in removed_by_event]
        sampled = rng.sample(damaged_held, min(2 * self.sample_size, len(damaged_held)))
        test_ids, val_ids = set(sampled[:self.sample_size]), set(sampled[self.sample_size:])
        test_removed = tuple(l for e in sorted(test_ids) for l in removed_by_event[e])
        val_removed = tuple(l for e in sorted(val_ids) for l in removed_by_event[e])

        def view(ids):
            return OcelLog(ocel_log.events, ocel_log.objects,
                           tuple(l for l in damaged.event_object_links if l.event_id in ids), o2o)

        logger.info("Log-wide gap protocol: p=%.2f removed=%d of %d links | train_events=%d "
                    "test_events=%d test_queries=%d (events without any link left: %d) validation_queries=%d",
                    self.p, len(removed), len(links), len(train_ids), len(test_ids), len(test_removed),
                    sum(1 for e in test_ids if not any(l.event_id == e for l in damaged.event_object_links)),
                    len(val_removed))
        return CorruptionResult(
            original=damaged,  # the observed log: every removed link is gone (also for the temporal feature)
            corrupted=view(train_ids | test_ids),
            removed_event_object_links=test_removed,
            training=view(train_ids),
            validation=view(train_ids | val_ids),
            validation_removed_event_object_links=val_removed,
        )
