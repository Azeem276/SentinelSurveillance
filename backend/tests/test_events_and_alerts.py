"""Event aggregation/deduplication and the alarm state machine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.alerts.engine import AlertEngine
from app.events.bus import BusMessage, EventBus
from app.events.engine import DedupCache, EventContext, EventEmitter
from app.models.enums import AlertState, AlertType, EventSeverity, EventType
from app.repositories.event_repository import AlertRepository, EventRepository

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def emitter(session, make_source):
    source = make_source()
    return EventEmitter(session, EventContext(source.id, source.uid)), source


# ------------------------------------------------------------- dedup cache
class TestDedupCache:
    def test_first_occurrence_is_emitted(self):
        cache = DedupCache(window_seconds=30)
        assert cache.should_emit("k", when=NOW)

    def test_repeat_inside_the_window_is_suppressed(self):
        cache = DedupCache(window_seconds=30)
        cache.should_emit("k", when=NOW)
        assert not cache.should_emit("k", when=NOW + timedelta(seconds=5))

    def test_repeat_after_the_window_is_emitted(self):
        cache = DedupCache(window_seconds=30)
        cache.should_emit("k", when=NOW)
        assert cache.should_emit("k", when=NOW + timedelta(seconds=31))

    def test_frame_rate_flood_collapses_to_one(self):
        """25 fps for 10 s must not produce 250 events."""
        cache = DedupCache(window_seconds=30)
        emitted = sum(
            1 for i in range(250)
            if cache.should_emit("unknown_face", when=NOW + timedelta(seconds=i / 25))
        )
        assert emitted == 1

    def test_different_keys_are_independent(self):
        cache = DedupCache(window_seconds=30)
        assert cache.should_emit("a", when=NOW)
        assert cache.should_emit("b", when=NOW)

    def test_forget_prefix_clears_a_tracks_keys(self):
        cache = DedupCache(window_seconds=30)
        cache.should_emit("track:1:face", when=NOW)
        cache.should_emit("track:2:face", when=NOW)
        cache.forget_prefix("track:1")
        assert cache.should_emit("track:1:face", when=NOW)
        assert not cache.should_emit("track:2:face", when=NOW)


# ------------------------------------------------------- continuing events
class TestContinuingEvents:
    def test_open_creates_one_row(self, emitter):
        em, source = emitter
        event = em.open(EventType.UNKNOWN_FACE, dedup_key="track:1:unknown", when=NOW)
        assert event.is_open
        assert event.occurrence_count == 1

    def test_reopening_extends_rather_than_duplicates(self, emitter, session):
        em, source = emitter
        first = em.open(EventType.UNKNOWN_FACE, dedup_key="track:1:unknown", when=NOW)
        for i in range(1, 60):
            em.open(EventType.UNKNOWN_FACE, dedup_key="track:1:unknown",
                    when=NOW + timedelta(seconds=i))
        rows = EventRepository(session).list_events(
            source_id=source.id, event_types=[EventType.UNKNOWN_FACE], limit=200
        )
        assert len(rows) == 1, "an unknown person in frame is ONE event"
        session.refresh(first)
        assert first.duration_seconds and first.duration_seconds >= 59

    def test_closing_emits_the_paired_event(self, emitter, session):
        em, source = emitter
        event = em.open(EventType.MOTION_STARTED, dedup_key="motion", when=NOW)
        em.close(event.id, when=NOW + timedelta(seconds=20),
                 closing_type=EventType.MOTION_ENDED)
        session.refresh(event)
        assert not event.is_open
        assert event.duration_seconds == pytest.approx(20.0, abs=0.5)
        ended = EventRepository(session).list_events(
            source_id=source.id, event_types=[EventType.MOTION_ENDED]
        )
        assert len(ended) == 1

    def test_a_closed_event_does_not_block_a_later_one(self, emitter, session):
        em, source = emitter
        first = em.open(EventType.MOTION_STARTED, dedup_key="motion", when=NOW)
        em.close(first.id, when=NOW + timedelta(seconds=5))
        second = em.open(EventType.MOTION_STARTED, dedup_key="motion",
                         when=NOW + timedelta(seconds=60))
        assert second.id != first.id

    def test_different_tracks_get_different_events(self, emitter, session):
        em, source = emitter
        a = em.open(EventType.UNKNOWN_FACE, dedup_key="track:1:unknown", when=NOW)
        b = em.open(EventType.UNKNOWN_FACE, dedup_key="track:2:unknown", when=NOW)
        assert a.id != b.id

    def test_instant_events_are_not_open(self, emitter):
        em, _ = emitter
        event = em.emit(EventType.FACE_RECOGNIZED, when=NOW)
        assert not event.is_open
        assert event.ended_at is not None

    def test_severity_defaults_are_applied(self, emitter):
        em, _ = emitter
        assert em.emit(EventType.ALARM_STARTED, when=NOW).severity is EventSeverity.CRITICAL
        assert em.emit(EventType.UNKNOWN_FACE, when=NOW).severity is EventSeverity.WARNING
        assert em.emit(EventType.FACE_DETECTED, when=NOW).severity is EventSeverity.INFO


# ------------------------------------------------------------- alarm state
class TestAlertEngine:
    def test_continuous_alarm_starts_once_per_track(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        first = engine.start_continuous(
            session, source_id=source.id, source_uid=source.uid,
            reason="unfamiliar_person_in_alarm_zone", track_key=7,
        )
        second = engine.start_continuous(
            session, source_id=source.id, source_uid=source.uid,
            reason="unfamiliar_person_in_alarm_zone", track_key=7,
        )
        assert first is not None
        assert second is None, "one track must not raise two alarms"
        assert engine.active_count == 1

    def test_a_frame_loop_cannot_stack_alarms(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        for _ in range(300):
            engine.start_continuous(
                session, source_id=source.id, source_uid=source.uid,
                reason="unfamiliar_person_in_alarm_zone", track_key=7,
            )
        assert engine.active_count == 1

    def test_alarm_persists_until_manually_stopped(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        alert = engine.start_continuous(
            session, source_id=source.id, source_uid=source.uid,
            reason="unfamiliar_person_in_alarm_zone", track_key=7,
        )
        assert engine.is_alarming(source.id, 7)
        assert AlertRepository(session).active_alerts()[0].id == alert.id

        engine.stop(session, alert.id, source_uid=source.uid, stopped_by="operator")
        session.refresh(alert)
        assert alert.state is AlertState.STOPPED
        assert alert.stopped_by == "operator"
        assert engine.active_count == 0
        assert AlertRepository(session).active_alerts() == []

    def test_track_ending_does_not_silence_the_alarm(self, session, make_source):
        """The operator must acknowledge; walking away is not enough."""
        source = make_source()
        engine = AlertEngine()
        alert = engine.start_continuous(
            session, source_id=source.id, source_uid=source.uid,
            reason="unfamiliar_person_in_alarm_zone", track_key=7,
        )
        engine.release_track(source.id, 7)
        session.refresh(alert)
        assert alert.state is AlertState.ACTIVE
        assert len(AlertRepository(session).active_alerts()) == 1

    def test_a_new_track_after_release_can_alarm_again(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        engine.start_continuous(session, source_id=source.id, source_uid=source.uid,
                                reason="r", track_key=7)
        engine.release_track(source.id, 7)
        again = engine.start_continuous(session, source_id=source.id,
                                        source_uid=source.uid, reason="r", track_key=7)
        assert again is not None

    def test_beep_is_recorded_and_auto_cleared(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        alert = engine.beep(
            session, source_id=source.id, source_uid=source.uid,
            reason="temporary_familiar_recognized", track_key=3,
        )
        session.refresh(alert)
        assert alert.alert_type is AlertType.BEEP
        assert alert.state is AlertState.AUTO_CLEARED
        assert AlertRepository(session).active_alerts() == []

    def test_stop_all_clears_every_alarm(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        for key in (1, 2, 3):
            engine.start_continuous(session, source_id=source.id,
                                    source_uid=source.uid, reason="r", track_key=key)
        assert engine.active_count == 3
        assert engine.stop_all(session) == 3
        assert engine.active_count == 0

    def test_hydrate_restores_latches_after_a_restart(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        engine.start_continuous(session, source_id=source.id, source_uid=source.uid,
                                reason="r", track_key=11)

        fresh = AlertEngine()
        assert not fresh.is_alarming(source.id, 11)
        fresh.hydrate(session)
        assert fresh.is_alarming(source.id, 11)

    def test_stopping_an_already_stopped_alarm_is_safe(self, session, make_source):
        source = make_source()
        engine = AlertEngine()
        alert = engine.start_continuous(session, source_id=source.id,
                                        source_uid=source.uid, reason="r", track_key=1)
        engine.stop(session, alert.id)
        engine.stop(session, alert.id)
        assert engine.active_count == 0


# ---------------------------------------------------------------- event bus
class TestEventBus:
    def test_subscriber_receives_published_messages(self):
        bus = EventBus()
        sub = bus.subscribe()
        bus.publish(BusMessage(type="event", payload={"x": 1}))
        assert sub.queue.qsize() == 1

    def test_topic_filter_applies(self):
        bus = EventBus()
        sub = bus.subscribe(topics=["alarm_started"])
        bus.publish(BusMessage(type="motion", payload={}))
        bus.publish(BusMessage(type="alarm_started", payload={}))
        assert sub.queue.qsize() == 1

    def test_source_filter_applies(self):
        bus = EventBus()
        sub = bus.subscribe(sources=["camera_01"])
        bus.publish(BusMessage(type="motion", payload={}, source_uid="camera_02"))
        bus.publish(BusMessage(type="motion", payload={}, source_uid="camera_01"))
        assert sub.queue.qsize() == 1

    def test_a_slow_subscriber_never_blocks_the_producer(self):
        """Dropping frames beats stalling a capture thread."""
        bus = EventBus()
        sub = bus.subscribe()
        for i in range(1000):
            bus.publish(BusMessage(type="frame_analysis", payload={"i": i}))
        assert sub.queue.qsize() <= 256
        assert sub.dropped > 0

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        sub = bus.subscribe()
        bus.unsubscribe(sub)
        bus.publish(BusMessage(type="event", payload={}))
        assert sub.queue.qsize() == 0

    def test_recent_buffer_is_available_for_late_joiners(self):
        bus = EventBus()
        for i in range(5):
            bus.publish(BusMessage(type="event", payload={"i": i}))
        assert len(bus.recent(10)) == 5
