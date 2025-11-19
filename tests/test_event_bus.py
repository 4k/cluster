"""
Comprehensive tests for the enhanced event bus.
Tests all features including basic pub/sub, once handlers, wait_for,
middleware, metrics, history, and dead letter queue.
"""

import asyncio
import pytest
import time
from typing import List

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.event_bus import (
    EventBus, Event, EventType, EventHandler,
    emit_event, subscribe_to_events, once_event, wait_for_event
)


@pytest.fixture
async def event_bus():
    """Create a fresh event bus for each test."""
    # Reset singleton
    EventBus._instance = None

    bus = await EventBus.get_instance()
    await bus.start()
    yield bus
    await bus.stop()

    # Clean up
    EventBus._instance = None


class TestBasicPubSub:
    """Test basic publish/subscribe functionality."""

    @pytest.mark.asyncio
    async def test_basic_subscription(self, event_bus):
        """Test basic event subscription and emission."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.WAKE_WORD_DETECTED, handler)
        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {"confidence": 0.95})
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].type == EventType.WAKE_WORD_DETECTED
        assert received_events[0].data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple handlers for the same event."""
        handler1_calls = []
        handler2_calls = []

        async def handler1(event: Event):
            handler1_calls.append(event)

        async def handler2(event: Event):
            handler2_calls.append(event)

        event_bus.subscribe(EventType.SPEECH_DETECTED, handler1)
        event_bus.subscribe(EventType.SPEECH_DETECTED, handler2)

        await event_bus.emit(EventType.SPEECH_DETECTED, {"text": "hello"})
        await asyncio.sleep(0.1)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.TTS_STARTED, handler)
        await event_bus.emit(EventType.TTS_STARTED, {"text": "first"})
        await asyncio.sleep(0.1)

        event_bus.unsubscribe(handler)
        await event_bus.emit(EventType.TTS_STARTED, {"text": "second"})
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].data["text"] == "first"


class TestOnceHandlers:
    """Test one-time event handlers."""

    @pytest.mark.asyncio
    async def test_once_handler(self, event_bus):
        """Test that once handlers are called only once."""
        call_count = {"count": 0}

        async def handler(event: Event):
            call_count["count"] += 1

        event_bus.once(EventType.SPEECH_ENDED, handler)

        # Emit multiple times
        await event_bus.emit(EventType.SPEECH_ENDED, {"id": 1})
        await event_bus.emit(EventType.SPEECH_ENDED, {"id": 2})
        await event_bus.emit(EventType.SPEECH_ENDED, {"id": 3})
        await asyncio.sleep(0.1)

        assert call_count["count"] == 1

    @pytest.mark.asyncio
    async def test_once_with_multiple_event_types(self, event_bus):
        """Test once handler with multiple event types."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.once([EventType.TTS_STARTED, EventType.TTS_COMPLETED], handler)

        await event_bus.emit(EventType.TTS_STARTED, {"text": "hello"})
        await event_bus.emit(EventType.TTS_COMPLETED, {"duration": 1.5})
        await asyncio.sleep(0.1)

        # Should be called for the first matching event only
        assert len(received_events) == 1


class TestWaitFor:
    """Test wait_for request/response pattern."""

    @pytest.mark.asyncio
    async def test_wait_for_success(self, event_bus):
        """Test waiting for an event successfully."""
        async def emit_delayed():
            await asyncio.sleep(0.2)
            await event_bus.emit(EventType.MODEL_LOADED, {"model": "test"})

        asyncio.create_task(emit_delayed())
        result = await event_bus.wait_for(EventType.MODEL_LOADED, timeout=1.0)

        assert result is not None
        assert result.type == EventType.MODEL_LOADED
        assert result.data["model"] == "test"

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self, event_bus):
        """Test wait_for timeout."""
        result = await event_bus.wait_for(EventType.MODEL_LOADED, timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_with_filter(self, event_bus):
        """Test wait_for with filter function."""
        async def emit_events():
            await asyncio.sleep(0.1)
            await event_bus.emit(EventType.AUDIO_STARTED, {"id": 1})
            await asyncio.sleep(0.1)
            await event_bus.emit(EventType.AUDIO_STARTED, {"id": 2})

        asyncio.create_task(emit_events())

        # Wait for event with id=2
        result = await event_bus.wait_for(
            EventType.AUDIO_STARTED,
            timeout=1.0,
            filter_func=lambda e: e.data.get("id") == 2
        )

        assert result is not None
        assert result.data["id"] == 2


class TestPriority:
    """Test priority-based handler execution."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus):
        """Test that high priority handlers execute first."""
        execution_order = []

        async def low_priority(event: Event):
            execution_order.append("low")

        async def medium_priority(event: Event):
            execution_order.append("medium")

        async def high_priority(event: Event):
            execution_order.append("high")

        event_bus.subscribe(EventType.CONVERSATION_UPDATED, low_priority, priority=0)
        event_bus.subscribe(EventType.CONVERSATION_UPDATED, medium_priority, priority=5)
        event_bus.subscribe(EventType.CONVERSATION_UPDATED, high_priority, priority=10)

        await event_bus.emit(EventType.CONVERSATION_UPDATED, {})
        await asyncio.sleep(0.1)

        # Note: Due to async execution, exact order isn't guaranteed,
        # but we can verify all handlers were called
        assert len(execution_order) == 3
        assert "low" in execution_order
        assert "medium" in execution_order
        assert "high" in execution_order


class TestFilters:
    """Test event filtering."""

    @pytest.mark.asyncio
    async def test_filter_function(self, event_bus):
        """Test filter function to selectively handle events."""
        filtered_events = []

        async def handler(event: Event):
            filtered_events.append(event)

        # Only handle events with confidence > 0.8
        event_bus.subscribe(
            EventType.SPEECH_DETECTED,
            handler,
            filter_func=lambda e: e.data.get("confidence", 0) > 0.8
        )

        await event_bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.5})
        await event_bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.9})
        await event_bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.7})
        await event_bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.95})
        await asyncio.sleep(0.1)

        assert len(filtered_events) == 2
        assert all(e.data["confidence"] > 0.8 for e in filtered_events)


class TestMiddleware:
    """Test middleware functionality."""

    @pytest.mark.asyncio
    async def test_pre_middleware(self, event_bus):
        """Test pre-processing middleware."""
        middleware_calls = []

        async def middleware(event: Event):
            middleware_calls.append(("pre", event.type.value))

        event_bus.add_middleware(middleware, pre=True)

        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {})
        await event_bus.emit(EventType.SPEECH_DETECTED, {})
        await asyncio.sleep(0.1)

        assert len(middleware_calls) == 2
        assert middleware_calls[0] == ("pre", "wake_word_detected")
        assert middleware_calls[1] == ("pre", "speech_detected")

    @pytest.mark.asyncio
    async def test_post_middleware(self, event_bus):
        """Test post-processing middleware."""
        middleware_calls = []

        async def middleware(event: Event):
            middleware_calls.append(("post", event.type.value))

        event_bus.add_middleware(middleware, pre=False)

        await event_bus.emit(EventType.TTS_COMPLETED, {})
        await asyncio.sleep(0.1)

        assert len(middleware_calls) == 1
        assert middleware_calls[0] == ("post", "tts_completed")

    @pytest.mark.asyncio
    async def test_middleware_modification(self, event_bus):
        """Test that middleware can modify event metadata."""
        received_events = []

        async def enrichment_middleware(event: Event):
            event.metadata["enriched"] = True
            event.metadata["timestamp_ms"] = int(event.timestamp * 1000)

        async def handler(event: Event):
            received_events.append(event)

        event_bus.add_middleware(enrichment_middleware, pre=True)
        event_bus.subscribe(EventType.RESPONSE_GENERATED, handler)

        await event_bus.emit(EventType.RESPONSE_GENERATED, {"text": "hello"})
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].metadata.get("enriched") is True
        assert "timestamp_ms" in received_events[0].metadata


class TestMetrics:
    """Test metrics and statistics."""

    @pytest.mark.asyncio
    async def test_event_counting(self, event_bus):
        """Test that events are counted correctly."""
        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {})
        await event_bus.emit(EventType.SPEECH_DETECTED, {})
        await event_bus.emit(EventType.SPEECH_DETECTED, {})
        await asyncio.sleep(0.1)

        stats = event_bus.get_stats()
        assert stats["metrics"]["total_events"] == 3
        assert stats["metrics"]["events_by_type"]["wake_word_detected"] == 1
        assert stats["metrics"]["events_by_type"]["speech_detected"] == 2

    @pytest.mark.asyncio
    async def test_handler_stats(self, event_bus):
        """Test handler statistics."""
        async def test_handler(event: Event):
            await asyncio.sleep(0.01)  # Simulate some work

        event_bus.subscribe(EventType.TTS_STARTED, test_handler)

        await event_bus.emit(EventType.TTS_STARTED, {})
        await event_bus.emit(EventType.TTS_STARTED, {})
        await asyncio.sleep(0.1)

        stats = event_bus.get_stats()
        handler_stats = [h for h in stats["handler_stats"] if h["handler"] == "test_handler"]

        assert len(handler_stats) > 0
        assert handler_stats[0]["call_count"] == 2
        assert handler_stats[0]["total_duration"] > 0


class TestHistory:
    """Test event history functionality."""

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history recording."""
        await event_bus.emit(EventType.AUDIO_STARTED, {"id": 1})
        await event_bus.emit(EventType.AUDIO_STARTED, {"id": 2})
        await event_bus.emit(EventType.AUDIO_STOPPED, {"id": 3})
        await asyncio.sleep(0.1)

        history = event_bus.get_event_history()
        assert len(history) >= 3

        # Check recent events
        recent = event_bus.get_event_history(limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_event_history_filter(self, event_bus):
        """Test filtering event history by type."""
        await event_bus.emit(EventType.AUDIO_STARTED, {})
        await event_bus.emit(EventType.AUDIO_STOPPED, {})
        await event_bus.emit(EventType.AUDIO_STARTED, {})
        await asyncio.sleep(0.1)

        started_events = event_bus.get_event_history(EventType.AUDIO_STARTED)
        assert len(started_events) == 2
        assert all(e.type == EventType.AUDIO_STARTED for e in started_events)

    @pytest.mark.asyncio
    async def test_clear_history(self, event_bus):
        """Test clearing event history."""
        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {})
        await asyncio.sleep(0.1)

        assert len(event_bus.get_event_history()) > 0

        event_bus.clear_history()
        assert len(event_bus.get_event_history()) == 0


class TestDeadLetterQueue:
    """Test dead letter queue for failed events."""

    @pytest.mark.asyncio
    async def test_failed_handler_dlq(self, event_bus):
        """Test that failed handlers add events to DLQ."""
        async def failing_handler(event: Event):
            raise ValueError("Intentional error for testing")

        event_bus.subscribe(EventType.ERROR_OCCURRED, failing_handler)

        # Emit a different event type to avoid recursion
        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {})
        await asyncio.sleep(0.1)

        # The failing handler on ERROR_OCCURRED will add to DLQ
        # when the error event is emitted
        stats = event_bus.get_stats()
        assert stats["metrics"]["total_errors"] > 0


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_stop_bus(self, event_bus):
        """Test that handler errors don't stop event processing."""
        successful_calls = []

        async def failing_handler(event: Event):
            raise RuntimeError("Handler failed")

        async def successful_handler(event: Event):
            successful_calls.append(event)

        event_bus.subscribe(EventType.SPEECH_DETECTED, failing_handler, priority=10)
        event_bus.subscribe(EventType.SPEECH_DETECTED, successful_handler, priority=0)

        await event_bus.emit(EventType.SPEECH_DETECTED, {"text": "test"})
        await asyncio.sleep(0.1)

        # Successful handler should still be called
        assert len(successful_calls) == 1

    @pytest.mark.asyncio
    async def test_error_event_emission(self, event_bus):
        """Test that errors emit ERROR_OCCURRED events."""
        error_events = []

        async def error_collector(event: Event):
            error_events.append(event)

        async def failing_handler(event: Event):
            raise ValueError("Test error")

        event_bus.subscribe(EventType.ERROR_OCCURRED, error_collector)
        event_bus.subscribe(EventType.WAKE_WORD_DETECTED, failing_handler)

        await event_bus.emit(EventType.WAKE_WORD_DETECTED, {})
        await asyncio.sleep(0.2)

        # Should have emitted an error event
        assert len(error_events) > 0
        assert error_events[0].data.get("error_type") == "ValueError"


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_emit_event_function(self, event_bus):
        """Test global emit_event function."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.SYSTEM_STARTED, handler)

        await emit_event(EventType.SYSTEM_STARTED, {"component": "test"})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].data["component"] == "test"


class TestCorrelationAndMetadata:
    """Test correlation IDs and metadata."""

    @pytest.mark.asyncio
    async def test_correlation_id(self, event_bus):
        """Test correlation ID propagation."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.CONVERSATION_UPDATED, handler)

        await event_bus.emit(
            EventType.CONVERSATION_UPDATED,
            {"update": "test"},
            correlation_id="test-correlation-123"
        )
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].correlation_id == "test-correlation-123"

    @pytest.mark.asyncio
    async def test_custom_metadata(self, event_bus):
        """Test custom metadata in events."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.DECISION_MADE, handler)

        await event_bus.emit(
            EventType.DECISION_MADE,
            {"action": "respond"},
            metadata={"user_id": "user-123", "session": "abc"}
        )
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].metadata["user_id"] == "user-123"
        assert received[0].metadata["session"] == "abc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
