#!/usr/bin/env python3
"""
Demo script to test the enhanced event bus functionality.
Run this script to see all the new features in action.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.event_bus import EventBus, Event, EventType


async def main():
    """Run event bus demo."""
    print("=" * 70)
    print("Enhanced Event Bus Demo")
    print("=" * 70)

    bus = await EventBus.get_instance()
    await bus.start()

    print("\n1. Testing basic subscription...")
    async def basic_handler(event: Event):
        print(f"   ✓ Basic handler received: {event.type.value} - {event.data}")

    bus.subscribe([EventType.WAKE_WORD_DETECTED], basic_handler)
    await bus.emit(EventType.WAKE_WORD_DETECTED, {"confidence": 0.95})
    await asyncio.sleep(0.1)

    print("\n2. Testing one-time handler...")
    call_count = {"count": 0}

    async def once_handler(event: Event):
        call_count["count"] += 1
        print(f"   ✓ Once handler called (count: {call_count['count']})")

    bus.once(EventType.SPEECH_DETECTED, once_handler)
    await bus.emit(EventType.SPEECH_DETECTED, {"text": "First call"})
    await bus.emit(EventType.SPEECH_DETECTED, {"text": "Second call - should not trigger handler"})
    await asyncio.sleep(0.1)
    print(f"   → Handler called {call_count['count']} time(s) (expected: 1)")

    print("\n3. Testing wait_for pattern (request/response)...")
    async def emit_delayed():
        await asyncio.sleep(0.2)
        await bus.emit(EventType.TTS_COMPLETED, {"duration": 2.5})

    asyncio.create_task(emit_delayed())
    print("   → Waiting for TTS_COMPLETED event...")
    result = await bus.wait_for(EventType.TTS_COMPLETED, timeout=1.0)
    if result:
        print(f"   ✓ Received waited event: {result.data}")
    else:
        print("   ✗ Timeout waiting for event")

    print("\n4. Testing middleware...")
    async def logging_middleware(event: Event):
        print(f"   → [PRE-MIDDLEWARE] Processing event: {event.type.value}")

    bus.add_middleware(logging_middleware, pre=True)
    await bus.emit(EventType.RESPONSE_GENERATED, {"text": "Hello from AI"})
    await asyncio.sleep(0.1)

    print("\n5. Testing priority handlers...")
    async def low_priority(event: Event):
        print("   → Low priority handler (priority 0)")

    async def high_priority(event: Event):
        print("   → High priority handler (priority 10)")

    bus.subscribe(EventType.CONVERSATION_UPDATED, low_priority, priority=0)
    bus.subscribe(EventType.CONVERSATION_UPDATED, high_priority, priority=10)
    await bus.emit(EventType.CONVERSATION_UPDATED, {"update": "new message"})
    await asyncio.sleep(0.1)

    print("\n6. Testing filtered subscription...")
    filtered_count = {"count": 0}

    async def filtered_handler(event: Event):
        filtered_count["count"] += 1
        print(f"   ✓ Filtered handler: confidence={event.data['confidence']}")

    # Only handle high-confidence events
    bus.subscribe(
        EventType.SPEECH_DETECTED,
        filtered_handler,
        filter_func=lambda e: e.data.get("confidence", 0) > 0.8
    )

    print("   → Emitting events with varying confidence...")
    await bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.5, "text": "low"})
    await bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.9, "text": "high"})
    await bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.7, "text": "medium"})
    await bus.emit(EventType.SPEECH_DETECTED, {"confidence": 0.95, "text": "very high"})
    await asyncio.sleep(0.1)
    print(f"   → Filtered handler called {filtered_count['count']} times (expected: 2)")

    print("\n7. Event Bus Statistics:")
    print("-" * 70)
    stats = bus.get_stats()
    print(f"   Is running: {stats['is_running']}")
    print(f"   Total events processed: {stats['metrics'].get('total_events', 0)}")
    print(f"   Total handlers registered: {stats['total_handlers']}")
    print(f"   Queue size: {stats['queue_size']}")
    print(f"   Event history: {stats['event_history_size']} events")
    print(f"   Middleware: {stats['middleware_count']['pre']} pre, {stats['middleware_count']['post']} post")
    print(f"   Average processing time: {stats['metrics'].get('avg_processing_time', 0):.6f}s")

    print("\n   Events by type:")
    for event_type, count in sorted(stats['metrics'].get('events_by_type', {}).items()):
        print(f"      - {event_type}: {count}")

    print("\n8. Recent Event History:")
    print("-" * 70)
    history = bus.get_event_history(limit=5)
    for i, event in enumerate(history[-5:], 1):
        print(f"   {i}. {event.type.value} from '{event.source}' at {event.timestamp:.2f}")
        print(f"      Data: {event.data}")

    print("\n9. Testing correlation IDs and metadata...")
    received_events = []

    async def correlation_handler(event: Event):
        received_events.append(event)
        print(f"   ✓ Received event with correlation_id: {event.correlation_id}")
        print(f"      Metadata: {event.metadata}")

    bus.subscribe(EventType.DECISION_MADE, correlation_handler)
    await bus.emit(
        EventType.DECISION_MADE,
        {"action": "respond"},
        correlation_id="request-12345",
        metadata={"user": "alice", "session": "abc-123"}
    )
    await asyncio.sleep(0.1)

    print("\n10. Testing error handling...")
    async def failing_handler(event: Event):
        raise ValueError("This handler intentionally fails!")

    async def working_handler(event: Event):
        print("   ✓ This handler works despite other handlers failing")

    error_count_before = stats['metrics'].get('total_errors', 0)

    bus.subscribe(EventType.MODEL_LOADED, failing_handler, priority=10)
    bus.subscribe(EventType.MODEL_LOADED, working_handler, priority=0)

    await bus.emit(EventType.MODEL_LOADED, {"model": "test-model"})
    await asyncio.sleep(0.2)

    stats = bus.get_stats()
    error_count_after = stats['metrics'].get('total_errors', 0)
    print(f"   → Errors detected: {error_count_after - error_count_before}")
    print(f"   → Dead letter queue size: {stats['dead_letter_queue_size']}")

    # Show final statistics
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    stats = bus.get_stats()
    print(f"Total events processed: {stats['metrics'].get('total_events', 0)}")
    print(f"Total errors: {stats['metrics'].get('total_errors', 0)}")
    print(f"Queue overflows: {stats['metrics'].get('queue_overflows', 0)}")
    print(f"Average processing time: {stats['metrics'].get('avg_processing_time', 0):.6f}s")

    await bus.stop()
    print("\n✓ Event bus stopped successfully")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
