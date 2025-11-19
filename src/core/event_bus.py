"""
Event-driven message bus for voice assistant communication.
Supports pub/sub pattern with type safety and async operations.

Enhanced with production patterns from Mycroft AI, Rhasspy, and other voice assistants:
- One-time subscriptions (once)
- Request/response pattern (wait_for)
- Wildcard event subscriptions
- Event history and replay
- Middleware for pre/post processing
- Dead letter queue for failed events
- Enhanced metrics and monitoring
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Core event types for the voice assistant system."""
    
    # Audio Events
    AUDIO_STARTED = "audio_started"
    AUDIO_STOPPED = "audio_stopped"
    WAKE_WORD_DETECTED = "wake_word_detected"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"
    SILENCE_DETECTED = "silence_detected"
    AUDIO_PLAYBACK_STARTED = "audio_playback_started"
    AUDIO_PLAYBACK_PROGRESS = "audio_playback_progress"
    AUDIO_PLAYBACK_ENDED = "audio_playback_ended"

    # Ambient Speech Events
    AMBIENT_SPEECH_DETECTED = "ambient_speech_detected"  # Ambient mode speech
    WAKEWORD_SPEECH_DETECTED = "wakeword_speech_detected"  # Wake word triggered speech
    
    # AI/Decision Events
    CONVERSATION_UPDATED = "conversation_updated"
    DECISION_MADE = "decision_made"
    RESPONSE_GENERATING = "response_generating"
    RESPONSE_GENERATED = "response_generated"
    EMOTION_CHANGED = "emotion_changed"

    # Model Management Events
    MODEL_DOWNLOAD_STARTED = "model_download_started"
    MODEL_DOWNLOAD_PROGRESS = "model_download_progress"
    MODEL_DOWNLOAD_COMPLETED = "model_download_completed"
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    PROVIDER_SWITCHED = "provider_switched"
    
    # TTS Events
    TTS_STARTED = "tts_started"
    PHONEME_EVENT = "phoneme_event"
    TTS_COMPLETED = "tts_completed"
    
    # Animation Events
    EXPRESSION_CHANGE = "expression_change"
    GAZE_UPDATE = "gaze_update"
    MOUTH_SHAPE_UPDATE = "mouth_shape_update"
    BLINK_TRIGGERED = "blink_triggered"
    
    # Camera Events
    PERSON_DETECTED = "person_detected"
    PERSON_APPROACHING = "person_approaching"
    GESTURE_DETECTED = "gesture_detected"
    OBJECT_IN_VIEW = "object_in_view"
    SCENE_UPDATE = "scene_update"
    EMOTION_DETECTED = "emotion_detected"
    
    # System Events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    ERROR_OCCURRED = "error_occurred"
    HEALTH_CHECK = "health_check"

    # Event Bus Events
    EVENT_FAILED = "event_failed"
    EVENT_DEADLETTER = "event_deadletter"
    HANDLER_REGISTERED = "handler_registered"
    HANDLER_UNREGISTERED = "handler_unregistered"


@dataclass
class Event:
    """Base event structure with metadata."""
    type: EventType
    data: Dict[str, Any]
    timestamp: float
    source: str
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None  # For request/response pattern
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            type=EventType(data["type"]),
            data=data["data"],
            timestamp=data["timestamp"],
            source=data["source"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {})
        )


class EventHandler:
    """Handler for processing events with async support."""

    def __init__(self, handler_func: Callable, event_types: List[EventType],
                 priority: int = 0, filter_func: Optional[Callable] = None,
                 once: bool = False, pattern: Optional[str] = None):
        self.handler_func = handler_func
        self.event_types = event_types
        self.priority = priority  # Higher priority = processed first
        self.filter_func = filter_func
        self.is_async = asyncio.iscoroutinefunction(handler_func)
        self.once = once  # One-time handler (Mycroft pattern)
        self.pattern = pattern  # Wildcard pattern for event matching
        self.call_count = 0
        self.last_called = 0.0
        self.total_duration = 0.0
        self._compiled_pattern = None  # Compiled regex pattern

        # Compile pattern if provided
        if pattern:
            self._compiled_pattern = re.compile(pattern.replace("*", ".*"))

    def matches(self, event: Event) -> bool:
        """Check if this handler matches the event."""
        # Check event type match
        if event.type not in self.event_types:
            # Check pattern match if pattern is provided
            if self._compiled_pattern:
                if not self._compiled_pattern.match(event.type.value):
                    return False
            else:
                return False

        # Check filter function
        if self.filter_func and not self.filter_func(event):
            return False

        return True

    async def handle(self, event: Event) -> Optional[Any]:
        """Handle an event."""
        start_time = time.time()
        try:
            if not self.matches(event):
                return None

            self.call_count += 1
            self.last_called = start_time

            if self.is_async:
                result = await self.handler_func(event)
            else:
                result = self.handler_func(event)

            self.total_duration += time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Error in event handler {self.handler_func.__name__}: {e}", exc_info=True)
            # Emit error event (avoid recursion by checking event type)
            if event.type != EventType.ERROR_OCCURRED:
                bus = EventBus._instance
                if bus:
                    await bus.emit(EventType.ERROR_OCCURRED, {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "handler": self.handler_func.__name__,
                        "event_type": event.type.value,
                        "source": event.source
                    })
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        avg_duration = self.total_duration / self.call_count if self.call_count > 0 else 0.0
        return {
            "handler": self.handler_func.__name__,
            "call_count": self.call_count,
            "last_called": self.last_called,
            "average_duration": avg_duration,
            "total_duration": self.total_duration,
            "is_once": self.once,
            "pattern": self.pattern
        }


class EventBus:
    """Central event bus for pub/sub communication with enhanced features."""

    _instance: Optional["EventBus"] = None
    _lock = asyncio.Lock()

    def __init__(self, max_queue_size: int = 1000, enable_history: bool = True,
                 history_size: int = 100, enable_metrics: bool = True):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Event history for replay and debugging
        self.enable_history = enable_history
        self.event_history: deque = deque(maxlen=history_size)

        # Middleware for pre/post processing
        self.middleware_pre: List[Callable] = []
        self.middleware_post: List[Callable] = []

        # Dead letter queue for failed events
        self.dead_letter_queue: deque = deque(maxlen=100)

        # Wait futures for request/response pattern
        self._wait_futures: Dict[str, asyncio.Future] = {}

        # Metrics
        self.enable_metrics = enable_metrics
        self.metrics = {
            "total_events": 0,
            "total_errors": 0,
            "events_by_type": defaultdict(int),
            "handler_errors": defaultdict(int),
            "queue_overflows": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
    
    @classmethod
    async def get_instance(cls) -> "EventBus":
        """Get singleton instance of EventBus."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus processing loop."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Main event processing loop."""
        while self.is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to all registered handlers with middleware support."""
        start_time = time.time()

        try:
            # Run pre-middleware
            for middleware in self.middleware_pre:
                try:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(event)
                    else:
                        middleware(event)
                except Exception as e:
                    logger.error(f"Error in pre-middleware: {e}")

            # Get handlers for this event type
            handlers = self.handlers.get(event.type, [])

            # Sort by priority (higher priority first)
            handlers.sort(key=lambda h: h.priority, reverse=True)

            # Execute handlers concurrently
            tasks = []
            handlers_to_remove = []

            for handler in handlers:
                task = asyncio.create_task(handler.handle(event))
                tasks.append((handler, task))

                # Mark one-time handlers for removal
                if handler.once:
                    handlers_to_remove.append(handler)

            # Wait for all handlers to complete
            if tasks:
                results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

                # Check for errors and responses
                for (handler, task), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        self.metrics["total_errors"] += 1
                        self.metrics["handler_errors"][handler.handler_func.__name__] += 1
                        logger.error(f"Handler {handler.handler_func.__name__} failed: {result}")

                        # Add to dead letter queue
                        self.dead_letter_queue.append({
                            "event": event,
                            "handler": handler.handler_func.__name__,
                            "error": str(result),
                            "timestamp": time.time()
                        })
                    elif event.reply_to and result is not None:
                        # Handle request/response pattern
                        if event.reply_to in self._wait_futures:
                            future = self._wait_futures[event.reply_to]
                            if not future.done():
                                future.set_result(result)

            # Remove one-time handlers
            for handler in handlers_to_remove:
                if handler in self.handlers[event.type]:
                    self.handlers[event.type].remove(handler)

            # Run post-middleware
            for middleware in self.middleware_post:
                try:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(event)
                    else:
                        middleware(event)
                except Exception as e:
                    logger.error(f"Error in post-middleware: {e}")

            # Update metrics
            if self.enable_metrics:
                processing_time = time.time() - start_time
                self.metrics["total_processing_time"] += processing_time
                self.metrics["avg_processing_time"] = (
                    self.metrics["total_processing_time"] / self.metrics["total_events"]
                    if self.metrics["total_events"] > 0 else 0.0
                )

        except Exception as e:
            logger.error(f"Error dispatching event {event.type.value}: {e}", exc_info=True)
            self.metrics["total_errors"] += 1
    
    async def emit(self, event_type: EventType, data: Dict[str, Any],
                   source: str = "system", correlation_id: Optional[str] = None,
                   reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to the bus."""
        event = Event(
            type=event_type,
            data=data,
            timestamp=time.time(),
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            metadata=metadata or {}
        )

        try:
            self.event_queue.put_nowait(event)

            # Add to history
            if self.enable_history:
                self.event_history.append(event)

            # Update metrics
            if self.enable_metrics:
                self.metrics["total_events"] += 1
                self.metrics["events_by_type"][event_type.value] += 1

        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event_type.value}")
            self.metrics["queue_overflows"] += 1
    
    def subscribe(self, event_types: Union[EventType, List[EventType]],
                  handler: Callable, priority: int = 0,
                  filter_func: Optional[Callable] = None,
                  pattern: Optional[str] = None) -> None:
        """Subscribe to events with optional wildcard pattern."""
        if isinstance(event_types, EventType):
            event_types = [event_types]

        event_handler = EventHandler(handler, event_types, priority, filter_func,
                                     once=False, pattern=pattern)

        for event_type in event_types:
            self.handlers[event_type].append(event_handler)

        logger.debug(f"Subscribed {handler.__name__} to {[et.value for et in event_types]}")

    def once(self, event_types: Union[EventType, List[EventType]],
             handler: Callable, priority: int = 0,
             filter_func: Optional[Callable] = None) -> None:
        """Subscribe to events with one-time execution (Mycroft pattern)."""
        if isinstance(event_types, EventType):
            event_types = [event_types]

        event_handler = EventHandler(handler, event_types, priority, filter_func, once=True)

        for event_type in event_types:
            self.handlers[event_type].append(event_handler)

        logger.debug(f"One-time subscription: {handler.__name__} to {[et.value for et in event_types]}")

    async def wait_for(self, event_type: EventType, timeout: float = 5.0,
                      filter_func: Optional[Callable] = None) -> Optional[Event]:
        """Wait for a specific event with timeout (request/response pattern)."""
        future = asyncio.Future()
        wait_id = f"wait_{event_type.value}_{time.time()}"

        async def wait_handler(event: Event):
            if not future.done():
                future.set_result(event)

        # Subscribe with once handler
        self.once(event_type, wait_handler, filter_func=filter_func)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for event {event_type.value}")
            return None
        finally:
            # Clean up handler
            self.unsubscribe(wait_handler)

    def add_middleware(self, middleware: Callable, pre: bool = True) -> None:
        """Add middleware for event pre/post processing."""
        if pre:
            self.middleware_pre.append(middleware)
            logger.debug(f"Added pre-middleware: {middleware.__name__}")
        else:
            self.middleware_post.append(middleware)
            logger.debug(f"Added post-middleware: {middleware.__name__}")

    def remove_middleware(self, middleware: Callable) -> None:
        """Remove middleware."""
        if middleware in self.middleware_pre:
            self.middleware_pre.remove(middleware)
        if middleware in self.middleware_post:
            self.middleware_post.remove(middleware)
        logger.debug(f"Removed middleware: {middleware.__name__}")
    
    def unsubscribe(self, handler: Callable) -> None:
        """Unsubscribe from all events."""
        for event_type in self.handlers:
            self.handlers[event_type] = [
                h for h in self.handlers[event_type] 
                if h.handler_func != handler
            ]
        logger.debug(f"Unsubscribed {handler.__name__}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive event bus statistics."""
        handler_stats = []
        for event_type, handlers in self.handlers.items():
            for handler in handlers:
                handler_stats.append({
                    "event_type": event_type.value,
                    **handler.get_stats()
                })

        return {
            "is_running": self.is_running,
            "queue_size": self.event_queue.qsize(),
            "total_handlers": sum(len(handlers) for handlers in self.handlers.values()),
            "handlers_by_type": {
                event_type.value: len(handlers)
                for event_type, handlers in self.handlers.items()
            },
            "metrics": dict(self.metrics) if self.enable_metrics else {},
            "handler_stats": handler_stats,
            "event_history_size": len(self.event_history) if self.enable_history else 0,
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "middleware_count": {
                "pre": len(self.middleware_pre),
                "post": len(self.middleware_post)
            }
        }

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 10) -> List[Event]:
        """Get recent event history, optionally filtered by type."""
        if not self.enable_history:
            return []

        if event_type:
            events = [e for e in self.event_history if e.type == event_type]
        else:
            events = list(self.event_history)

        return events[-limit:]

    def get_dead_letter_queue(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed events from dead letter queue."""
        return list(self.dead_letter_queue)[-limit:]

    async def replay_events(self, events: List[Event]) -> None:
        """Replay events from history (useful for debugging/recovery)."""
        logger.info(f"Replaying {len(events)} events")
        for event in events:
            await self.emit(
                event.type,
                event.data,
                source=f"replay_{event.source}",
                correlation_id=event.correlation_id,
                reply_to=event.reply_to,
                metadata={**event.metadata, "replayed": True}
            )

    def clear_history(self) -> None:
        """Clear event history."""
        self.event_history.clear()
        logger.debug("Event history cleared")

    def clear_dead_letter_queue(self) -> None:
        """Clear dead letter queue."""
        self.dead_letter_queue.clear()
        logger.debug("Dead letter queue cleared")

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "total_events": 0,
            "total_errors": 0,
            "events_by_type": defaultdict(int),
            "handler_errors": defaultdict(int),
            "queue_overflows": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        logger.debug("Metrics reset")


# Convenience functions for easy access
async def emit_event(event_type: EventType, data: Dict[str, Any],
                    source: str = "system", correlation_id: Optional[str] = None,
                    reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Emit an event to the global event bus."""
    bus = await EventBus.get_instance()
    await bus.emit(event_type, data, source, correlation_id, reply_to, metadata)


def subscribe_to_events(event_types: Union[EventType, List[EventType]],
                       handler: Callable, priority: int = 0,
                       filter_func: Optional[Callable] = None,
                       pattern: Optional[str] = None) -> None:
    """Subscribe to events on the global event bus."""
    bus = EventBus._instance
    if bus:
        bus.subscribe(event_types, handler, priority, filter_func, pattern)
    else:
        logger.warning("Event bus not initialized, subscription will be added when bus starts")


def once_event(event_types: Union[EventType, List[EventType]],
              handler: Callable, priority: int = 0,
              filter_func: Optional[Callable] = None) -> None:
    """Subscribe to events with one-time execution."""
    bus = EventBus._instance
    if bus:
        bus.once(event_types, handler, priority, filter_func)
    else:
        logger.warning("Event bus not initialized, subscription will be added when bus starts")


async def wait_for_event(event_type: EventType, timeout: float = 5.0,
                        filter_func: Optional[Callable] = None) -> Optional[Event]:
    """Wait for a specific event with timeout."""
    bus = await EventBus.get_instance()
    return await bus.wait_for(event_type, timeout, filter_func)


# Example usage and testing
if __name__ == "__main__":
    async def test_event_bus():
        """Test enhanced event bus functionality."""
        bus = await EventBus.get_instance()
        await bus.start()

        print("=== Testing Enhanced Event Bus ===\n")

        # 1. Test basic subscription
        print("1. Testing basic subscription...")
        async def basic_handler(event: Event):
            print(f"   ✓ Basic handler received: {event.type.value} - {event.data}")

        bus.subscribe([EventType.WAKE_WORD_DETECTED], basic_handler)
        await bus.emit(EventType.WAKE_WORD_DETECTED, {"confidence": 0.95})
        await asyncio.sleep(0.1)

        # 2. Test one-time handler
        print("\n2. Testing one-time handler...")
        call_count = {"count": 0}

        async def once_handler(event: Event):
            call_count["count"] += 1
            print(f"   ✓ Once handler called (count: {call_count['count']})")

        bus.once(EventType.SPEECH_DETECTED, once_handler)
        await bus.emit(EventType.SPEECH_DETECTED, {"text": "First call"})
        await bus.emit(EventType.SPEECH_DETECTED, {"text": "Second call"})
        await asyncio.sleep(0.1)
        print(f"   → Handler called {call_count['count']} time(s) (should be 1)")

        # 3. Test wait_for pattern
        print("\n3. Testing wait_for pattern...")
        async def emit_delayed():
            await asyncio.sleep(0.2)
            await bus.emit(EventType.TTS_COMPLETED, {"duration": 2.5})

        asyncio.create_task(emit_delayed())
        result = await bus.wait_for(EventType.TTS_COMPLETED, timeout=1.0)
        if result:
            print(f"   ✓ Received waited event: {result.data}")
        else:
            print("   ✗ Timeout waiting for event")

        # 4. Test middleware
        print("\n4. Testing middleware...")
        async def logging_middleware(event: Event):
            print(f"   → [PRE] Processing event: {event.type.value}")

        bus.add_middleware(logging_middleware, pre=True)
        await bus.emit(EventType.RESPONSE_GENERATED, {"text": "Hello"})
        await asyncio.sleep(0.1)

        # 5. Test priority handlers
        print("\n5. Testing priority handlers...")
        async def low_priority(event: Event):
            print("   → Low priority handler (priority 0)")

        async def high_priority(event: Event):
            print("   → High priority handler (priority 10)")

        bus.subscribe(EventType.CONVERSATION_UPDATED, low_priority, priority=0)
        bus.subscribe(EventType.CONVERSATION_UPDATED, high_priority, priority=10)
        await bus.emit(EventType.CONVERSATION_UPDATED, {"update": "test"})
        await asyncio.sleep(0.1)

        # 6. Show statistics
        print("\n6. Event bus statistics:")
        stats = bus.get_stats()
        print(f"   Total events: {stats['metrics'].get('total_events', 0)}")
        print(f"   Total handlers: {stats['total_handlers']}")
        print(f"   Queue size: {stats['queue_size']}")
        print(f"   Event history: {stats['event_history_size']} events")
        print(f"   Middleware: {stats['middleware_count']['pre']} pre, {stats['middleware_count']['post']} post")

        # 7. Show event history
        print("\n7. Recent event history:")
        history = bus.get_event_history(limit=5)
        for event in history[-3:]:
            print(f"   - {event.type.value} from {event.source} at {event.timestamp:.2f}")

        # Stop the bus
        await bus.stop()
        print("\n=== Test Complete ===")

    # Run test
    asyncio.run(test_event_bus())
