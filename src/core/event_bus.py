"""
Event-driven message bus for voice assistant communication.
Supports pub/sub pattern with type safety and async operations.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from collections import defaultdict

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


@dataclass
class Event:
    """Base event structure with metadata."""
    type: EventType
    data: Dict[str, Any]
    timestamp: float
    source: str
    correlation_id: Optional[str] = None
    
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
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            type=EventType(data["type"]),
            data=data["data"],
            timestamp=data["timestamp"],
            source=data["source"],
            correlation_id=data.get("correlation_id")
        )


class EventHandler:
    """Handler for processing events with async support."""
    
    def __init__(self, handler_func: Callable, event_types: List[EventType], 
                 priority: int = 0, filter_func: Optional[Callable] = None):
        self.handler_func = handler_func
        self.event_types = event_types
        self.priority = priority  # Higher priority = processed first
        self.filter_func = filter_func
        self.is_async = asyncio.iscoroutinefunction(handler_func)
    
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        try:
            if self.filter_func and not self.filter_func(event):
                return
            
            if self.is_async:
                await self.handler_func(event)
            else:
                self.handler_func(event)
        except Exception as e:
            logger.error(f"Error in event handler {self.handler_func.__name__}: {e}")
            # Emit error event
            await EventBus.get_instance().emit(EventType.ERROR_OCCURRED, {
                "error": str(e),
                "handler": self.handler_func.__name__,
                "event_type": event.type.value
            })


class EventBus:
    """Central event bus for pub/sub communication."""
    
    _instance: Optional["EventBus"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
    
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
        """Dispatch event to all registered handlers."""
        handlers = self.handlers.get(event.type, [])
        
        # Sort by priority (higher priority first)
        handlers.sort(key=lambda h: h.priority, reverse=True)
        
        # Execute handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(handler.handle(event))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit(self, event_type: EventType, data: Dict[str, Any], 
                   source: str = "system", correlation_id: Optional[str] = None) -> None:
        """Emit an event to the bus."""
        event = Event(
            type=event_type,
            data=data,
            timestamp=time.time(),
            source=source,
            correlation_id=correlation_id
        )
        
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event_type.value}")
    
    def subscribe(self, event_types: Union[EventType, List[EventType]], 
                  handler: Callable, priority: int = 0, 
                  filter_func: Optional[Callable] = None) -> None:
        """Subscribe to events."""
        if isinstance(event_types, EventType):
            event_types = [event_types]
        
        event_handler = EventHandler(handler, event_types, priority, filter_func)
        
        for event_type in event_types:
            self.handlers[event_type].append(event_handler)
        
        logger.debug(f"Subscribed {handler.__name__} to {[et.value for et in event_types]}")
    
    def unsubscribe(self, handler: Callable) -> None:
        """Unsubscribe from all events."""
        for event_type in self.handlers:
            self.handlers[event_type] = [
                h for h in self.handlers[event_type] 
                if h.handler_func != handler
            ]
        logger.debug(f"Unsubscribed {handler.__name__}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "is_running": self.is_running,
            "queue_size": self.event_queue.qsize(),
            "total_handlers": sum(len(handlers) for handlers in self.handlers.values()),
            "handlers_by_type": {
                event_type.value: len(handlers) 
                for event_type, handlers in self.handlers.items()
            }
        }


# Convenience functions for easy access
async def emit_event(event_type: EventType, data: Dict[str, Any], 
                    source: str = "system", correlation_id: Optional[str] = None) -> None:
    """Emit an event to the global event bus."""
    bus = await EventBus.get_instance()
    await bus.emit(event_type, data, source, correlation_id)


def subscribe_to_events(event_types: Union[EventType, List[EventType]], 
                       handler: Callable, priority: int = 0, 
                       filter_func: Optional[Callable] = None) -> None:
    """Subscribe to events on the global event bus."""
    bus = EventBus._instance
    if bus:
        bus.subscribe(event_types, handler, priority, filter_func)
    else:
        logger.warning("Event bus not initialized, subscription will be added when bus starts")


# Example usage and testing
if __name__ == "__main__":
    async def test_event_bus():
        """Test the event bus functionality."""
        bus = await EventBus.get_instance()
        await bus.start()
        
        # Test handler
        async def test_handler(event: Event):
            print(f"Received event: {event.type.value} with data: {event.data}")
        
        # Subscribe to events
        bus.subscribe([EventType.WAKE_WORD_DETECTED, EventType.SPEECH_DETECTED], test_handler)
        
        # Emit test events
        await bus.emit(EventType.WAKE_WORD_DETECTED, {"confidence": 0.95})
        await bus.emit(EventType.SPEECH_DETECTED, {"text": "Hello world"})
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop the bus
        await bus.stop()
    
    # Run test
    asyncio.run(test_event_bus())
