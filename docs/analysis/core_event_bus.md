# event_bus.py - Event-Driven Message Bus Analysis

## Overview

`event_bus.py` implements a sophisticated pub/sub event system that serves as the communication backbone of the voice assistant. It provides async event dispatching, middleware support, request/response patterns, and comprehensive monitoring.

## File Location
`/home/user/cluster/src/core/event_bus.py`

## Key Features
- Singleton pattern for global event bus access
- Async event processing with priority-based handlers
- One-time subscriptions (Mycroft pattern)
- Request/response pattern with `wait_for`
- Wildcard event subscriptions
- Event history and replay
- Dead letter queue for failed events
- Middleware for pre/post processing

## Classes

### EventType (Enum)

Comprehensive enum defining all event types in the system:

| Category | Events |
|----------|--------|
| Audio | `AUDIO_STARTED`, `AUDIO_STOPPED`, `WAKE_WORD_DETECTED`, `SPEECH_DETECTED`, `SPEECH_ENDED` |
| Speech | `AMBIENT_SPEECH_DETECTED`, `WAKEWORD_SPEECH_DETECTED` |
| AI | `CONVERSATION_UPDATED`, `DECISION_MADE`, `RESPONSE_GENERATING`, `RESPONSE_GENERATED` |
| TTS | `TTS_STARTED`, `PHONEME_EVENT`, `TTS_COMPLETED` |
| Lip Sync | `LIP_SYNC_READY`, `LIP_SYNC_STARTED`, `LIP_SYNC_VISEME`, `LIP_SYNC_COMPLETED` |
| Animation | `EXPRESSION_CHANGE`, `GAZE_UPDATE`, `MOUTH_SHAPE_UPDATE`, `BLINK_TRIGGERED` |
| System | `SYSTEM_STARTED`, `SYSTEM_STOPPED`, `ERROR_OCCURRED`, `HEALTH_CHECK` |

### Event (Dataclass)

**Attributes**:
- `type: EventType` - Event classification
- `data: Dict[str, Any]` - Event payload
- `timestamp: float` - Unix timestamp
- `source: str` - Event originator
- `correlation_id: Optional[str]` - Request tracing ID
- `reply_to: Optional[str]` - For request/response pattern
- `metadata: Dict[str, Any]` - Additional context

**Methods**:
- `to_dict()` - Serialize for storage/transmission
- `from_dict()` - Deserialize from dictionary

### EventHandler

Wraps handler functions with execution context.

**Attributes**:
- `handler_func: Callable` - The actual handler
- `event_types: List[EventType]` - Subscribed events
- `priority: int` - Execution order (higher = first)
- `filter_func: Optional[Callable]` - Custom filtering
- `once: bool` - One-time execution flag
- `pattern: Optional[str]` - Wildcard matching

**Methods**:
- `matches(event)` - Check if handler should process event
- `handle(event)` - Execute handler with error recovery
- `get_stats()` - Performance statistics

### EventBus

Central event bus with singleton pattern.

**Configuration**:
- `max_queue_size: int = 1000` - Queue capacity
- `enable_history: bool = True` - Event history
- `history_size: int = 100` - History buffer size
- `enable_metrics: bool = True` - Performance tracking

**Key Methods**:

| Method | Purpose | Async |
|--------|---------|-------|
| `get_instance()` | Get singleton instance | Yes |
| `start()` | Begin event processing | Yes |
| `stop()` | Stop event processing | Yes |
| `emit()` | Publish an event | Yes |
| `subscribe()` | Register handler | No |
| `once()` | One-time subscription | No |
| `wait_for()` | Request/response pattern | Yes |
| `unsubscribe()` | Remove handler | No |
| `add_middleware()` | Pre/post processing | No |
| `get_stats()` | Comprehensive statistics | No |
| `replay_events()` | Replay from history | Yes |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        EventBus                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Event Queue │→ │ Dispatcher  │→ │ Handler Registry    │  │
│  │ (async)     │  │             │  │ (by EventType)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          ↓                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Middleware  │  │ History     │  │ Dead Letter Queue   │  │
│  │ (pre/post)  │  │ (replay)    │  │ (failed events)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                      Metrics                           │ │
│  │ total_events | errors | avg_processing_time | overflows│ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Event Flow

```
emit()
  ↓
Queue Event
  ↓
_process_events() loop
  ↓
_dispatch_event()
  ├── Run pre-middleware
  ├── Get handlers sorted by priority
  ├── Execute handlers concurrently
  │   ├── If error → add to dead_letter_queue
  │   └── If reply_to → resolve wait futures
  ├── Remove one-time handlers
  ├── Run post-middleware
  └── Update metrics
```

## Usage Examples

### Basic Subscription
```python
bus = await EventBus.get_instance()
await bus.start()

def on_speech(event):
    print(f"Speech: {event.data['text']}")

bus.subscribe(EventType.SPEECH_DETECTED, on_speech)
```

### One-Time Handler
```python
bus.once(EventType.WAKE_WORD_DETECTED, lambda e: print("Wake word!"))
```

### Request/Response Pattern
```python
# Emit event and wait for response
await bus.emit(EventType.RESPONSE_GENERATING, {"prompt": "Hello"})
response = await bus.wait_for(EventType.RESPONSE_GENERATED, timeout=5.0)
```

### Middleware
```python
async def log_middleware(event):
    logger.info(f"Processing: {event.type.value}")

bus.add_middleware(log_middleware, pre=True)
```

## Improvements Suggested

### 1. Event Batching
For high-frequency events (like audio frames):
```python
async def emit_batch(self, events: List[Tuple[EventType, Dict]]) -> None:
    """Emit multiple events as a batch for efficiency."""
    for event_type, data in events:
        await self.emit(event_type, data)
```

### 2. Event Persistence
Add optional persistence for system recovery:
```python
async def persist_event(self, event: Event) -> None:
    """Persist event to disk for recovery."""
    # Write to SQLite or append-only log
```

### 3. Event Schema Validation
Add type validation for event data:
```python
EVENT_SCHEMAS = {
    EventType.SPEECH_DETECTED: {"text": str, "confidence": float}
}
```

### 4. Backpressure Handling
Implement backpressure when queue is near capacity:
```python
if self.event_queue.qsize() > self.max_queue_size * 0.8:
    logger.warning("Event queue approaching capacity")
    await asyncio.sleep(0.1)  # Brief pause
```

### 5. Handler Timeout
Add timeout protection for slow handlers:
```python
async def handle_with_timeout(self, handler, event, timeout=5.0):
    try:
        await asyncio.wait_for(handler.handle(event), timeout)
    except asyncio.TimeoutError:
        logger.error(f"Handler {handler.handler_func.__name__} timed out")
```

### 6. Event Correlation Graphs
Track event chains for debugging:
```python
def get_event_chain(self, correlation_id: str) -> List[Event]:
    """Get all events with same correlation_id."""
    return [e for e in self.event_history if e.correlation_id == correlation_id]
```

## Thread Safety

- Uses `asyncio.Lock` for singleton creation
- Event queue is thread-safe via `asyncio.Queue`
- Handler registration is not thread-safe (should be done during initialization)

## Performance Considerations

- Handlers execute concurrently via `asyncio.gather()`
- Priority sorting on each dispatch (consider caching sorted handlers)
- History and metrics have bounded memory usage (deque with maxlen)
- Queue overflow drops events with warning
