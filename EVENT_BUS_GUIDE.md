# Event Bus Guide

## Overview

The enhanced Event Bus provides a robust, production-ready pub/sub messaging system for the voice assistant. It's designed based on patterns from successful voice assistants like Mycroft AI, Rhasspy, and Home Assistant.

## Features

### Core Features
- ✅ **Async-first design** using Python asyncio
- ✅ **Type-safe** with EventType enum and Event dataclasses
- ✅ **Priority-based handlers** for execution order control
- ✅ **Filter functions** for selective event processing
- ✅ **Error resilience** with dead letter queue
- ✅ **Comprehensive metrics** and performance tracking

### Advanced Features
- ✅ **One-time subscriptions** (`once`) - Mycroft pattern
- ✅ **Request/response pattern** (`wait_for`) with timeout
- ✅ **Wildcard patterns** for flexible event matching
- ✅ **Middleware support** for pre/post processing
- ✅ **Event history** for debugging and replay
- ✅ **Dead letter queue** for failed events
- ✅ **Correlation IDs** for request tracing
- ✅ **Custom metadata** for event enrichment

## Quick Start

### Basic Usage

```python
import asyncio
from core.event_bus import EventBus, EventType, Event

async def main():
    # Get event bus instance (singleton)
    bus = await EventBus.get_instance()
    await bus.start()

    # Define a handler
    async def my_handler(event: Event):
        print(f"Received: {event.type.value} - {event.data}")

    # Subscribe to events
    bus.subscribe(EventType.SPEECH_DETECTED, my_handler)

    # Emit an event
    await bus.emit(EventType.SPEECH_DETECTED, {"text": "Hello world"})

    await asyncio.sleep(0.1)  # Wait for processing
    await bus.stop()

asyncio.run(main())
```

### Using Convenience Functions

```python
from core.event_bus import emit_event, subscribe_to_events, EventType

# These work with the global singleton instance
async def handler(event):
    print(f"Got event: {event.data}")

subscribe_to_events(EventType.WAKE_WORD_DETECTED, handler)
await emit_event(EventType.WAKE_WORD_DETECTED, {"confidence": 0.95})
```

## Event Types

All available event types are defined in the `EventType` enum:

### Audio Events
- `AUDIO_STARTED`, `AUDIO_STOPPED`
- `WAKE_WORD_DETECTED`
- `SPEECH_DETECTED`, `SPEECH_ENDED`
- `SILENCE_DETECTED`
- `AUDIO_PLAYBACK_STARTED`, `AUDIO_PLAYBACK_PROGRESS`, `AUDIO_PLAYBACK_ENDED`

### AI/Decision Events
- `CONVERSATION_UPDATED`
- `DECISION_MADE`
- `RESPONSE_GENERATING`, `RESPONSE_GENERATED`
- `EMOTION_CHANGED`

### Model Management Events
- `MODEL_DOWNLOAD_STARTED`, `MODEL_DOWNLOAD_PROGRESS`, `MODEL_DOWNLOAD_COMPLETED`, `MODEL_DOWNLOAD_FAILED`
- `MODEL_LOADED`, `MODEL_UNLOADED`
- `PROVIDER_SWITCHED`

### TTS Events
- `TTS_STARTED`
- `PHONEME_EVENT`
- `TTS_COMPLETED`

### System Events
- `SYSTEM_STARTED`, `SYSTEM_STOPPED`
- `ERROR_OCCURRED`
- `HEALTH_CHECK`

## Advanced Patterns

### 1. One-Time Subscriptions (Mycroft Pattern)

Subscribe to an event that auto-unsubscribes after first execution:

```python
async def one_time_handler(event: Event):
    print("This will only be called once!")

bus.once(EventType.MODEL_LOADED, one_time_handler)

await bus.emit(EventType.MODEL_LOADED, {"model": "llama"})
await bus.emit(EventType.MODEL_LOADED, {"model": "gemma"})  # Handler not called
```

### 2. Request/Response Pattern

Wait for a specific event with timeout (useful for synchronous-like flows):

```python
# Emit an event that triggers async processing
await bus.emit(EventType.MODEL_DOWNLOAD_STARTED, {"model": "llama"})

# Wait for completion (with timeout)
result = await bus.wait_for(EventType.MODEL_DOWNLOAD_COMPLETED, timeout=30.0)

if result:
    print(f"Download completed: {result.data}")
else:
    print("Timeout waiting for download")
```

### 3. Priority Handlers

Control execution order with priorities (higher = first):

```python
async def high_priority_handler(event: Event):
    print("Executed first")

async def low_priority_handler(event: Event):
    print("Executed second")

bus.subscribe(EventType.SPEECH_DETECTED, high_priority_handler, priority=10)
bus.subscribe(EventType.SPEECH_DETECTED, low_priority_handler, priority=0)
```

### 4. Filtered Subscriptions

Only handle events that match specific criteria:

```python
# Only handle high-confidence speech detection
async def high_confidence_handler(event: Event):
    print(f"High confidence speech: {event.data['text']}")

bus.subscribe(
    EventType.SPEECH_DETECTED,
    high_confidence_handler,
    filter_func=lambda e: e.data.get("confidence", 0) > 0.8
)
```

### 5. Middleware

Add pre/post processing logic that runs for all events:

```python
# Logging middleware
async def logging_middleware(event: Event):
    print(f"[LOG] Event: {event.type.value} from {event.source}")

bus.add_middleware(logging_middleware, pre=True)

# Metrics middleware
async def metrics_middleware(event: Event):
    event.metadata["processed_at"] = time.time()

bus.add_middleware(metrics_middleware, post=False)
```

### 6. Correlation IDs and Metadata

Track related events across services:

```python
# Emit with correlation ID
await bus.emit(
    EventType.SPEECH_DETECTED,
    {"text": "Hello"},
    correlation_id="request-123",
    metadata={"user_id": "alice", "session": "xyz"}
)

# Handler can access correlation info
async def handler(event: Event):
    print(f"Correlation: {event.correlation_id}")
    print(f"User: {event.metadata.get('user_id')}")
```

## Service Integration Examples

### STT Service Integration

```python
class STTService:
    async def initialize(self):
        bus = await EventBus.get_instance()

        # Subscribe to audio events
        bus.subscribe(EventType.AUDIO_STARTED, self.on_audio_started)
        bus.subscribe(EventType.SILENCE_DETECTED, self.on_silence)

    async def on_audio_started(self, event: Event):
        # Start transcription
        pass

    async def on_transcription_complete(self, text: str):
        # Emit speech detected event
        await emit_event(EventType.SPEECH_DETECTED, {
            "text": text,
            "confidence": 0.95,
            "timestamp": time.time()
        })
```

### LLM Service Integration

```python
class LLMService:
    async def initialize(self):
        bus = await EventBus.get_instance()

        # Subscribe to speech events
        bus.subscribe(EventType.SPEECH_DETECTED, self.on_speech_detected)

    async def on_speech_detected(self, event: Event):
        text = event.data.get("text")

        # Emit generating event
        await emit_event(EventType.RESPONSE_GENERATING, {
            "prompt": text
        }, correlation_id=event.correlation_id)

        # Generate response
        response = await self.generate(text)

        # Emit generated event
        await emit_event(EventType.RESPONSE_GENERATED, {
            "response": response,
            "prompt": text
        }, correlation_id=event.correlation_id)
```

### TTS Service Integration

```python
class TTSService:
    async def initialize(self):
        bus = await EventBus.get_instance()

        # Subscribe to LLM responses
        bus.subscribe(EventType.RESPONSE_GENERATED, self.on_response_generated)

    async def on_response_generated(self, event: Event):
        text = event.data.get("response")

        # Emit TTS started
        await emit_event(EventType.TTS_STARTED, {
            "text": text
        }, correlation_id=event.correlation_id)

        # Generate speech
        audio = await self.synthesize(text)

        # Emit completion
        await emit_event(EventType.TTS_COMPLETED, {
            "duration": audio.duration
        }, correlation_id=event.correlation_id)
```

## Monitoring and Debugging

### Get Statistics

```python
stats = bus.get_stats()
print(f"Total events: {stats['metrics']['total_events']}")
print(f"Total errors: {stats['metrics']['total_errors']}")
print(f"Average processing time: {stats['metrics']['avg_processing_time']}")
print(f"Events by type: {stats['metrics']['events_by_type']}")
```

### View Event History

```python
# Get recent events
history = bus.get_event_history(limit=10)
for event in history:
    print(f"{event.type.value}: {event.data}")

# Filter by event type
speech_events = bus.get_event_history(EventType.SPEECH_DETECTED, limit=5)
```

### Check Dead Letter Queue

```python
# View failed events
failed = bus.get_dead_letter_queue(limit=10)
for failure in failed:
    print(f"Failed: {failure['event'].type.value}")
    print(f"Error: {failure['error']}")
    print(f"Handler: {failure['handler']}")
```

### Handler Statistics

```python
stats = bus.get_stats()
for handler_stat in stats['handler_stats']:
    print(f"Handler: {handler_stat['handler']}")
    print(f"  Calls: {handler_stat['call_count']}")
    print(f"  Avg duration: {handler_stat['average_duration']:.6f}s")
```

## Error Handling

The event bus is resilient to errors:

1. **Handler errors don't stop the bus** - Other handlers continue executing
2. **Errors emit ERROR_OCCURRED events** - For centralized error monitoring
3. **Failed events go to dead letter queue** - For later inspection
4. **Comprehensive logging** - All errors are logged with stack traces

```python
# Monitor errors
async def error_monitor(event: Event):
    error = event.data.get("error")
    handler = event.data.get("handler")
    print(f"Error in {handler}: {error}")

bus.subscribe(EventType.ERROR_OCCURRED, error_monitor)
```

## Best Practices

### 1. Use Correlation IDs for Request Tracing

```python
correlation_id = f"request-{uuid.uuid4()}"

await emit_event(EventType.SPEECH_DETECTED, data, correlation_id=correlation_id)
# Pass same correlation_id through the pipeline
```

### 2. Use Filters for Selective Processing

```python
# Don't do this - processes all events then filters
async def handler(event: Event):
    if event.data.get("confidence") > 0.8:
        process(event)

# Do this instead - filter at subscription level
bus.subscribe(
    EventType.SPEECH_DETECTED,
    handler,
    filter_func=lambda e: e.data.get("confidence", 0) > 0.8
)
```

### 3. Use Priorities Wisely

```python
# Critical handlers first
bus.subscribe(EventType.ERROR_OCCURRED, critical_error_handler, priority=100)

# Normal processing
bus.subscribe(EventType.SPEECH_DETECTED, speech_processor, priority=50)

# Logging/metrics last
bus.subscribe(EventType.SPEECH_DETECTED, logger, priority=0)
```

### 4. Clean Up Resources

```python
async def cleanup():
    bus = await EventBus.get_instance()

    # Unsubscribe handlers
    bus.unsubscribe(my_handler)

    # Stop the bus
    await bus.stop()
```

### 5. Use Middleware for Cross-Cutting Concerns

```python
# Logging
async def log_all_events(event: Event):
    logger.info(f"Event: {event.type.value}")

bus.add_middleware(log_all_events, pre=True)

# Timing
async def add_timing(event: Event):
    event.metadata["received_at"] = time.time()

bus.add_middleware(add_timing, pre=True)
```

## Performance Considerations

1. **Event Queue Size**: Default is 1000 events. Adjust based on your load:
   ```python
   bus = EventBus(max_queue_size=5000)
   ```

2. **History Size**: Default is 100 events. Disable if not needed:
   ```python
   bus = EventBus(enable_history=False)
   ```

3. **Async Handlers**: Always prefer async handlers for I/O operations:
   ```python
   # Good
   async def handler(event: Event):
       await database.save(event.data)

   # Avoid (blocks event loop)
   def sync_handler(event: Event):
       database.sync_save(event.data)  # Blocking!
   ```

4. **Middleware Performance**: Keep middleware lightweight as it runs on every event.

## Testing

See `test_event_bus_demo.py` for a comprehensive example demonstrating all features.

Run tests:
```bash
python3 test_event_bus_demo.py
```

## Architecture Decisions

This event bus design is based on research of production voice assistants:

1. **Mycroft AI** - Inspired the `once()` pattern and request/response with `wait_for()`
2. **Rhasspy** - Influenced the middleware approach for event processing pipelines
3. **aiopubsub** - Guided the async-first design and subscriber management
4. **AsyncIO best practices** - Non-blocking emit(), task-based handler execution

## Migration from Old Event Bus

The enhanced event bus is backward compatible. Existing code will continue to work:

```python
# Old code - still works
bus.subscribe(EventType.WAKE_WORD_DETECTED, handler)
await bus.emit(EventType.WAKE_WORD_DETECTED, {"confidence": 0.9})

# New features - opt-in
bus.once(EventType.MODEL_LOADED, handler)  # New!
result = await bus.wait_for(EventType.TTS_COMPLETED)  # New!
```

## License

This event bus implementation is part of the voice assistant project.
