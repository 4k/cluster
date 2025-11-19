# Event Bus Integration for Voice Assistant

This document describes the event-driven architecture implementation for the voice assistant, following industry best practices from Mycroft AI, Rhasspy, Home Assistant, and modern voice AI research (2024/2025).

## Architecture Overview

The voice assistant uses a **turn-based (cascading) architecture** with event bus communication:

```
User speaks → STT detects wake word → STT transcribes speech →
LLM generates response → TTS synthesizes speech → Audio plays
```

All services communicate via the event bus using:
- **Correlation IDs** for request tracing
- **Type-safe events** via EventType enum
- **Async-first design** for non-blocking operations
- **Priority handlers** for execution order control
- **Error resilience** with dead letter queue

## Services

### 1. STT Service (Speech-to-Text)

**Subscribes to:**
- `SYSTEM_STOPPED` - Gracefully shutdown

**Emits:**
- `WAKE_WORD_DETECTED` - When wake word is detected
- `AUDIO_STARTED` - When audio recording begins
- `SPEECH_DETECTED` - When speech is transcribed (with text, confidence)
- `SPEECH_ENDED` - When user stops speaking
- `SILENCE_DETECTED` - When silence is detected
- `AUDIO_STOPPED` - When audio recording ends
- `ERROR_OCCURRED` - On transcription errors

**Features:**
- Wake word detection using OpenWakeWord
- Speech recognition using Vosk
- Correlation ID tracking for each request
- Voice activity detection (VAD)

### 2. LLM Service

**Subscribes to:**
- `SPEECH_DETECTED` - Process user speech
- `AMBIENT_SPEECH_DETECTED` - Process ambient speech
- `SYSTEM_STOPPED` - Gracefully shutdown

**Emits:**
- `RESPONSE_GENERATING` - When starting to generate response
- `RESPONSE_GENERATED` - When response is ready (with text, elapsed time, word count)
- `ERROR_OCCURRED` - On generation errors

**Features:**
- Async request handling (non-blocking)
- System prompt for voice-optimized responses
- Elapsed time tracking
- Error recovery

### 3. TTS Service (Text-to-Speech)

**Subscribes to:**
- `RESPONSE_GENERATED` - Synthesize LLM response
- `SYSTEM_STOPPED` - Gracefully shutdown

**Emits:**
- `TTS_STARTED` - When synthesis begins
- `AUDIO_PLAYBACK_STARTED` - When audio playback starts
- `PHONEME_EVENT` - For lip-sync/animation (future)
- `AUDIO_PLAYBACK_ENDED` - When audio playback ends
- `TTS_COMPLETED` - When synthesis is complete
- `ERROR_OCCURRED` - On synthesis errors

**Features:**
- **Message queuing** - Prevents interruption (critical!)
- Async audio playback
- Piper TTS integration
- Automatic queue processing

## Event Flow Example

Here's a complete request flow with correlation tracking:

```
1. User says "Hey Jarvis"
   → WAKE_WORD_DETECTED (correlation_id: stt-abc123, confidence: 0.95)

2. STT starts listening
   → AUDIO_STARTED (correlation_id: stt-abc123)

3. User says "What's the weather?"
   → SPEECH_DETECTED (correlation_id: stt-abc123, text: "What's the weather?")
   → SPEECH_ENDED (correlation_id: stt-abc123)
   → AUDIO_STOPPED (correlation_id: stt-abc123)

4. LLM processes request
   → RESPONSE_GENERATING (correlation_id: stt-abc123, prompt: "What's the weather?")
   → RESPONSE_GENERATED (correlation_id: stt-abc123, response: "The weather today is...", elapsed_time: 1.2s)

5. TTS queues and synthesizes
   → TTS_STARTED (correlation_id: stt-abc123, text: "The weather today is...")
   → AUDIO_PLAYBACK_STARTED (correlation_id: stt-abc123)
   → AUDIO_PLAYBACK_ENDED (correlation_id: stt-abc123)
   → TTS_COMPLETED (correlation_id: stt-abc123)
```

## Critical Best Practices Implemented

### 1. TTS Queuing (Home Assistant Pattern)

**Problem:** Sending multiple TTS messages interrupts the first one.

**Solution:** Message queue in TTS service processes requests sequentially.

```python
# Messages are queued, not interrupted
1. "Hello there"        ← Playing
2. "How are you?"       ← Queued
3. "Have a nice day!"   ← Queued
```

### 2. Correlation ID Tracking (Mycroft Pattern)

Every request gets a correlation ID that follows it through all services:

```python
correlation_id = f"stt-{uuid.uuid4().hex[:12]}"
```

This enables:
- End-to-end request tracing
- Debugging specific requests
- Performance analysis per request

### 3. Async-First Design (aiopubsub Pattern)

All services use async/await for non-blocking operations:
- `emit()` is synchronous (returns immediately)
- Handlers run as asyncio tasks (concurrent)
- Blocking operations run in executors

### 4. Error Resilience (Enterprise Pattern)

- Handler errors don't stop the event bus
- Failed events go to dead letter queue
- ERROR_OCCURRED events for centralized monitoring
- Comprehensive logging with stack traces

## Usage

### Run Full Voice Assistant

```bash
# Default settings (Jarvis wake word)
python voice_assistant.py

# Custom wake word
python voice_assistant.py --wake-word alexa

# Custom LLM model
python voice_assistant.py --llm-model llama3:8b

# List audio devices
python voice_assistant.py --list-devices

# Verbose mode (show detection scores)
python voice_assistant.py --verbose
```

### Run Individual Services (Event-Driven Mode)

```bash
# STT service (in one terminal)
python stt_service.py

# LLM service (in another terminal)
python llm_service.py

# TTS service (in another terminal)
python tts_service.py --event-driven
```

All services will communicate via the shared event bus.

### Direct Mode (Testing)

```bash
# Test TTS directly
python tts_service.py "Hello, this is a test"

# Test LLM directly (legacy mode)
# (See llm_service.py for standalone examples)
```

## Monitoring & Debugging

### View Event History

```python
from src.core.event_bus import EventBus

bus = await EventBus.get_instance()
history = bus.get_event_history(limit=20)
for event in history:
    print(f"{event.type.value}: {event.data}")
```

### Check Statistics

The orchestrator prints statistics every 60 seconds:

```
📊 VOICE ASSISTANT STATISTICS
======================================================================
Requests Processed:    15
Responses Generated:   15
Speech Synthesized:    15
Errors:                0

Event Bus:
  Total Events:        120
  Total Errors:        0
  Avg Processing Time: 0.000234s
======================================================================
```

### View Dead Letter Queue

```python
failed = bus.get_dead_letter_queue(limit=10)
for failure in failed:
    print(f"Failed: {failure['event'].type.value}")
    print(f"Error: {failure['error']}")
```

## Performance Considerations

### Industry Benchmarks (2024/2025)

Best-in-class latency targets:
- STT (Deepgram): ~100ms
- LLM (GPT-4): ~320ms
- TTS (Cartesia): ~90ms
- **Total**: ~510ms (human conversation: ~230ms)

Our implementation:
- STT (Vosk): ~200-500ms (local)
- LLM (Ollama): ~1-3s (depends on model)
- TTS (Piper): ~200-500ms (local)
- **Total**: ~1.5-4s (reasonable for local setup)

### Optimizations

1. **Event Bus**: < 1ms average processing time
2. **Concurrent Handlers**: All handlers run in parallel via `asyncio.gather()`
3. **Non-Blocking**: All I/O operations use async/await
4. **Queue Size**: Default 1000 events (configurable)

## Research & Inspiration

This implementation is based on research of production voice assistants:

1. **Mycroft AI** - `once()` pattern, request/response with `wait_for()`
2. **Rhasspy** - Middleware for event processing pipelines
3. **Home Assistant** - Voice pipeline integration, TTS queuing
4. **aiopubsub** - Async-first design, subscriber management
5. **Cosmic Python** - Event-driven architecture patterns
6. **2024/2025 Voice AI Research** - Turn-based architecture, latency optimization

## Architecture Decisions

### Why Turn-Based (Cascading)?

**Chosen:** Sequential STT → LLM → TTS with event bus

**Alternatives Considered:**
- Real-time speech-to-speech (simultaneous streaming)
- WebSocket-based message passing

**Rationale:**
1. Simpler to implement and debug
2. Industry standard for most voice assistants
3. Easy to swap individual components
4. Clear separation of concerns
5. Suitable for local processing

### Why Event Bus vs Direct Calls?

**Benefits:**
1. Loose coupling between services
2. Easy to add new services (e.g., animation, logging)
3. Request tracing with correlation IDs
4. Error resilience (one service failure doesn't crash others)
5. Replay and debugging capabilities
6. Horizontal scaling potential (future: distributed mode)

## Future Enhancements

Potential improvements (not yet implemented):

1. **Distributed Mode** - Redis/MQTT backend for multi-process
2. **Event Persistence** - Database integration for durability
3. **Barge-In Support** - Interrupt TTS when user speaks
4. **Streaming STT** - Real-time transcription
5. **Streaming LLM** - Token-by-token generation
6. **Voice Activity Detection (VAD)** - Better silence detection
7. **Emotion Detection** - Analyze user sentiment
8. **Multi-Speaker Support** - Voice identification

## Troubleshooting

### Event Bus Not Starting

```python
# Check if event bus is running
bus = await EventBus.get_instance()
stats = bus.get_stats()
print(stats['is_running'])
```

### Services Not Communicating

1. Check all services are initialized: `await service.initialize()`
2. Verify event types match exactly (use EventType enum)
3. Check correlation IDs are being passed
4. Review event history: `bus.get_event_history()`

### TTS Messages Getting Interrupted

- Ensure TTS service is running in event-driven mode
- Check queue is processing: `tts.is_processing`
- Verify AUDIO_PLAYBACK_ENDED events are emitted

### High Latency

1. Check LLM server response time
2. Verify event bus average processing time (< 1ms)
3. Consider using smaller/faster models
4. Check network connectivity (if using remote LLM)

## Contributing

When adding new services or modifying event flows:

1. Use correlation IDs for all related events
2. Emit events at key lifecycle points
3. Handle errors gracefully with ERROR_OCCURRED events
4. Add appropriate logging
5. Update this documentation
6. Test with the orchestrator script

## References

- [Mycroft Message Bus](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/mycroft-core/message-bus)
- [Building Event Bus with AsyncIO](https://www.joeltok.com/posts/2021-03-building-an-event-bus-in-python/)
- [Home Assistant Voice Pipelines](https://developers.home-assistant.io/docs/voice/pipelines/)
- [Cosmic Python - Event-Driven Architecture](https://www.cosmicpython.com/book/chapter_08_events_and_message_bus.html)
- [Voice AI Stack 2025](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)
