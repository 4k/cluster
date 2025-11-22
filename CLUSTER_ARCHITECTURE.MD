# Cluster Voice Assistant - Architecture Documentation

## Executive Summary

Cluster is a modular, event-driven voice assistant framework designed for real-time conversational AI with sophisticated facial animation capabilities. The system integrates speech recognition, language models, text-to-speech synthesis, and animated display features through a central event bus architecture.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Core Components](#core-components)
4. [Service Layer](#service-layer)
5. [Feature System](#feature-system)
6. [Event Flow](#event-flow)
7. [Data Flow](#data-flow)
8. [Configuration System](#configuration-system)
9. [Improvement Recommendations](#improvement-recommendations)
10. [Development Guidelines](#development-guidelines)

---

## System Overview

### Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Async Runtime | asyncio |
| Speech Recognition | OpenWakeWord, Vosk |
| LLM Backend | Ollama (local), OpenAI-compatible APIs |
| Text-to-Speech | Piper-TTS |
| Lip Sync | Rhubarb Lip Sync |
| Display | Pygame (multi-window) |
| Audio | PyAudio |

### Key Design Principles

1. **Event-Driven Architecture**: All components communicate through a central event bus
2. **Modular Design**: Services and features are loosely coupled and independently deployable
3. **Async-First**: Built on asyncio for non-blocking operations
4. **Graceful Degradation**: Mock implementations allow partial operation
5. **Configuration-Driven**: YAML-based configuration with environment overrides

---

## Architecture Diagrams

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLUSTER VOICE ASSISTANT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         main.py (Entry Point)                        │    │
│  │                      VoiceAssistant Orchestrator                     │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          EVENT BUS (Core)                            │    │
│  │     Pub/Sub | Middleware | History | Metrics | Request/Response      │    │
│  └──────┬────────────────┬────────────────┬────────────────┬───────────┘    │
│         │                │                │                │                 │
│         ▼                ▼                ▼                ▼                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌─────────────────┐    │
│  │    STT    │    │    LLM    │    │    TTS    │    │    FEATURES     │    │
│  │  Service  │    │  Service  │    │  Service  │    │    (Display)    │    │
│  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘    └────────┬────────┘    │
│        │                │                │                   │              │
│        ▼                ▼                ▼                   ▼              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌─────────────────┐    │
│  │OpenWakeWrd│    │  Ollama   │    │ Piper-TTS │    │   DisplayMgr    │    │
│  │   Vosk    │    │  (local)  │    │           │    │   Rhubarb       │    │
│  └───────────┘    └───────────┘    └───────────┘    └─────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CONVERSATION FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────┘

User speaks          Wake word           Speech              LLM processes
     │               detected            transcribed         generates response
     ▼                   ▼                   ▼                     ▼
┌─────────┐      ┌─────────────┐      ┌───────────────┐     ┌────────────────┐
│  Audio  │──────│ WAKE_WORD_  │──────│SPEECH_DETECTED│─────│RESPONSE_      │
│ Stream  │      │  DETECTED   │      │               │     │ GENERATING    │
└─────────┘      └─────────────┘      └───────────────┘     └───────┬────────┘
                       │                     │                       │
                       ▼                     ▼                       ▼
              ┌─────────────────┐   ┌───────────────────┐  ┌────────────────┐
              │ Display:        │   │ LLMService:       │  │ Display:       │
              │ LISTENING       │   │ generate_async()  │  │ THINKING       │
              │ SURPRISED       │   │                   │  │ animation      │
              └─────────────────┘   └───────────────────┘  └────────────────┘
                                                                    │
                                                                    ▼
     TTS speaks           Lip sync           Audio plays     ┌────────────────┐
     response             generated          with sync       │RESPONSE_       │
          │                   │                  │           │ GENERATED      │
          ▼                   ▼                  ▼           └───────┬────────┘
   ┌────────────┐      ┌───────────┐      ┌────────────┐            │
   │TTS_STARTED │──────│LIP_SYNC_  │──────│AUDIO_      │◄───────────┘
   │            │      │  READY    │      │PLAYBACK_   │
   └────────────┘      └───────────┘      │ STARTED    │
          │                  │            └─────┬──────┘
          ▼                  ▼                  │
   ┌─────────────────────────────────────┐     │
   │ Display: MOUTH_SHAPE_UPDATE events  │◄────┘
   │ (30Hz viseme updates from Rhubarb)  │
   └─────────────────────────────────────┘
                      │
                      ▼
              ┌────────────┐
              │TTS_COMPLETE│───▶ Return to IDLE
              └────────────┘
```

### Component Interaction Diagram

```
                           ┌─────────────────────────────────┐
                           │          ConfigManager          │
                           │   (.env, YAML, env overrides)   │
                           └───────────────┬─────────────────┘
                                           │ configures
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
           │   STTService   │    │   LLMService   │    │   TTSService   │
           │                │    │                │    │                │
           │ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
           │ │OpenWakeWord│ │    │ │  Ollama    │ │    │ │ Piper-TTS  │ │
           │ │   Model    │ │    │ │  Client    │ │    │ │  Engine    │ │
           │ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
           │ ┌────────────┐ │    │                │    │ ┌────────────┐ │
           │ │   Vosk     │ │    │                │    │ │ AudioQueue │ │
           │ │   Model    │ │    │                │    │ │            │ │
           │ └────────────┘ │    │                │    │ └────────────┘ │
           └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
                   │                     │                     │
                   │    ┌────────────────┴────────────────┐   │
                   │    │                                 │   │
                   ▼    ▼                                 ▼   ▼
         ┌──────────────────────────────────────────────────────────┐
         │                         EventBus                          │
         │  ┌──────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
         │  │Queue │  │Handlers │  │Middleware│ │ Metrics/History │  │
         │  └──────┘  └─────────┘  └─────────┘  └─────────────────┘  │
         └───────────────────────────────┬──────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
           │  FeatureLoader │   │ AnimationSvc   │   │   AmbientSTT   │
           └───────┬────────┘   └───────┬────────┘   └────────────────┘
                   │                    │
                   ▼                    ▼
           ┌────────────────┐   ┌────────────────┐
           │ DisplayManager │   │RhubarbLipSync  │
           │                │   │    Service     │
           │ ┌────────────┐ │   └───────┬────────┘
           │ │WindowMgr   │ │           │
           │ └────────────┘ │           ▼
           │ ┌────────────┐ │   ┌────────────────┐
           │ │DecisionMod │ │   │ VisemeControl  │
           │ └────────────┘ │   │                │
           │ ┌────────────┐ │   └────────────────┘
           │ │Renderers   │ │
           │ │(Eye, Mouth)│ │
           │ └────────────┘ │
           └────────────────┘
```

### Lip Sync Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LIP SYNC PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

 TTS Service                Animation Service              Display Manager
      │                           │                              │
      │  TTS_STARTED              │                              │
      │  (audio_file, text)       │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │                    ┌──────┴──────┐                       │
      │                    │   Rhubarb   │                       │
      │                    │ Processing  │                       │
      │                    │ (blocking)  │                       │
      │                    └──────┬──────┘                       │
      │                           │                              │
      │                    Cache LipSyncData                     │
      │                           │                              │
      │  LIP_SYNC_READY           │                              │
      │◀──────────────────────────│                              │
      │                           │                              │
      │  Start audio playback     │                              │
      │  AUDIO_PLAYBACK_STARTED   │                              │
      │  (start_timestamp)        │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │                    Start LipSyncSession                  │
      │                    at start_timestamp                    │
      │                           │                              │
      │                           │     LIP_SYNC_STARTED         │
      │                           │─────────────────────────────▶│
      │                           │                              │
      │                    ┌──────┴──────┐                       │
      │                    │   30Hz      │    MOUTH_SHAPE_UPDATE │
      │                    │   Viseme    │─────────────────────▶ │
      │                    │   Loop      │    (every 33ms)       │
      │                    └──────┬──────┘                       │
      │                           │                              │
      │  AUDIO_PLAYBACK_ENDED     │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │                    Stop session    LIP_SYNC_COMPLETED    │
      │                           │─────────────────────────────▶│
      │                           │                              │
      │  TTS_COMPLETED            │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
```

---

## Core Components

### Event Bus (`src/core/event_bus.py`)

The central nervous system of the application.

**Capabilities**:
- Async event queue with configurable size
- Priority-based handler execution
- One-time subscriptions (Mycroft pattern)
- Request/response pattern (`wait_for`)
- Wildcard event matching
- Event history and replay
- Dead letter queue
- Pre/post middleware
- Comprehensive metrics

**Key Event Categories**:
| Category | Events |
|----------|--------|
| Audio | `AUDIO_STARTED`, `WAKE_WORD_DETECTED`, `SPEECH_DETECTED` |
| AI | `RESPONSE_GENERATING`, `RESPONSE_GENERATED` |
| TTS | `TTS_STARTED`, `TTS_COMPLETED` |
| Lip Sync | `LIP_SYNC_READY`, `LIP_SYNC_STARTED`, `MOUTH_SHAPE_UPDATE` |
| Animation | `EXPRESSION_CHANGE`, `GAZE_UPDATE`, `BLINK_TRIGGERED` |
| System | `SYSTEM_STARTED`, `SYSTEM_STOPPED`, `ERROR_OCCURRED` |

### Configuration (`src/core/config.py`)

Hierarchical configuration management.

**Sources** (in priority order):
1. Default values (dataclass)
2. `.env` file
3. Environment variables

**Key Config Sections**:
- `llm`: LLM provider settings
- `tts`: TTS engine settings
- `audio`: Audio device settings
- `display`: Display settings
- `mock_*`: Mock mode flags

### Types (`src/core/types.py`)

Shared type definitions:
- `EmotionType`: NEUTRAL, HAPPY, SAD, ANGRY, etc.
- `AnimationState`: IDLE, LISTENING, THINKING, SPEAKING
- `MouthShape`: Viseme definitions
- `ConversationTurn`, `ConversationContext`: Conversation data
- `AudioFrame`, `AudioConfig`: Audio data structures

---

## Service Layer

### STT Service (`src/services/stt_service.py`)

**Pipeline**:
```
PyAudio Stream → OpenWakeWord → Vosk → Event Bus
```

**Features**:
- Wake word detection (configurable)
- Continuous speech recognition
- RMS-based silence detection
- Multi-sample-rate support with resampling

### LLM Service (`src/services/llm_service.py`)

**Supported APIs**:
- Ollama Chat (`/api/chat`)
- Ollama Generate (`/api/generate`)
- OpenAI-compatible (`/v1/chat/completions`)

**Features**:
- Auto-detection of API type
- Connection diagnostics
- Request/response correlation

### TTS Service (`src/services/tts_service.py`)

**Pipeline**:
```
Text → Piper-TTS → WAV file → PyAudio playback
```

**Features**:
- Queue-based processing (prevents interruption)
- Audio file generation for lip sync
- Precise playback timing for animation sync
- Audio file cleanup management

---

## Feature System

### Feature Architecture (`src/features/__init__.py`)

Plugin-based feature system:

```python
@register_feature("display")
class DisplayFeature(Feature):
    async def initialize(self) -> bool: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

### Display Feature

**Components**:
- `DisplayManager`: Main interface
- `WindowManager`: Multi-window orchestration
- `DisplayDecisionModule`: Content routing
- `AnimationService`: Lip sync coordination
- `RhubarbLipSyncService`: Viseme generation
- `RhubarbVisemeController`: Smooth animation

**Animation Features**:
- Multi-window support (eyes, mouth)
- Rhubarb lip sync integration
- Viseme lookahead (anticipatory animation)
- Coarticulation (viseme blending)
- Smooth easing functions
- 60Hz update rate

---

## Configuration System

### File Structure

```
cluster/
├── .env                          # Environment variables
├── config/
│   ├── default.yaml             # Default settings
│   ├── stt.yaml                 # STT service config
│   ├── llm.yaml                 # LLM service config
│   ├── tts.yaml                 # TTS service config
│   └── display.yaml             # Display config
```

### Environment Variables

```bash
# Core
CLUSTER_DEBUG=true
CLUSTER_LOG_LEVEL=DEBUG

# LLM
LLM_MODEL_ID=llama3.2:3b
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7

# TTS
TTS_MODEL_PATH=/path/to/voice.onnx

# Mock modes
MOCK_LLM=false
MOCK_TTS=false
MOCK_STT=false
```

---

## Improvement Recommendations

### Priority 1: Critical Improvements

#### 1.1 Response Streaming
**Current**: Full response generation before speech
**Proposed**: Stream tokens and begin TTS incrementally

```python
async def stream_and_speak(self, prompt: str):
    buffer = ""
    async for token in self.llm.stream_generate(prompt):
        buffer += token
        if self._is_speakable_chunk(buffer):
            await self.tts.speak_async(buffer)
            buffer = ""
```

**Benefits**: 50-70% reduction in perceived latency

#### 1.2 Service Health Monitoring
**Current**: No runtime health checks
**Proposed**: Add health monitoring with auto-recovery

```python
class ServiceHealthMonitor:
    async def check_all_services(self):
        for service in self.services:
            if not await service.health_check():
                await self.attempt_recovery(service)
```

#### 1.3 Graceful Degradation
**Current**: Failure in one service stops all
**Proposed**: Continue with reduced functionality

### Priority 2: Performance Improvements

#### 2.1 Parallel Rhubarb Processing
Pre-process lip sync while TTS generates audio

#### 2.2 Event Batching
Batch high-frequency events (audio frames) for efficiency

#### 2.3 Caching Layer
Cache LLM responses, TTS audio, and lip sync data

### Priority 3: Feature Enhancements

#### 3.1 Multi-Language Support
- Language detection in STT
- Multi-language TTS models
- Language-specific LLM prompts

#### 3.2 Speaker Diarization
Identify and track multiple speakers

#### 3.3 Emotion Detection
Detect user emotion from voice and adjust responses

#### 3.4 Conversation Memory
Persistent conversation history and user preferences

### Priority 4: Architecture Improvements

#### 4.1 Microservices Option
Split into independent services for scalability:
- STT microservice
- LLM microservice
- TTS microservice
- Display microservice

#### 4.2 WebSocket API
Add WebSocket interface for remote clients

#### 4.3 Plugin System Enhancement
- Hot-loading of plugins
- Plugin dependencies
- Plugin marketplace

---

## Development Guidelines

### Code Style

```python
# Use type hints
def process_audio(self, frame: AudioFrame) -> Optional[str]:
    ...

# Use dataclasses for structured data
@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    timestamp: float

# Use enums for fixed sets
class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
```

### Event Naming Convention

```
NOUN_VERB_OPTIONAL_MODIFIER

Examples:
- SPEECH_DETECTED
- RESPONSE_GENERATING
- AUDIO_PLAYBACK_STARTED
- LIP_SYNC_COMPLETED
```

### Logging Guidelines

```python
# Debug: Detailed information for debugging
logger.debug(f"Processing frame {frame_id} with size {len(data)}")

# Info: General operational messages
logger.info(f"Service started successfully")

# Warning: Something unexpected but handled
logger.warning(f"Retry {attempt}/3 for API call")

# Error: Failure that needs attention
logger.error(f"Failed to process: {error}", exc_info=True)
```

### Testing Strategy

1. **Unit Tests**: Individual functions and methods
2. **Integration Tests**: Service interactions via event bus
3. **Mock Mode Testing**: Use mock implementations for isolated testing
4. **End-to-End Tests**: Full conversation flow testing

---

## File Reference

| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `src/core/event_bus.py` | Central event system |
| `src/core/config.py` | Configuration management |
| `src/core/types.py` | Shared type definitions |
| `src/core/service_config.py` | Service configuration loader |
| `src/services/stt_service.py` | Speech-to-text service |
| `src/services/llm_service.py` | Language model service |
| `src/services/tts_service.py` | Text-to-speech service |
| `src/audio/audio_manager.py` | Audio device management |
| `src/audio/ambient_stt.py` | Continuous listening |
| `src/features/__init__.py` | Feature plugin system |
| `src/features/display/display_manager.py` | Display coordination |
| `src/features/display/animation.py` | Animation service |
| `src/features/display/lip_sync.py` | Rhubarb integration |
| `src/features/display/rhubarb_controller.py` | Viseme control |

---

## Appendix: Event Reference

### Complete Event Type List

```python
class EventType(Enum):
    # Audio Events
    AUDIO_STARTED = "audio.started"
    AUDIO_STOPPED = "audio.stopped"
    WAKE_WORD_DETECTED = "audio.wake_word_detected"
    SPEECH_DETECTED = "audio.speech_detected"
    SPEECH_ENDED = "audio.speech_ended"
    AMBIENT_SPEECH_DETECTED = "audio.ambient_speech_detected"
    WAKEWORD_SPEECH_DETECTED = "audio.wakeword_speech_detected"

    # AI Events
    CONVERSATION_UPDATED = "ai.conversation_updated"
    DECISION_MADE = "ai.decision_made"
    RESPONSE_GENERATING = "ai.response_generating"
    RESPONSE_GENERATED = "ai.response_generated"

    # TTS Events
    TTS_STARTED = "tts.started"
    TTS_COMPLETED = "tts.completed"
    PHONEME_EVENT = "tts.phoneme"
    AUDIO_PLAYBACK_STARTED = "tts.audio_playback_started"
    AUDIO_PLAYBACK_ENDED = "tts.audio_playback_ended"

    # Lip Sync Events
    LIP_SYNC_READY = "lipsync.ready"
    LIP_SYNC_STARTED = "lipsync.started"
    LIP_SYNC_VISEME = "lipsync.viseme"
    LIP_SYNC_COMPLETED = "lipsync.completed"

    # Animation Events
    EXPRESSION_CHANGE = "animation.expression_change"
    EMOTION_CHANGED = "animation.emotion_changed"
    GAZE_UPDATE = "animation.gaze_update"
    MOUTH_SHAPE_UPDATE = "animation.mouth_shape_update"
    BLINK_TRIGGERED = "animation.blink_triggered"

    # System Events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    ERROR_OCCURRED = "system.error"
    HEALTH_CHECK = "system.health_check"
```

---

*Document generated: 2025-11-22*
*Analysis performed on Cluster Voice Assistant codebase*
