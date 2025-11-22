# types.py - Core Type Definitions Analysis

## Overview

`types.py` defines shared type definitions used across all modules of the voice assistant. It provides enums for states and dataclasses for structured data.

## File Location
`/home/user/cluster/src/core/types.py`

## Enums

### EmotionType
Defines emotional states for the assistant display:
- `NEUTRAL`, `HAPPY`, `SAD`, `ANGRY`, `SURPRISED`, `FEARFUL`, `DISGUSTED`, `CONFUSED`
- Activity states: `THINKING`, `LISTENING`, `SPEAKING`

### GazeDirection
Eye gaze positions:
- Cardinal: `CENTER`, `UP`, `DOWN`, `LEFT`, `RIGHT`
- Diagonal: `UP_LEFT`, `UP_RIGHT`, `DOWN_LEFT`, `DOWN_RIGHT`

### MouthShape
Mouth shapes for lip-sync animation:
- Basic: `CLOSED`, `OPEN`, `SMILE`, `FROWN`, `NEUTRAL`
- Visemes: `SILENCE`, `BMP`, `LNT`, `AH`, `EE`, `OH`, `OO`, `FV`

### AnimationState
Animation system states:
- `IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`, `ERROR`, `WAITING`

## Dataclasses

### ConversationTurn
```python
@dataclass
class ConversationTurn:
    role: str       # 'user', 'assistant', 'system'
    content: str    # Message content
    timestamp: float
    metadata: Dict[str, Any]
```

### ConversationContext
```python
@dataclass
class ConversationContext:
    turns: List[ConversationTurn]
    current_emotion: EmotionType
    last_activity: float
    metadata: Dict[str, Any]

    # Methods:
    def add_turn(role, content)
    def get_recent_turns(count=10)
    def clear()
```

### AudioConfig
```python
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 1024
    format: str = "int16"
```

### AudioFrame
```python
@dataclass
class AudioFrame:
    data: bytes
    sample_rate: int
    channels: int
    timestamp: float
```

### TTSConfig
```python
@dataclass
class TTSConfig:
    engine_type: str = "piper"
    model_path: Optional[str] = None
    emotion_support: bool = False
    phoneme_output: bool = False
    sample_rate: int = 22050
```

### CameraConfig / DisplayConfig
Configuration dataclasses for camera and display subsystems.

## Type Relationships

```
EmotionType ─────────┐
                     ├──→ ConversationContext
GazeDirection ───────┤
                     │
MouthShape ──────────┼──→ Display System
                     │
AnimationState ──────┘

AudioConfig ─────────┬──→ AudioFrame ──→ STT/TTS Services
                     │
TTSConfig ───────────┘

ConversationTurn ────┬──→ ConversationContext ──→ LLM Service
                     │
                     └──→ Memory System
```

## Improvements Suggested

### 1. Protocol Classes
Add Protocol definitions for service interfaces:
```python
from typing import Protocol

class AudioProcessor(Protocol):
    async def process(self, frame: AudioFrame) -> Any: ...
```

### 2. Typed Event Data
Create typed dataclasses for event payloads:
```python
@dataclass
class SpeechDetectedEvent:
    text: str
    confidence: float
    timestamp: float
    language: str = "en"
```

### 3. Value Objects
Use frozen dataclasses for immutable data:
```python
@dataclass(frozen=True)
class VisemeFrame:
    shape: MouthShape
    intensity: float
    timestamp: float
```

### 4. Type Aliases
Define type aliases for complex types:
```python
EventPayload = Dict[str, Any]
HandlerFunc = Callable[[Event], Awaitable[None]]
MouthParams = Dict[str, float]
```
