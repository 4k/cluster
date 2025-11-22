# rhubarb_controller.py - Rhubarb Viseme Controller Analysis

## Overview

`rhubarb_controller.py` provides advanced lip-sync animation control with viseme lookahead, smooth easing between visemes, and coarticulation support for natural-looking mouth animations.

## File Location
`/home/user/cluster/src/features/display/rhubarb_controller.py`

## Key Features

- **Viseme lookahead**: Show visemes slightly early (anticipatory animation)
- **Smooth easing**: Various interpolation functions between visemes
- **Coarticulation**: Blend adjacent visemes for natural transitions
- **Extended shapes**: Support for Rhubarb's full A-X shape set
- **Real-time updates**: High-frequency update loop (60Hz default)

## Enums and Data Structures

### RhubarbShape
```python
class RhubarbShape(Enum):
    # Basic shapes (required)
    A = "A"  # Closed (M, B, P)
    B = "B"  # Slightly open (consonants, EE)
    C = "C"  # Open (EH, AE)
    D = "D"  # Wide open (AA)
    E = "E"  # Rounded (AO, ER)
    F = "F"  # Puckered (OO, W)
    # Extended shapes (optional)
    G = "G"  # Upper teeth on lower lip (F, V)
    H = "H"  # Tongue visible (L)
    X = "X"  # Idle/rest
```

### EasingFunction
```python
class EasingFunction(Enum):
    LINEAR = auto()       # Constant speed
    EASE_IN = auto()      # Slow start
    EASE_OUT = auto()     # Slow end
    EASE_IN_OUT = auto()  # Slow start and end
    ANTICIPATE = auto()   # Overshoot then settle
```

### InterpolatedViseme
```python
@dataclass
class InterpolatedViseme:
    primary_shape: RhubarbShape
    primary_weight: float = 1.0
    secondary_shape: Optional[RhubarbShape] = None
    secondary_weight: float = 0.0
    intensity: float = 1.0

    def to_mouth_params() -> Dict[str, float]
```

## Mouth Shape Parameters

Each shape maps to detailed mouth parameters:

```python
RHUBARB_SHAPE_PARAMS = {
    RhubarbShape.A: {
        'open': 0.0,           # Mouth opening
        'width': 0.5,          # Horizontal stretch
        'pucker': 0.1,         # Lip pucker
        'stretch': 0.0,        # Lip stretch
        'teeth_visible': False,
        'tongue_visible': False
    },
    # ... more shapes
}
```

## Coarticulation Rules

Natural transitions between shapes:

```python
COARTICULATION_PAIRS = {
    (RhubarbShape.A, RhubarbShape.B): 0.7,  # High blend
    (RhubarbShape.B, RhubarbShape.C): 0.8,  # High blend
    (RhubarbShape.A, RhubarbShape.D): 0.4,  # Lower blend
    # ...
}
```

## Classes

### RhubarbControllerConfig

```python
@dataclass
class RhubarbControllerConfig:
    # Timing
    lookahead_ms: float = 50.0        # Show visemes early
    hold_minimum_ms: float = 40.0     # Min viseme hold time
    transition_duration_ms: float = 60.0  # Transition time

    # Interpolation
    easing_function: EasingFunction = EASE_IN_OUT

    # Coarticulation
    enable_coarticulation: bool = True
    coarticulation_window_ms: float = 100.0
    coarticulation_strength: float = 0.3

    # Animation quality
    update_rate_hz: float = 60.0
    use_extended_shapes: bool = True
    intensity_scale: float = 1.0
```

### RhubarbVisemeController

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `set_viseme_callback(callback)` | Set viseme change callback |
| `load_lip_sync_data(cues, session_id, duration)` | Load Rhubarb data |
| `start_session()` | Begin playback |
| `stop_session()` | End playback |
| `update(dt)` | Update and get current state |
| `get_current_viseme()` | Get current viseme name |
| `get_current_mouth_params()` | Get current parameters |
| `get_progress()` | Get playback progress (0-1) |
| `is_active()` | Check if session active |
| `get_stats()` | Get statistics |

**Update Flow**:
```
update(dt)
    ↓
Get elapsed time since session start
    ↓
Apply lookahead offset
    ↓
Find current cue at adjusted time
    ↓
If new cue:
    ├── Create target viseme
    ├── Apply coarticulation if enabled
    ├── Store current as transition start
    └── Reset transition progress
    ↓
Update transition:
    ├── Calculate progress based on dt
    ├── Apply easing function
    ├── Interpolate current → target
    └── Return (viseme_name, mouth_params)
```

### RhubarbLipSyncIntegration

Event bus integration layer:

| Event | Action |
|-------|--------|
| `LIP_SYNC_READY` | Cache lip sync data |
| `LIP_SYNC_STARTED` | Load data, start session |
| `LIP_SYNC_COMPLETED` | Stop session |
| `AUDIO_PLAYBACK_ENDED` | Ensure session stopped |

## Improvements Suggested

### 1. Dynamic Lookahead
Adjust lookahead based on speech rate:
```python
def _calculate_dynamic_lookahead(self) -> float:
    """Adjust lookahead based on speech rate."""
    if len(self._cues) < 2:
        return self.config.lookahead_ms

    # Calculate average cue duration
    durations = [c.duration for c in self._cues]
    avg_duration = sum(durations) / len(durations)

    # Faster speech = less lookahead
    if avg_duration < 0.05:  # Fast speech
        return self.config.lookahead_ms * 0.5
    return self.config.lookahead_ms
```

### 2. Intensity Envelope
Vary intensity over utterance:
```python
def _calculate_intensity_envelope(self, progress: float) -> float:
    """Apply intensity envelope over utterance."""
    # Fade in at start, fade out at end
    if progress < 0.1:
        return progress * 10  # Fade in
    elif progress > 0.9:
        return (1 - progress) * 10  # Fade out
    return 1.0
```

### 3. Micro-Expression Injection
Add subtle variations:
```python
def _add_micro_expressions(self, params: Dict[str, float]) -> Dict[str, float]:
    """Add subtle random variations for realism."""
    import random
    jitter = self.config.micro_expression_strength
    for key in ['open', 'width']:
        params[key] += random.uniform(-jitter, jitter)
    return params
```

### 4. Phoneme-Level Control
Support phoneme-level timing:
```python
def load_phoneme_data(self, phonemes: List[Dict]):
    """Load phoneme-level timing for higher precision."""
    # Convert phonemes to visemes with sub-phoneme timing
    cues = []
    for phoneme in phonemes:
        viseme = PHONEME_TO_VISEME[phoneme['value']]
        cues.append(RhubarbVisemeCue(
            start_time=phoneme['start'],
            end_time=phoneme['end'],
            shape=viseme
        ))
    self._cues = cues
```

### 5. Emotion Modifiers
Adjust animation based on emotion:
```python
EMOTION_MODIFIERS = {
    "HAPPY": {'width': 0.1, 'open': 0.05},
    "SAD": {'width': -0.1, 'open': -0.05},
    "ANGRY": {'width': 0.15, 'open': 0.1}
}

def apply_emotion_modifier(self, params: Dict, emotion: str) -> Dict:
    modifier = EMOTION_MODIFIERS.get(emotion, {})
    for key, value in modifier.items():
        params[key] = params.get(key, 0) + value
    return params
```

### 6. Recording/Playback
Record and replay animation sessions:
```python
def start_recording(self):
    self._recorded_frames = []

def update(self, dt):
    result = self._do_update(dt)
    if self._recording and result:
        self._recorded_frames.append((time.time(), result))
    return result

def save_recording(self, path: str):
    with open(path, 'w') as f:
        json.dump(self._recorded_frames, f)
```
