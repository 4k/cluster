# display_manager.py - Display Manager Analysis

## Overview

`display_manager.py` is the main interface for the multi-window emotion display system. It provides high-level API for managing display windows and integrates with the event bus for coordinated facial animations with Rhubarb lip sync.

## File Location
`/home/user/cluster/src/features/display/display_manager.py`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DisplayManager                                │
│  (Main interface - this class)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      WindowManager                            │   │
│  │              (Orchestrates multiple windows)                  │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          ↓                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  DisplayDecisionModule                        │   │
│  │                (Central content routing)                      │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          ↓                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │  WindowProcess  │  │  WindowProcess  │  │  WindowProcess    │   │
│  │   (Left Eye)    │  │  (Right Eye)    │  │    (Mouth)        │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬──────────┘   │
│           ↓                    ↓                    ↓              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │    Renderer     │  │    Renderer     │  │    Renderer       │   │
│  │   (Eye anim)    │  │   (Eye anim)    │  │   (Mouth anim)    │   │
│  └─────────────────┘  └─────────────────┘  └───────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               RhubarbVisemeController                         │   │
│  │          (Precise lip-sync timing and interpolation)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Class: DisplayManager

### Initialization

```python
DisplayManager(
    settings: DisplaySettings = None,
    rhubarb_config: RhubarbControllerConfig = None
)
```

### Event Subscriptions

| Event | Handler | Action |
|-------|---------|--------|
| `TTS_STARTED` | `_on_tts_started` | Store pending text |
| `TTS_COMPLETED` | `_on_tts_completed` | Stop speaking animation |
| `PHONEME_EVENT` | `_on_phoneme_event` | Map phoneme to viseme |
| `EXPRESSION_CHANGE` | `_on_expression_change` | Set emotion |
| `EMOTION_CHANGED` | `_on_emotion_changed` | Set emotion |
| `GAZE_UPDATE` | `_on_gaze_update` | Update eye gaze |
| `MOUTH_SHAPE_UPDATE` | `_on_mouth_shape_update` | Set viseme |
| `BLINK_TRIGGERED` | `_on_blink_triggered` | Trigger blink |
| `WAKE_WORD_DETECTED` | `_on_wake_word` | Set LISTENING + SURPRISED |
| `SPEECH_DETECTED` | `_on_speech_detected` | Set LISTENING |
| `RESPONSE_GENERATING` | `_on_response_generating` | Set THINKING |
| `LIP_SYNC_STARTED` | `_on_lip_sync_started` | Start Rhubarb session |
| `LIP_SYNC_COMPLETED` | `_on_lip_sync_completed` | End Rhubarb session |
| `AUDIO_PLAYBACK_STARTED` | `_on_audio_playback_started` | Animation sync point |

### Public API

**Animation Control**:
- `trigger_blink()` - Trigger eye blink
- `set_gaze(x, y)` - Set gaze position (0-1)
- `set_emotion(emotion)` - Set emotional expression
- `set_animation_state(state)` - Set animation state
- `start_speaking()` - Start speaking animation
- `speak_text(text, duration)` - Text-based lip sync
- `stop_speaking()` - Stop speaking animation
- `set_viseme(viseme)` - Set mouth shape
- `set_viseme_with_params(viseme, params)` - Enhanced viseme control
- `set_rhubarb_shape(shape, intensity)` - Direct Rhubarb shape

**Window Management**:
- `add_window(window_id, settings)` - Add new window
- `remove_window(window_id)` - Remove window
- `get_window_ids()` - List window IDs
- `is_window_alive(window_id)` - Check window status
- `all_windows_alive()` - Check all windows
- `any_window_alive()` - Check any window

**Status**:
- `get_state()` - Get full state with Rhubarb status
- `get_statistics()` - Get display statistics
- `get_rhubarb_config()` - Get Rhubarb config
- `update_rhubarb_config(**kwargs)` - Update Rhubarb config

### Rhubarb Integration

```
TTS_STARTED (audio_file)
     ↓
AnimationService generates lip sync data
     ↓
LIP_SYNC_STARTED (cues, duration)
     ↓
Load cues into RhubarbVisemeController
     ↓
Start controller session
     ↓
AUDIO_PLAYBACK_STARTED (start_timestamp)
     ↓
Controller emits MOUTH_SHAPE_UPDATE events
     ↓
Display renders mouth animations
     ↓
LIP_SYNC_COMPLETED
     ↓
Stop controller, return to idle
```

## Improvements Suggested

### 1. Multi-Character Support
Display multiple animated characters:
```python
class MultiCharacterDisplayManager:
    def __init__(self, character_configs: List[CharacterConfig]):
        self.characters = {
            c.name: DisplayManager(c.settings)
            for c in character_configs
        }

    def speak_character(self, name: str, text: str):
        self.characters[name].speak_text(text)
```

### 2. Expression Blending
Blend multiple emotions:
```python
def set_blended_emotion(self,
    primary: str, secondary: str,
    blend: float = 0.3
):
    """Blend two emotions for nuanced expression."""
    self.decision_module.route_event(
        DisplayEvent.EMOTION_BLEND,
        {'primary': primary, 'secondary': secondary, 'blend': blend}
    )
```

### 3. Animation Presets
Save and load animation presets:
```python
ANIMATION_PRESETS = {
    "excited": {
        "emotion": "HAPPY",
        "blink_rate": 1.5,
        "gaze_jitter": 0.1
    },
    "calm": {
        "emotion": "NEUTRAL",
        "blink_rate": 0.7,
        "gaze_jitter": 0.02
    }
}

def apply_preset(self, preset_name: str):
    preset = ANIMATION_PRESETS.get(preset_name)
    if preset:
        self.set_emotion(preset["emotion"])
        # Apply other settings
```

### 4. Performance Metrics
Track animation performance:
```python
def get_performance_metrics(self) -> Dict[str, float]:
    return {
        'avg_frame_time_ms': self._avg_frame_time,
        'dropped_frames': self._dropped_frames,
        'viseme_latency_ms': self._avg_viseme_latency,
        'fps': self._current_fps
    }
```

### 5. Remote Display Protocol
Support remote displays:
```python
class RemoteDisplayManager(DisplayManager):
    def __init__(self, host: str, port: int):
        self.websocket = WebSocket(host, port)

    def set_emotion(self, emotion: str):
        self.websocket.send({"type": "emotion", "value": emotion})
```

### 6. Screen Layout Manager
Manage multi-monitor layouts:
```python
class ScreenLayoutManager:
    LAYOUTS = {
        "single": [(0, "full")],
        "dual_eyes": [(0, "left_eye"), (0, "right_eye")],
        "triple": [(0, "left_eye"), (1, "mouth"), (2, "right_eye")]
    }

    def apply_layout(self, layout_name: str):
        layout = self.LAYOUTS.get(layout_name)
        for screen_id, window_type in layout:
            self._position_window(window_type, screen_id)
```
