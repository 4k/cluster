# animation.py - Animation Service Analysis

## Overview

`animation.py` coordinates facial animations with the event bus, watching for TTS events and generating synchronized lip sync animations using Rhubarb Lip Sync.

## File Location
`/home/user/cluster/src/features/display/animation.py`

## Key Responsibilities

1. Monitor TTS events for audio file availability
2. Generate lip sync data using Rhubarb (synchronously before playback)
3. Schedule and emit viseme events synchronized with audio playback
4. Coordinate with display manager for smooth animations

## Classes

### LipSyncSession (Dataclass)

```python
@dataclass
class LipSyncSession:
    session_id: str
    lip_sync_data: LipSyncData
    start_time: float          # When audio playback started
    is_active: bool = True
    correlation_id: Optional[str] = None

    # Methods:
    def get_elapsed_time() -> float
    def is_complete() -> bool
```

### AnimationService

**Initialization**:
```python
AnimationService(
    rhubarb_path: str = None,       # Path to rhubarb executable
    temp_dir: str = None,           # For temporary files
    viseme_emit_interval: float = 0.033,  # ~30 FPS
    enable_fallback: bool = True     # Text-based fallback
)
```

**Event Subscriptions**:
| Event | Handler | Purpose |
|-------|---------|---------|
| `TTS_STARTED` | `_on_tts_started` | Generate lip sync synchronously |
| `AUDIO_PLAYBACK_STARTED` | `_on_audio_playback_started` | Start playback session |
| `AUDIO_PLAYBACK_ENDED` | `_on_audio_playback_ended` | Stop session |
| `TTS_COMPLETED` | `_on_tts_completed` | Ensure cleanup |
| `SYSTEM_STOPPED` | `_on_system_stopped` | Shutdown |

**Event Emissions**:
| Event | When |
|-------|------|
| `LIP_SYNC_READY` | Lip sync data generated (or fallback) |
| `LIP_SYNC_STARTED` | Playback session started |
| `LIP_SYNC_COMPLETED` | Playback session ended |
| `MOUTH_SHAPE_UPDATE` | Every viseme emit interval |

## Lip Sync Flow

```
TTS_STARTED (with audio_file)
         ↓
_on_tts_started()
         ↓
_generate_lip_sync(audio_file, text)  [BLOCKING - synchronous]
         ↓
Cache LipSyncData with correlation_id
         ↓
Emit LIP_SYNC_READY (TTS can proceed)
         ↓
AUDIO_PLAYBACK_STARTED (with start_timestamp)
         ↓
_on_audio_playback_started()
         ↓
Retrieve cached LipSyncData
         ↓
_start_lip_sync_session(data, correlation_id, start_timestamp)
         ↓
Emit LIP_SYNC_STARTED
         ↓
_viseme_emission_loop() runs at 30Hz
         ↓
For each tick:
  - Get elapsed time since start_timestamp
  - Find viseme at elapsed time
  - Convert to internal viseme
  - Emit MOUTH_SHAPE_UPDATE
         ↓
When session complete:
  - Emit LIP_SYNC_COMPLETED
  - Emit MOUTH_SHAPE_UPDATE with SILENCE
```

## Timing Synchronization

Critical for lip sync accuracy:

1. **TTS_STARTED**: AnimationService generates lip sync data (blocks TTS)
2. **LIP_SYNC_READY**: TTS proceeds with playback
3. **AUDIO_PLAYBACK_STARTED**: Includes `start_timestamp` from audio thread
4. Animation uses `start_timestamp` as time reference (not event receive time)
5. Elapsed time = `current_time - start_timestamp`
6. Viseme selected based on elapsed time matching cue timings

## Statistics Tracking

```python
self.stats = {
    "sessions_started": 0,
    "sessions_completed": 0,
    "rhubarb_successes": 0,
    "rhubarb_failures": 0,
    "fallback_used": 0,
    "visemes_emitted": 0,
}
```

## Improvements Suggested

### 1. Parallel Rhubarb Processing
Pre-process while TTS generates audio:
```python
async def _parallel_generation(self, text: str):
    """Start Rhubarb as soon as text is available."""
    # Create placeholder audio
    # Start Rhubarb in background
    # Merge results when audio ready
```

### 2. Viseme Interpolation
Smooth transitions between visemes:
```python
def _interpolate_viseme(self,
    current: str, target: str,
    progress: float
) -> Dict[str, float]:
    """Interpolate between two visemes."""
    current_params = VISEME_PARAMS[current]
    target_params = VISEME_PARAMS[target]
    return {
        k: current_params[k] * (1 - progress) + target_params[k] * progress
        for k in current_params
    }
```

### 3. Emotion-Aware Visemes
Adjust visemes based on emotion:
```python
def _apply_emotion_modifier(self,
    viseme: str,
    emotion: str
) -> Dict[str, float]:
    """Modify viseme parameters based on emotion."""
    params = VISEME_PARAMS[viseme].copy()
    if emotion == "HAPPY":
        params['smile'] = min(1.0, params.get('smile', 0) + 0.3)
    return params
```

### 4. Adaptive Frame Rate
Adjust emission rate based on system load:
```python
def _adjust_frame_rate(self):
    """Adjust viseme emit rate based on CPU load."""
    cpu_load = psutil.cpu_percent()
    if cpu_load > 80:
        self.viseme_emit_interval = 0.066  # 15 FPS
    elif cpu_load > 60:
        self.viseme_emit_interval = 0.05   # 20 FPS
    else:
        self.viseme_emit_interval = 0.033  # 30 FPS
```

### 5. Lip Sync Preview
Preview lip sync before playback:
```python
async def preview_lip_sync(self,
    audio_file: str
) -> List[Tuple[float, str]]:
    """Generate and return lip sync timeline without playback."""
    data = await self._generate_lip_sync(audio_file, None)
    return [(cue.start_time, cue.viseme.value) for cue in data.cues]
```

### 6. Multiple Concurrent Sessions
Support overlapping audio (e.g., sound effects):
```python
def _get_active_sessions_at_time(self,
    timestamp: float
) -> List[LipSyncSession]:
    """Get all active sessions at a timestamp."""
    return [s for s in self._active_sessions.values()
            if s.is_active and not s.is_complete()]
```
