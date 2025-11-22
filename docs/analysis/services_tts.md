# tts_service.py - Text-to-Speech Service Analysis

## Overview

`tts_service.py` provides high-quality text-to-speech synthesis using Piper-TTS with event-driven queuing to prevent message interruption and Rhubarb lip sync integration.

## File Location
`/home/user/cluster/src/services/tts_service.py`

## Class: TTSService

### Initialization

```python
TTSService(
    model_path: str = None,           # Path to Piper .onnx model
    keep_audio_files: bool = None,    # Save audio for lip sync
    config: TTSServiceConfig = None
)
```

### Key Methods

| Method | Purpose | Async |
|--------|---------|-------|
| `initialize()` | Connect to event bus, start queue processor | Yes |
| `speak_async()` | Async speak with events | Yes |
| `speak()` | Blocking speak | No |
| `speak_to_file()` | Save to WAV file | No |
| `_process_queue()` | Queue processing loop | Yes |
| `_generate_audio_file()` | Create WAV for lip sync | No |
| `_play_audio_file()` | Play existing WAV | No |

### Queue System

Prevents speech interruption with FIFO queue:
```
RESPONSE_GENERATED event
         ↓
    Add to tts_queue
         ↓
   _process_queue() picks up
         ↓
   speak_async() processes
         ↓
   Emit events, play audio
```

### Event Flow

```
TTS_STARTED (with audio_file path)
         ↓
[Wait for LIP_SYNC_READY from AnimationService]
         ↓
AUDIO_PLAYBACK_STARTED (precise timestamp for sync)
         ↓
[Play audio chunks]
         ↓
AUDIO_PLAYBACK_ENDED
         ↓
TTS_COMPLETED
```

### Lip Sync Integration

1. **TTS_STARTED**: Includes `audio_file` path for Rhubarb analysis
2. **Wait**: Waits for `LIP_SYNC_READY` event (max 30s)
3. **AUDIO_PLAYBACK_STARTED**: Includes `start_timestamp` for animation sync
4. Audio and animation play synchronized

### Audio Generation Methods

1. **Streaming** (`_speak_streaming`): Direct from PiperVoice.synthesize()
2. **WAV file** (`_speak_via_wav`): Generate temp WAV, then play

### Audio File Management

- Files saved to `audio_dir` (temp or configured)
- Cleanup maintains max `_max_audio_files` recent files
- Old files deleted when limit exceeded

## Threading

- Queue processor runs as async task
- Audio playback runs in executor (blocking I/O)
- Event emission from threads uses `_emit_event_sync()`

## Improvements Suggested

### 1. SSML Support
Add SSML parsing for prosody control:
```python
def speak_ssml(self, ssml: str) -> bool:
    """Speak with SSML markup for prosody control."""
    text, prosody = self._parse_ssml(ssml)
    # Apply pitch, rate, volume from prosody
```

### 2. Emotion-Based Voice
Adjust voice parameters by emotion:
```python
EMOTION_VOICE_PARAMS = {
    "HAPPY": {"pitch": 1.1, "rate": 1.05},
    "SAD": {"pitch": 0.9, "rate": 0.9},
    "ANGRY": {"pitch": 1.0, "rate": 1.2},
}

def speak_with_emotion(self, text: str, emotion: str) -> bool:
    params = EMOTION_VOICE_PARAMS.get(emotion, {})
    return self.speak(text, **params)
```

### 3. Audio Ducking
Lower volume when user speaks:
```python
async def _on_speech_detected(self, event):
    """Duck audio when user starts speaking."""
    self._duck_volume(0.3)  # 30% volume

async def _on_speech_ended(self, event):
    """Restore volume after user stops."""
    self._restore_volume()
```

### 4. Pre-synthesis
Pre-synthesize common responses:
```python
COMMON_RESPONSES = [
    "I'm listening",
    "One moment please",
    "I didn't understand that",
]

async def presynthsize_common(self):
    """Pre-generate common responses for instant playback."""
    for text in COMMON_RESPONSES:
        path = self._get_cache_path(text)
        if not path.exists():
            self.speak_to_file(text, path)
```

### 5. Multi-Voice Support
Support switching between voices:
```python
def set_voice(self, model_path: str) -> None:
    """Switch to a different voice model."""
    self.voice = PiperVoice.load(model_path)
    logger.info(f"Switched to voice: {model_path}")
```

### 6. Audio Effects
Add optional audio processing:
```python
def _apply_effects(self, audio_data: np.ndarray) -> np.ndarray:
    """Apply optional audio effects."""
    if self.config.reverb:
        audio_data = self._add_reverb(audio_data)
    if self.config.normalize:
        audio_data = self._normalize(audio_data)
    return audio_data
```
