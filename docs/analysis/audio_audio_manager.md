# audio_manager.py - Audio Management Analysis

## Overview

`audio_manager.py` provides centralized audio device management and routing for the voice assistant, handling input/output streams and device selection.

## File Location
`/home/user/cluster/src/audio/audio_manager.py`

## Purpose

The AudioManager serves as the audio subsystem controller, managing:
- Input device selection and streaming
- Output device selection and playback
- Audio format conversion
- Buffer management
- Device hot-plugging

## Key Responsibilities

### Device Management

```
┌─────────────────────────────────────────────────────────────┐
│                       AudioManager                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │   Input Device  │        │  Output Device  │             │
│  │  (Microphone)   │        │   (Speaker)     │             │
│  └────────┬────────┘        └────────┬────────┘             │
│           ↓                          ↑                       │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │  Input Stream   │        │ Output Stream   │             │
│  │  (16kHz, mono)  │        │ (22050Hz, mono) │             │
│  └────────┬────────┘        └────────┬────────┘             │
│           ↓                          ↑                       │
│  ┌─────────────────────────────────────────────┐            │
│  │              Audio Router                     │            │
│  │   STT ← Input    |    Output → TTS           │            │
│  │   AmbientSTT ← Input                         │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Audio Routing

- **Input** → STT Service (wake word detection, transcription)
- **Input** → Ambient STT (continuous monitoring)
- **Output** ← TTS Service (speech synthesis)

## Integration Points

| Component | Input/Output | Format |
|-----------|-------------|--------|
| STT Service | Input | 16000Hz, int16, mono |
| Ambient STT | Input | 16000Hz, int16, mono |
| TTS Service | Output | 22050Hz, int16, mono |
| Wake Word | Input | 16000Hz, int16, mono |

## Configuration

```python
@dataclass
class AudioManagerConfig:
    input_device_index: Optional[int] = None   # Auto-select if None
    output_device_index: Optional[int] = None  # Auto-select if None
    input_sample_rate: int = 16000
    output_sample_rate: int = 22050
    input_channels: int = 1
    output_channels: int = 1
    buffer_size: int = 1024
    format: int = pyaudio.paInt16
```

## Device Selection Logic

```
1. Check specified device index
2. If None, enumerate available devices
3. Filter by capability (input/output)
4. Prefer default system device
5. Fall back to first available
```

## Improvements Suggested

### 1. Virtual Device Support
Support virtual audio devices for routing:
```python
class VirtualAudioDevice:
    """Virtual device for audio routing."""
    def __init__(self, name: str):
        self.buffer = queue.Queue()
        self.name = name

    def write(self, data: bytes) -> None:
        self.buffer.put(data)

    def read(self, size: int) -> bytes:
        return self.buffer.get()
```

### 2. Audio Level Monitoring
Real-time audio level monitoring:
```python
def get_input_level(self) -> float:
    """Get current input audio level (0-1)."""
    return self._current_rms / self._max_rms

def get_output_level(self) -> float:
    """Get current output audio level (0-1)."""
    return self._output_rms / self._max_output_rms
```

### 3. Device Hot-Plugging
Handle device connection/disconnection:
```python
async def _device_monitor_loop(self):
    """Monitor for device changes."""
    while self.is_running:
        current_devices = self._enumerate_devices()
        if current_devices != self._known_devices:
            await self._handle_device_change(current_devices)
        await asyncio.sleep(2.0)
```

### 4. Audio Ducking
Lower output volume during input:
```python
def set_ducking(self, enabled: bool, level: float = 0.3):
    """Enable audio ducking during input."""
    self.ducking_enabled = enabled
    self.ducking_level = level
```

### 5. Latency Optimization
Minimize audio latency:
```python
def optimize_for_latency(self):
    """Configure for lowest latency."""
    self.buffer_size = 256
    self._update_streams()
```

### 6. Audio Recording
Record audio for debugging/training:
```python
def start_recording(self, output_path: str):
    """Start recording all audio."""
    self.recording = wave.open(output_path, 'wb')
    self.is_recording = True

def stop_recording(self):
    """Stop and save recording."""
    self.is_recording = False
    self.recording.close()
```
