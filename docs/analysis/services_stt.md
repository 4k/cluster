# stt_service.py - Speech-to-Text Service Analysis

## Overview

`stt_service.py` implements wake word detection using OpenWakeWord followed by speech transcription using Vosk. It's the primary audio input pipeline for the voice assistant.

## File Location
`/home/user/cluster/src/services/stt_service.py`

## Class: STTService

### Initialization

```python
STTService(
    wake_word: str = None,              # "jarvis", "alexa", etc.
    chunk_size: int = None,             # Audio chunk size (1280 = 80ms @ 16kHz)
    sample_rate: int = None,            # 16000 Hz required
    threshold: float = None,            # Detection threshold 0-1
    device_index: int = None,           # Audio input device
    verbose: bool = None,               # Show detection scores
    vosk_model_path: str = None,        # Path to Vosk model
    config: STTServiceConfig = None
)
```

### Key Methods

| Method | Purpose | Async |
|--------|---------|-------|
| `initialize()` | Connect to event bus | Yes |
| `start()` | Begin listening loop | No (blocks) |
| `stop()` | Stop listening | No |
| `_detect_wake_word()` | Check audio for wake word | No |
| `_transcribe_speech()` | Transcribe after wake word | Yes |
| `_transcribe_speech_blocking()` | Vosk transcription | No (in thread) |

### Audio Pipeline

```
PyAudio Stream (16kHz, mono)
         ↓
    Read chunk (1280 samples)
         ↓
    Convert to int16 numpy array
         ↓
    [If stereo] Convert to mono
         ↓
    [If not 16kHz] Resample
         ↓
    OpenWakeWord.predict()
         ↓
    [If detected] → Vosk transcription
         ↓
    Emit SPEECH_DETECTED event
```

### Event Emissions

| Event | When |
|-------|------|
| `WAKE_WORD_DETECTED` | Wake word detected with confidence |
| `AUDIO_STARTED` | Recording started |
| `SPEECH_DETECTED` | Transcription complete |
| `SPEECH_ENDED` | User finished speaking |
| `AUDIO_STOPPED` | Recording stopped |
| `ERROR_OCCURRED` | Processing error |

### Device Handling

Robust device initialization with fallbacks:
1. Try 16kHz first (OpenWakeWord requirement)
2. Fall back to device native rate with resampling
3. Try mono, then stereo with mono conversion
4. Try multiple buffer sizes (1280, 1024, 2048, 4096)

### Voice Activity Detection

Simple RMS-based silence detection:
```python
rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
if rms < silence_threshold_rms:
    silent_chunks += 1
    if silent_chunks > max_silent_chunks and final_text:
        break  # End of utterance
```

## Thread Safety

- Main listening loop runs in main thread
- Event emission uses `run_coroutine_threadsafe()`
- Vosk transcription runs in executor thread pool

## Improvements Suggested

### 1. Continuous Listening Mode
Add option for always-on without wake word:
```python
class ListeningMode(Enum):
    WAKE_WORD = "wake_word"
    CONTINUOUS = "continuous"
    PUSH_TO_TALK = "push_to_talk"
```

### 2. Multiple Wake Words
Support multiple wake word detection:
```python
def _detect_any_wake_word(self, audio_data) -> Tuple[bool, str, float]:
    """Check for any configured wake words."""
    for wake_word in self.wake_words:
        detected, model, score = self._detect_wake_word(audio_data, wake_word)
        if detected:
            return detected, wake_word, score
    return False, None, 0.0
```

### 3. Noise Profiling
Adapt to ambient noise:
```python
def _calibrate_noise_level(self, duration: float = 2.0) -> float:
    """Measure ambient noise level for adaptive threshold."""
    samples = []
    for _ in range(int(duration * self.sample_rate / self.chunk_size)):
        data = self.stream.read(self.chunk_size)
        samples.append(self._calculate_rms(data))
    return np.mean(samples) * 1.5  # 50% above ambient
```

### 4. Utterance Confidence
Return confidence with transcription:
```python
@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    words: List[Dict[str, Any]]  # Word-level timestamps
    duration: float
```

### 5. Language Detection
Auto-detect spoken language:
```python
async def detect_language(self, audio: np.ndarray) -> str:
    """Detect language from audio sample."""
    # Use language identification model
```

### 6. Barge-In Support
Allow interrupting assistant speech:
```python
def enable_barge_in(self) -> None:
    """Enable detection during TTS playback."""
    self.barge_in_enabled = True
    # Lower threshold for faster detection
    self.barge_in_threshold = self.threshold * 0.8
```
