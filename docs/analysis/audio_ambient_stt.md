# ambient_stt.py - Ambient Speech-to-Text Analysis

## Overview

`ambient_stt.py` provides always-on continuous speech recognition that runs parallel to the wake word system. It tags transcriptions as 'ambient' or 'wakeword' mode based on wake word activation state.

## File Location
`/home/user/cluster/src/audio/ambient_stt.py`

## Key Concepts

### Dual Mode Operation

```
                    ┌─────────────────────────────────────────┐
                    │          Continuous Audio Stream         │
                    └────────────────────┬────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────┐
                    │            AmbientSTT Process            │
                    │     (Always running, no VAD gating)      │
                    └────────────────────┬────────────────────┘
                                         ↓
              ┌────────────────────┬─────┴─────┬───────────────────┐
              ↓                    ↓           ↓                   ↓
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
    │  mode='ambient' │  │mode='wakeword'  │  │  timeout → ambient  │
    │ (passive listen)│  │ (active session)│  │     (5s default)    │
    └─────────────────┘  └─────────────────┘  └─────────────────────┘
```

## Classes

### AmbientSTTConfig (Dataclass)

```python
@dataclass
class AmbientSTTConfig:
    enabled: bool = True
    model_path: str = "models/vosk-model-small-en-us-0.15"  # Lighter model
    language: str = "en"
    sample_rate: int = 16000
    partial_results: bool = True
    confidence_threshold: float = 0.3      # Lower for ambient
    max_alternatives: int = 3
    wake_word_timeout: float = 5.0         # Seconds in wakeword mode
    frame_skip: int = 1                    # Performance optimization
    min_confidence: float = 0.3
```

### AmbientSTT

**Purpose**: Real continuous speech recognition using Vosk.

**Key Methods**:

| Method | Purpose | Async |
|--------|---------|-------|
| `initialize()` | Load Vosk model | Yes |
| `process_audio()` | Process audio frame | Yes |
| `on_wake_word_detected()` | Switch to wakeword mode | No |
| `_determine_mode()` | Get current mode with timeout | No |
| `set_callbacks()` | Set result callbacks | No |
| `get_state()` | Get current state | No |
| `reset()` | Reset state | No |
| `cleanup()` | Release resources | No |

**State Tracking**:
- `is_wake_word_active` - Currently in wakeword mode
- `last_wake_word_time` - Time of last wake word
- `current_utterance` - Current final text
- `partial_text` - In-progress text
- `frame_counter` - For frame skipping

### MockAmbientSTT

**Purpose**: Development/testing mock that generates periodic detections.

**Features**:
- Generates detection every 20 seconds
- Random confidence values
- Respects wake word mode
- Same interface as real implementation

## Event Emissions

| Event | Data | When |
|-------|------|------|
| `AMBIENT_SPEECH_DETECTED` | text, confidence, mode, timestamp, duration | Ambient mode final result |
| `WAKEWORD_SPEECH_DETECTED` | text, confidence, mode, timestamp, duration | Wakeword mode final result |

## Factory Function

```python
def create_ambient_stt(config: AmbientSTTConfig, mock: bool = False) -> Any:
    """Create ambient STT instance with fallback to mock."""
    if mock:
        return MockAmbientSTT(config)
    try:
        return AmbientSTT(config)
    except ImportError:
        return MockAmbientSTT(config)  # Fallback
```

## Callback System

```python
ambient_stt.set_callbacks(
    on_ambient_result=async_handler,      # Ambient mode detections
    on_wakeword_result=async_handler,     # Wakeword mode detections
    on_partial_result=async_handler       # Partial results (both modes)
)
```

## Performance Optimization

- **Frame skipping**: Process every Nth frame for CPU reduction
- **Lighter model**: Uses small Vosk model vs full model
- **No VAD gating**: Continuous processing simplifies logic

## Improvements Suggested

### 1. Speaker Diarization
Identify different speakers:
```python
def identify_speaker(self, audio_data: np.ndarray) -> str:
    """Identify speaker from voice signature."""
    embedding = self.speaker_model.encode(audio_data)
    return self.speaker_db.match(embedding)
```

### 2. Keyword Spotting
Listen for specific phrases:
```python
class KeywordSpotter:
    keywords = ["stop", "cancel", "help", "repeat"]

    def check_keywords(self, text: str) -> Optional[str]:
        for keyword in self.keywords:
            if keyword in text.lower():
                return keyword
        return None
```

### 3. Language Switching
Support multi-language recognition:
```python
def set_language(self, language: str) -> None:
    """Switch recognition language."""
    model_path = self.language_models.get(language)
    if model_path:
        self._load_model(model_path)
        self.config.language = language
```

### 4. Noise Filtering
Apply noise reduction before recognition:
```python
import noisereduce as nr

def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
    """Apply noise reduction."""
    return nr.reduce_noise(y=audio_data, sr=self.config.sample_rate)
```

### 5. Context Injection
Improve recognition with expected vocabulary:
```python
def set_vocabulary_hint(self, words: List[str]) -> None:
    """Set expected vocabulary for better accuracy."""
    grammar = json.dumps(words)
    self.recognizer.SetGrammar(grammar)
```

### 6. Confidence Calibration
Calibrate confidence scores:
```python
def calibrate_confidence(self, samples: List[Tuple[str, str, float]]) -> None:
    """Calibrate confidence scores with known samples."""
    # samples: [(audio_path, expected_text, label)]
    # Fit calibration curve
```
