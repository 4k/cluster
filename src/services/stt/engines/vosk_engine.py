"""
Vosk STT Engine.

Wraps the Vosk speech recognition library to implement the STTEngine protocol.
Vosk provides fast, offline speech recognition with good accuracy.

References:
- https://alphacephei.com/vosk/
- https://github.com/alphacep/vosk-api
"""

import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.core.audio.stt_engine import (
    STTEngine,
    STTConfigurationError,
    STTRecognitionError,
)
from src.core.audio.stt_types import (
    STTCapabilities,
    STTResult,
    TranscriptionSegment,
    TranscriptionWord,
)

logger = logging.getLogger(__name__)


@dataclass
class VoskEngineConfig:
    """Configuration for Vosk STT engine."""
    # Model path (None = auto-detect)
    model_path: Optional[str] = None

    # Directories to search for models
    model_search_paths: List[str] = None

    # Enable word-level timestamps
    enable_words: bool = True

    # Sample rate (16000 recommended)
    sample_rate: int = 16000

    def __post_init__(self):
        if self.model_search_paths is None:
            self.model_search_paths = [
                "models/vosk/vosk-model-small-en-us-0.15",
                "models/vosk/vosk-model-en-us-0.22",
                "models/vosk-model-small-en-us-0.15",
            ]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VoskEngineConfig":
        """Create from dictionary."""
        return cls(
            model_path=config.get("model_path"),
            model_search_paths=config.get("model_search_paths"),
            enable_words=config.get("enable_words", True),
            sample_rate=config.get("sample_rate", 16000),
        )


class VoskSTTEngine(STTEngine):
    """
    Vosk-based speech recognition engine.

    Provides fast, offline speech recognition using the Vosk library.
    Supports multiple languages and word-level timestamps.
    """

    def __init__(self, config: Optional[VoskEngineConfig] = None):
        """
        Initialize the Vosk STT engine.

        Args:
            config: Vosk engine configuration
        """
        self.config = config or VoskEngineConfig()
        self._model = None
        self._model_path: Optional[Path] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine by loading the Vosk model."""
        if self._initialized:
            return

        try:
            from vosk import Model as VoskModel
        except ImportError:
            raise STTConfigurationError(
                "vosk library not installed. "
                "Install with: pip install vosk"
            )

        # Find model path
        self._model_path = self._find_model_path()
        if not self._model_path:
            raise STTConfigurationError(
                "Vosk model not found. Please download a model from:\n"
                "https://alphacephei.com/vosk/models\n"
                "Example: vosk-model-small-en-us-0.15\n"
                "Extract to: models/vosk/"
            )

        # Load model
        logger.info(f"Loading Vosk model: {self._model_path}")
        try:
            self._model = VoskModel(str(self._model_path))
            self._initialized = True
            logger.info("Vosk model loaded successfully")
        except Exception as e:
            raise STTConfigurationError(f"Failed to load Vosk model: {e}")

    def _find_model_path(self) -> Optional[Path]:
        """Find the Vosk model path."""
        # Check explicit path
        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if path.exists() and path.is_dir():
                return path
            logger.warning(f"Configured model path not found: {path}")

        # Search configured paths
        for search_path in self.config.model_search_paths:
            path = Path(search_path).expanduser()
            if path.exists() and path.is_dir():
                return path

        # Try glob patterns
        for pattern in glob.glob("models/vosk/vosk-model-*"):
            path = Path(pattern)
            if path.is_dir():
                return path

        # Check common cache location
        cache_path = Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15"
        if cache_path.exists():
            return cache_path

        return None

    async def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate: int = 16000,
        language: str = "en",
    ) -> STTResult:
        """Transcribe audio data to text."""
        if not self._initialized:
            await self.initialize()

        if self._model is None:
            raise STTRecognitionError("Vosk model not loaded")

        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)
            audio_data = audio_data.tobytes()

        try:
            from vosk import KaldiRecognizer

            recognizer = KaldiRecognizer(self._model, sample_rate)
            recognizer.SetWords(self.config.enable_words)

            # Process audio
            recognizer.AcceptWaveform(audio_data)
            result = json.loads(recognizer.FinalResult())

            # Parse result
            text = result.get("text", "")
            words = []

            if "result" in result and self.config.enable_words:
                for word_info in result["result"]:
                    words.append(TranscriptionWord(
                        word=word_info.get("word", ""),
                        start_time=word_info.get("start", 0.0),
                        end_time=word_info.get("end", 0.0),
                        confidence=word_info.get("conf", 1.0),
                    ))

            # Create segment
            segments = []
            if text:
                segment = TranscriptionSegment(
                    text=text,
                    start_time=words[0].start_time if words else 0.0,
                    end_time=words[-1].end_time if words else 0.0,
                    words=words,
                    language=language,
                )
                segments.append(segment)

            # Calculate duration
            duration = len(audio_data) / (2 * sample_rate)  # 2 bytes per sample

            return STTResult(
                text=text,
                segments=segments,
                confidence=0.95,  # Vosk doesn't provide overall confidence
                language=language,
                duration=duration,
                engine=self.name,
                metadata={"model": str(self._model_path) if self._model_path else None},
            )

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            raise STTRecognitionError(f"Transcription failed: {e}")

    def create_recognizer(self, sample_rate: int = 16000) -> Any:
        """Create a Vosk recognizer for streaming."""
        if not self._initialized:
            raise STTRecognitionError("Engine not initialized")

        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._model, sample_rate)
        recognizer.SetWords(self.config.enable_words)
        return recognizer

    def get_capabilities(self) -> STTCapabilities:
        """Get Vosk capabilities."""
        return STTCapabilities(
            name="vosk",
            version="0.3.45",
            supports_streaming=True,
            supports_word_timestamps=True,
            supports_confidence_scores=True,
            supported_languages=["en", "de", "fr", "es", "pt", "ru", "zh", "ja", "ko"],
            required_sample_rate=16000,
            supported_sample_rates=[8000, 16000],
            requires_mono=True,
            requires_internet=False,
            requires_api_key=False,
        )

    def is_available(self) -> bool:
        """Check if Vosk is available."""
        try:
            from vosk import Model
            return self._find_model_path() is not None
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "vosk"

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        self._model = None
        self._initialized = False
        logger.info("Vosk engine shutdown")

    def get_info(self) -> Dict[str, Any]:
        """Get engine info."""
        info = super().get_info()
        info["model_path"] = str(self._model_path) if self._model_path else None
        return info
