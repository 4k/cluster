"""
Piper TTS Engine.

Wraps the Piper-TTS library to implement the TTSEngine protocol.
Piper is a fast, local neural text-to-speech system optimized for
Raspberry Pi and other embedded devices.

References:
- https://github.com/rhasspy/piper
"""

import logging
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.audio.tts_engine import (
    TTSEngine,
    TTSConfigurationError,
    TTSSynthesisError,
    TTSUnavailableError,
)
from src.core.audio.types import (
    AudioFormat,
    TTSCapabilities,
    TTSResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PiperEngineConfig:
    """Configuration for Piper TTS engine."""
    # Model settings
    model_path: Optional[str] = None
    model_search_dir: str = "~/.local/share/piper/voices"

    # Synthesis settings
    speaker_id: Optional[int] = None
    length_scale: float = 1.0    # Speed: < 1.0 = faster, > 1.0 = slower
    noise_scale: float = 0.667   # Variation in pronunciation
    noise_w: float = 0.8         # Variation in phoneme duration

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PiperEngineConfig":
        """Create from dictionary."""
        return cls(
            model_path=config.get("model_path"),
            model_search_dir=config.get("model_search_dir", cls.model_search_dir),
            speaker_id=config.get("speaker_id"),
            length_scale=config.get("length_scale", cls.length_scale),
            noise_scale=config.get("noise_scale", cls.noise_scale),
            noise_w=config.get("noise_w", cls.noise_w),
        )


class PiperTTSEngine(TTSEngine):
    """
    Piper TTS engine implementation.

    Uses piper-tts library for local neural text-to-speech synthesis.
    Optimized for Raspberry Pi and edge devices.
    """

    def __init__(self, config: Optional[PiperEngineConfig] = None):
        """
        Initialize the Piper TTS engine.

        Args:
            config: Piper engine configuration
        """
        self.config = config or PiperEngineConfig()
        self._voice = None
        self._model_path: Optional[Path] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine by loading the voice model."""
        if self._initialized:
            return

        try:
            from piper.voice import PiperVoice
        except ImportError:
            raise TTSConfigurationError(
                "piper-tts library not installed. "
                "Install with: pip install piper-tts"
            )

        # Find model path
        self._model_path = self._find_model_path()
        if not self._model_path:
            raise TTSConfigurationError(
                f"No Piper voice model found. "
                f"Please set model_path or download a model to {self.config.model_search_dir}"
            )

        # Load voice
        logger.info(f"Loading Piper voice model: {self._model_path}")
        try:
            self._voice = PiperVoice.load(str(self._model_path))
            self._initialized = True
            logger.info(f"Piper initialized: sample_rate={self._voice.config.sample_rate}Hz")
        except Exception as e:
            raise TTSConfigurationError(f"Failed to load Piper voice model: {e}")

    def _find_model_path(self) -> Optional[Path]:
        """Find the voice model path."""
        # Check explicit path first
        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if path.exists():
                return path
            logger.warning(f"Configured model path not found: {path}")

        # Search in model directory
        search_dir = Path(self.config.model_search_dir).expanduser()
        if search_dir.exists():
            models = list(search_dir.glob("*.onnx"))
            if models:
                logger.info(f"Found model: {models[0]}")
                return models[0]

        return None

    async def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        output_format: AudioFormat = AudioFormat.WAV,
    ) -> TTSResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            output_format: Desired audio format (only WAV supported)

        Returns:
            TTSResult with audio file path and metadata
        """
        if not self._initialized:
            await self.initialize()

        if not self._voice:
            raise TTSUnavailableError("Piper voice not loaded")

        if not text or not text.strip():
            raise TTSSynthesisError("Empty text provided")

        if output_format != AudioFormat.WAV:
            logger.warning(f"Piper only supports WAV format, ignoring {output_format}")

        # Determine output path
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)

        try:
            # Synthesize to WAV file
            with wave.open(str(output_path), "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)

            # Calculate duration
            with wave.open(str(output_path), "rb") as wav_file:
                n_frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = n_frames / sample_rate

            logger.debug(f"Synthesized {len(text)} chars to {output_path} ({duration:.2f}s)")

            return TTSResult(
                audio_file=output_path,
                duration=duration,
                sample_rate=self._voice.config.sample_rate,
                format=AudioFormat.WAV,
                text=text,
                engine=self.name,
                metadata={
                    "model": str(self._model_path) if self._model_path else None,
                },
            )

        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            raise TTSSynthesisError(f"Synthesis failed: {e}")

    def get_capabilities(self) -> TTSCapabilities:
        """Get Piper capabilities."""
        sample_rate = 22050
        if self._voice:
            sample_rate = self._voice.config.sample_rate

        return TTSCapabilities(
            name="piper",
            version="1.3.0",
            supported_formats=[AudioFormat.WAV],
            supports_streaming=True,  # Piper supports streaming via synthesize()
            sample_rates=[sample_rate],
            supports_ssml=False,
            supports_phonemes=False,  # Phoneme timing not yet exposed
            max_text_length=None,  # No practical limit
            provides_visemes=False,  # Requires external Rhubarb
            requires_internet=False,
            requires_api_key=False,
        )

    def is_available(self) -> bool:
        """Check if Piper is available."""
        try:
            from piper.voice import PiperVoice
            # Check if we have a model
            return self._find_model_path() is not None
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "piper"

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        self._voice = None
        self._initialized = False
        logger.info("Piper engine shutdown")

    def get_info(self) -> Dict[str, Any]:
        """Get engine info."""
        info = super().get_info()
        info["model_path"] = str(self._model_path) if self._model_path else None
        if self._voice:
            info["sample_rate"] = self._voice.config.sample_rate
            info["num_speakers"] = getattr(self._voice.config, "num_speakers", 1)
        return info
