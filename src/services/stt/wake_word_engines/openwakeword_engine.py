"""
OpenWakeWord Engine.

Wraps the openwakeword library to implement the WakeWordEngine protocol.
OpenWakeWord provides efficient, privacy-focused wake word detection.

References:
- https://github.com/dscripka/openWakeWord
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.audio.wake_word_engine import (
    WakeWordEngine,
    WakeWordConfigurationError,
    WakeWordDetectionError,
)
from src.core.audio.stt_types import WakeWordCapabilities, WakeWordResult

logger = logging.getLogger(__name__)


@dataclass
class OpenWakeWordConfig:
    """Configuration for OpenWakeWord engine."""
    # Wake word to detect (e.g., "jarvis", "alexa", "hey_mycroft")
    wake_word: str = "jarvis"

    # Detection threshold (0.0 - 1.0)
    threshold: float = 0.5

    # Inference framework ("onnx" or "tflite")
    inference_framework: str = "onnx"

    # Custom model paths (optional)
    custom_model_paths: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "OpenWakeWordConfig":
        """Create from dictionary."""
        return cls(
            wake_word=config.get("wake_word", cls.wake_word),
            threshold=config.get("threshold", cls.threshold),
            inference_framework=config.get("inference_framework", cls.inference_framework),
            custom_model_paths=config.get("custom_model_paths", []),
        )


class OpenWakeWordEngine(WakeWordEngine):
    """
    OpenWakeWord-based wake word detection engine.

    Provides efficient, privacy-focused wake word detection using
    pre-trained models for common wake words.
    """

    def __init__(self, config: Optional[OpenWakeWordConfig] = None):
        """
        Initialize the OpenWakeWord engine.

        Args:
            config: Engine configuration
        """
        self.config = config or OpenWakeWordConfig()
        self._model = None
        self._initialized = False
        self._available_models: List[str] = []

    async def initialize(self) -> None:
        """Initialize the engine by loading the model."""
        if self._initialized:
            return

        try:
            from openwakeword.model import Model
        except ImportError:
            raise WakeWordConfigurationError(
                "openwakeword library not installed. "
                "Install with: pip install openwakeword"
            )

        logger.info("Initializing OpenWakeWord...")
        try:
            self._model = Model(inference_framework=self.config.inference_framework)
            self._available_models = list(self._model.models.keys())
            self._initialized = True
            logger.info(f"OpenWakeWord initialized with models: {self._available_models}")
        except Exception as e:
            raise WakeWordConfigurationError(f"Failed to initialize OpenWakeWord: {e}")

    def detect(
        self,
        audio_data: np.ndarray,
        wake_words: Optional[List[str]] = None,
    ) -> WakeWordResult:
        """
        Detect wake word in audio data.

        Args:
            audio_data: Audio data as numpy array (int16, mono, 16kHz)
            wake_words: Optional list of wake words to detect

        Returns:
            WakeWordResult with detection status
        """
        if not self._initialized or self._model is None:
            raise WakeWordDetectionError("Engine not initialized")

        # Target wake word(s)
        target_words = wake_words or [self.config.wake_word]

        try:
            # Get prediction from model
            prediction = self._model.predict(audio_data)

            # Check all models for wake word detection
            max_score = 0.0
            max_model = None
            detected = False
            detected_word = ""

            for model_name, score in prediction.items():
                # Track highest score
                if score > max_score:
                    max_score = score
                    max_model = model_name

                # Check if this model matches our target wake word(s)
                model_simple = model_name.lower().replace("hey_", "").replace("_", " ")

                for target in target_words:
                    target_simple = target.lower().replace("_", " ")

                    if target_simple in model_simple or model_simple in target_simple:
                        if score > self.config.threshold:
                            detected = True
                            detected_word = target
                            logger.info(
                                f"Wake word '{target}' detected "
                                f"(model: {model_name}, confidence: {score:.3f})"
                            )
                            break

                if detected:
                    break

            return WakeWordResult(
                detected=detected,
                wake_word=detected_word if detected else "",
                model_name=max_model or "",
                confidence=max_score,
                timestamp=time.time(),
                all_scores=prediction,
            )

        except Exception as e:
            logger.error(f"Wake word detection failed: {e}")
            raise WakeWordDetectionError(f"Detection failed: {e}")

    def get_capabilities(self) -> WakeWordCapabilities:
        """Get OpenWakeWord capabilities."""
        return WakeWordCapabilities(
            name="openwakeword",
            version="0.5.0",
            supported_wake_words=self._available_models or [
                "alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "computer"
            ],
            supports_custom_wake_words=True,
            supports_multiple_wake_words=True,
            required_sample_rate=16000,
            chunk_size_samples=1280,  # 80ms at 16kHz
            requires_internet=False,
        )

    def is_available(self) -> bool:
        """Check if OpenWakeWord is available."""
        try:
            from openwakeword.model import Model
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "openwakeword"

    @property
    def supported_wake_words(self) -> List[str]:
        return self._available_models or []

    def reset(self) -> None:
        """Reset detection state."""
        if self._model:
            # OpenWakeWord maintains internal state, reset by creating new instance
            # For now, we just log - the model handles state internally
            logger.debug("Wake word detection state reset")

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        self._model = None
        self._initialized = False
        self._available_models = []
        logger.info("OpenWakeWord engine shutdown")

    def get_info(self) -> Dict[str, Any]:
        """Get engine info."""
        info = super().get_info()
        info["threshold"] = self.config.threshold
        info["wake_word"] = self.config.wake_word
        info["available_models"] = self._available_models
        return info
