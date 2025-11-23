"""
Wake Word Engine Protocol.

Defines the interface that all wake word detection engines must implement.
This allows the STT service to be backend-agnostic for wake word detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from .stt_types import WakeWordCapabilities, WakeWordResult


class WakeWordEngine(ABC):
    """
    Abstract base class for wake word detection engines.

    All wake word implementations (openwakeword, Porcupine, Snowboy, etc.)
    must implement this interface.
    """

    @abstractmethod
    def detect(
        self,
        audio_data: np.ndarray,
        wake_words: Optional[List[str]] = None,
    ) -> WakeWordResult:
        """
        Detect wake word in audio data.

        Args:
            audio_data: Audio data as numpy array (int16, mono, 16kHz)
            wake_words: Optional list of wake words to detect.
                       If None, detects any configured wake word.

        Returns:
            WakeWordResult with detection status and confidence
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> WakeWordCapabilities:
        """
        Get the capabilities of this wake word engine.

        Returns:
            WakeWordCapabilities describing what this engine can do
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this engine is available and ready to use.

        Returns:
            True if the engine can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this engine."""
        pass

    @property
    def supported_wake_words(self) -> List[str]:
        """Get list of supported wake words."""
        return self.get_capabilities().supported_wake_words

    def get_info(self) -> Dict[str, Any]:
        """Get information about this engine."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "capabilities": self.get_capabilities().to_dict(),
        }

    async def initialize(self) -> None:
        """
        Initialize the engine.

        Override for engines that need async initialization.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the engine and release resources.

        Override for engines that need cleanup.
        """
        pass

    def reset(self) -> None:
        """
        Reset the detection state.

        Call this after a wake word is detected to prepare
        for the next detection cycle.
        """
        pass


class NoneWakeWordEngine(WakeWordEngine):
    """
    A no-op wake word engine that always triggers.

    Use this when wake word detection is disabled (always listening).
    """

    def detect(
        self,
        audio_data: np.ndarray,
        wake_words: Optional[List[str]] = None,
    ) -> WakeWordResult:
        """Always returns detected=True."""
        return WakeWordResult(
            detected=True,
            wake_word="always",
            model_name="none",
            confidence=1.0,
        )

    def get_capabilities(self) -> WakeWordCapabilities:
        return WakeWordCapabilities(
            name="none",
            supported_wake_words=["always"],
            supports_custom_wake_words=False,
        )

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "none"


class WakeWordEngineError(Exception):
    """Base exception for wake word engine errors."""
    pass


class WakeWordDetectionError(WakeWordEngineError):
    """Error during wake word detection."""
    pass


class WakeWordConfigurationError(WakeWordEngineError):
    """Error in engine configuration."""
    pass
