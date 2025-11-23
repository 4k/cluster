"""
TTS Engine Protocol.

Defines the interface that all TTS engines must implement.
This allows the TTS service to be backend-agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from .types import AudioFormat, TTSCapabilities, TTSResult


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.

    All TTS implementations (Piper, Azure, OpenAI, etc.) must implement
    this interface to be used with the TTS service.
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        output_format: AudioFormat = AudioFormat.WAV,
    ) -> TTSResult:
        """
        Synthesize text to speech.

        Args:
            text: The text to synthesize
            output_path: Optional path to save audio file. If None, audio
                        data will be returned in memory or a temp file.
            output_format: Desired audio format (if supported)

        Returns:
            TTSResult containing audio file path or data, duration,
            and optionally inline visemes if the engine provides them.

        Raises:
            TTSSynthesisError: If synthesis fails
        """
        pass

    async def synthesize_streaming(
        self,
        text: str,
        output_format: AudioFormat = AudioFormat.PCM,
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech with streaming output.

        Override this method for engines that support streaming.
        Default implementation falls back to non-streaming synthesis.

        Args:
            text: The text to synthesize
            output_format: Desired audio format (typically PCM for streaming)

        Yields:
            Audio data chunks as bytes

        Raises:
            TTSSynthesisError: If synthesis fails
            NotImplementedError: If streaming is not supported
        """
        # Default: fall back to non-streaming
        result = await self.synthesize(text, output_format=output_format)
        if result.audio_data:
            yield result.audio_data
        elif result.audio_file and result.audio_file.exists():
            with open(result.audio_file, "rb") as f:
                while chunk := f.read(4096):
                    yield chunk

    @abstractmethod
    def get_capabilities(self) -> TTSCapabilities:
        """
        Get the capabilities of this TTS engine.

        Returns:
            TTSCapabilities describing what this engine can do
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

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this engine.

        Returns:
            Dictionary with engine info
        """
        caps = self.get_capabilities()
        return {
            "name": self.name,
            "available": self.is_available(),
            "capabilities": caps.to_dict(),
        }

    async def initialize(self) -> None:
        """
        Initialize the engine.

        Override this for engines that need async initialization
        (e.g., loading models, connecting to APIs).
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the engine and release resources.

        Override this for engines that need cleanup.
        """
        pass


class TTSEngineError(Exception):
    """Base exception for TTS engine errors."""
    pass


class TTSSynthesisError(TTSEngineError):
    """Error during speech synthesis."""
    pass


class TTSConfigurationError(TTSEngineError):
    """Error in engine configuration."""
    pass


class TTSUnavailableError(TTSEngineError):
    """Engine is not available."""
    pass
