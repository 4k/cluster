"""
STT Engine Protocol.

Defines the interface that all Speech-to-Text engines must implement.
This allows the STT service to be backend-agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional, Union
import numpy as np

from .stt_types import STTCapabilities, STTResult, TranscriptionSegment


class STTEngine(ABC):
    """
    Abstract base class for Speech-to-Text engines.

    All STT implementations (Vosk, Whisper, Azure, Google, etc.) must
    implement this interface to be used with the STT service.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate: int = 16000,
        language: str = "en",
    ) -> STTResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as bytes (int16) or numpy array
            sample_rate: Sample rate of the audio
            language: Language code for recognition

        Returns:
            STTResult containing transcribed text and metadata

        Raises:
            STTRecognitionError: If transcription fails
        """
        pass

    async def transcribe_file(
        self,
        audio_path: Union[str, Path],
        language: str = "en",
    ) -> STTResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code for recognition

        Returns:
            STTResult containing transcribed text and metadata

        Raises:
            STTRecognitionError: If transcription fails
        """
        import wave

        audio_path = Path(audio_path)
        with wave.open(str(audio_path), 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()

        return await self.transcribe(audio_data, sample_rate, language)

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        language: str = "en",
        on_partial: Optional[Callable[[TranscriptionSegment], None]] = None,
    ) -> STTResult:
        """
        Transcribe streaming audio to text.

        Override this for engines that support streaming recognition.
        Default implementation buffers audio and calls transcribe().

        Args:
            audio_stream: Async iterator yielding audio chunks
            sample_rate: Sample rate of the audio
            language: Language code for recognition
            on_partial: Optional callback for partial results

        Yields:
            TranscriptionSegment for each partial/final result

        Returns:
            Final STTResult with complete transcription
        """
        # Default: buffer all audio and transcribe
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        audio_data = b''.join(chunks)
        return await self.transcribe(audio_data, sample_rate, language)

    @abstractmethod
    def create_recognizer(self, sample_rate: int = 16000) -> Any:
        """
        Create a recognizer instance for streaming recognition.

        This is used for real-time transcription where audio
        is fed chunk by chunk.

        Args:
            sample_rate: Sample rate for the recognizer

        Returns:
            Recognizer instance (engine-specific type)
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> STTCapabilities:
        """
        Get the capabilities of this STT engine.

        Returns:
            STTCapabilities describing what this engine can do
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


class STTEngineError(Exception):
    """Base exception for STT engine errors."""
    pass


class STTRecognitionError(STTEngineError):
    """Error during speech recognition."""
    pass


class STTConfigurationError(STTEngineError):
    """Error in engine configuration."""
    pass


class STTUnavailableError(STTEngineError):
    """Engine is not available."""
    pass
