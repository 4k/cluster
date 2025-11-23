"""
Viseme Provider Protocol.

Defines the interface that all viseme extraction providers must implement.
This allows the animation service to be backend-agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .types import VisemeSequence, VisemeShape


class VisemeProvider(ABC):
    """
    Abstract base class for viseme extraction providers.

    All viseme implementations (Rhubarb, text-based, Azure inline, etc.)
    must implement this interface to be used with the animation service.
    """

    @abstractmethod
    async def extract_visemes(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> VisemeSequence:
        """
        Extract viseme timing from audio and/or text.

        Different providers require different inputs:
        - Rhubarb: Requires audio_path, text is optional but improves accuracy
        - Text-based: Requires text and audio_duration, no audio needed
        - Azure inline: Visemes come from TTS, this may be a no-op

        Args:
            audio_path: Path to the audio file (WAV format recommended)
            text: Optional transcript text for improved accuracy
            audio_duration: Duration of the audio in seconds (for text-based)

        Returns:
            VisemeSequence with timing cues

        Raises:
            VisemeExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is available and ready to use.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this provider."""
        pass

    @property
    def requires_audio(self) -> bool:
        """
        Whether this provider requires an audio file.

        Returns:
            True if audio_path is required, False if text-only
        """
        return True

    @property
    def requires_text(self) -> bool:
        """
        Whether this provider requires text input.

        Returns:
            True if text is required
        """
        return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider info
        """
        return {
            "name": self.name,
            "available": self.is_available(),
            "requires_audio": self.requires_audio,
            "requires_text": self.requires_text,
        }

    async def initialize(self) -> None:
        """
        Initialize the provider.

        Override this for providers that need async initialization.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the provider and release resources.

        Override this for providers that need cleanup.
        """
        pass


class NoneVisemeProvider(VisemeProvider):
    """
    A no-op viseme provider that returns empty sequences.

    Use this when lip sync is disabled.
    """

    async def extract_visemes(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> VisemeSequence:
        """Return an empty viseme sequence."""
        return VisemeSequence.empty(duration=audio_duration or 0.0)

    def is_available(self) -> bool:
        """Always available."""
        return True

    @property
    def name(self) -> str:
        return "none"

    @property
    def requires_audio(self) -> bool:
        return False


class VisemeProviderError(Exception):
    """Base exception for viseme provider errors."""
    pass


class VisemeExtractionError(VisemeProviderError):
    """Error during viseme extraction."""
    pass


class VisemeConfigurationError(VisemeProviderError):
    """Error in provider configuration."""
    pass


# Standard viseme mappings used across providers
# Maps common phoneme representations to our standard VisemeShape

PHONEME_TO_VISEME: Dict[str, VisemeShape] = {
    # Silence
    "sil": VisemeShape.SILENCE,
    "sp": VisemeShape.SILENCE,
    "": VisemeShape.SILENCE,

    # Bilabial stops (B, M, P)
    "b": VisemeShape.BMP,
    "m": VisemeShape.BMP,
    "p": VisemeShape.BMP,

    # Labiodental fricatives (F, V)
    "f": VisemeShape.FV,
    "v": VisemeShape.FV,

    # Dental fricatives (TH)
    "th": VisemeShape.TH,
    "dh": VisemeShape.TH,

    # Alveolar consonants (L, N, T, D, S, Z)
    "l": VisemeShape.LNT,
    "n": VisemeShape.LNT,
    "t": VisemeShape.LNT,
    "d": VisemeShape.LNT,
    "s": VisemeShape.LNT,
    "z": VisemeShape.LNT,

    # Velar/palatal consonants
    "k": VisemeShape.LNT,
    "g": VisemeShape.LNT,
    "ng": VisemeShape.LNT,
    "y": VisemeShape.EE,
    "ch": VisemeShape.LNT,
    "jh": VisemeShape.LNT,
    "sh": VisemeShape.LNT,
    "zh": VisemeShape.LNT,

    # Glottal
    "hh": VisemeShape.AH,
    "h": VisemeShape.AH,

    # Semi-vowels
    "w": VisemeShape.OO,
    "r": VisemeShape.AH,

    # Vowels - open
    "aa": VisemeShape.AH,
    "ae": VisemeShape.AH,
    "ah": VisemeShape.AH,
    "ax": VisemeShape.AH,
    "er": VisemeShape.AH,

    # Vowels - front/spread
    "iy": VisemeShape.EE,
    "ih": VisemeShape.EE,
    "ey": VisemeShape.EE,
    "eh": VisemeShape.EE,

    # Vowels - rounded
    "ao": VisemeShape.OH,
    "ow": VisemeShape.OH,
    "oy": VisemeShape.OH,

    # Vowels - close rounded
    "uw": VisemeShape.OO,
    "uh": VisemeShape.OO,

    # Diphthongs
    "aw": VisemeShape.OH,
    "ay": VisemeShape.AH,
}

# Character to viseme mapping for simple text-based animation
# Used when no audio analysis is available
CHAR_TO_VISEME: Dict[str, VisemeShape] = {
    # Vowels
    "a": VisemeShape.AH,
    "e": VisemeShape.EE,
    "i": VisemeShape.EE,
    "o": VisemeShape.OH,
    "u": VisemeShape.OO,

    # Consonants
    "b": VisemeShape.BMP,
    "c": VisemeShape.LNT,
    "d": VisemeShape.LNT,
    "f": VisemeShape.FV,
    "g": VisemeShape.LNT,
    "h": VisemeShape.AH,
    "j": VisemeShape.LNT,
    "k": VisemeShape.LNT,
    "l": VisemeShape.LNT,
    "m": VisemeShape.BMP,
    "n": VisemeShape.LNT,
    "p": VisemeShape.BMP,
    "q": VisemeShape.OO,
    "r": VisemeShape.AH,
    "s": VisemeShape.LNT,
    "t": VisemeShape.LNT,
    "v": VisemeShape.FV,
    "w": VisemeShape.OO,
    "x": VisemeShape.LNT,
    "y": VisemeShape.EE,
    "z": VisemeShape.LNT,

    # Space = silence
    " ": VisemeShape.SILENCE,
}
