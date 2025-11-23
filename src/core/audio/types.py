"""
Core types for TTS and viseme extraction.

These types are backend-agnostic and used across all TTS engines
and viseme providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AudioFormat(Enum):
    """Supported audio output formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    PCM = "pcm"  # Raw PCM data


class VisemeShape(Enum):
    """
    Standard viseme shapes based on Preston Blair's phoneme groups.

    These are the canonical mouth shapes used across all viseme providers.
    Each provider maps its native shapes to these standard shapes.
    """
    # Basic shapes (minimum required)
    SILENCE = "SILENCE"  # Rest/idle position
    BMP = "BMP"          # Closed mouth - B, M, P sounds
    FV = "FV"            # Upper teeth on lower lip - F, V sounds
    TH = "TH"            # Tongue between teeth - TH sounds
    LNT = "LNT"          # Tongue behind teeth - L, N, T, D sounds

    # Vowel shapes
    AH = "AH"            # Open mouth - AH, EH vowels
    EE = "EE"            # Wide/smile - EE, EY vowels
    OH = "OH"            # Rounded - OH, AO vowels
    OO = "OO"            # Puckered - OO, UW vowels

    # Extended shapes (optional, for higher quality)
    WQ = "WQ"            # W, Q sounds (lip rounding)
    REST = "REST"        # Neutral rest position (different from silence)


@dataclass
class VisemeCue:
    """A single viseme timing cue."""
    start_time: float      # Start time in seconds
    end_time: float        # End time in seconds
    shape: VisemeShape     # The viseme shape
    intensity: float = 1.0 # Optional intensity (0.0 - 1.0)

    @property
    def duration(self) -> float:
        """Get duration of this viseme."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "shape": self.shape.value,
            "intensity": self.intensity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisemeCue":
        """Create from dictionary."""
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            shape=VisemeShape(data["shape"]),
            intensity=data.get("intensity", 1.0),
        )


@dataclass
class VisemeSequence:
    """
    Complete viseme sequence for an utterance.

    This is the standard output format for all viseme providers.
    """
    duration: float                    # Total duration in seconds
    cues: List[VisemeCue] = field(default_factory=list)
    source_text: Optional[str] = None  # Original text (if available)
    provider: str = "unknown"          # Provider that generated this
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_viseme_at_time(self, time_seconds: float) -> Optional[VisemeCue]:
        """Get the active viseme at a specific time."""
        for cue in self.cues:
            if cue.start_time <= time_seconds < cue.end_time:
                return cue
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "duration": self.duration,
            "cues": [cue.to_dict() for cue in self.cues],
            "source_text": self.source_text,
            "provider": self.provider,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisemeSequence":
        """Create from dictionary."""
        return cls(
            duration=data["duration"],
            cues=[VisemeCue.from_dict(c) for c in data.get("cues", [])],
            source_text=data.get("source_text"),
            provider=data.get("provider", "unknown"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def empty(cls, duration: float = 0.0) -> "VisemeSequence":
        """Create an empty sequence (for when viseme extraction fails)."""
        return cls(duration=duration, provider="none")


@dataclass
class TTSCapabilities:
    """
    Capabilities of a TTS engine.

    Used to determine what features are available and how to configure
    the synthesis pipeline.
    """
    # Basic info
    name: str
    version: str = "unknown"

    # Audio capabilities
    supported_formats: List[AudioFormat] = field(
        default_factory=lambda: [AudioFormat.WAV]
    )
    supports_streaming: bool = False
    sample_rates: List[int] = field(default_factory=lambda: [22050])

    # Synthesis capabilities
    supports_ssml: bool = False
    supports_phonemes: bool = False
    max_text_length: Optional[int] = None

    # Viseme capabilities
    provides_visemes: bool = False      # Engine provides viseme timing directly
    viseme_format: Optional[str] = None # Format of visemes (e.g., "azure", "rhubarb")

    # Voice selection
    supports_voice_selection: bool = False
    available_voices: List[str] = field(default_factory=list)

    # Resource requirements
    requires_internet: bool = False
    requires_api_key: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "supported_formats": [f.value for f in self.supported_formats],
            "supports_streaming": self.supports_streaming,
            "sample_rates": self.sample_rates,
            "supports_ssml": self.supports_ssml,
            "supports_phonemes": self.supports_phonemes,
            "max_text_length": self.max_text_length,
            "provides_visemes": self.provides_visemes,
            "viseme_format": self.viseme_format,
            "supports_voice_selection": self.supports_voice_selection,
            "available_voices": self.available_voices,
            "requires_internet": self.requires_internet,
            "requires_api_key": self.requires_api_key,
        }


@dataclass
class TTSResult:
    """
    Result of TTS synthesis.

    Contains either a file path or raw audio data, plus optional
    viseme data if the engine provides it inline.
    """
    # Audio data (one of these will be set)
    audio_file: Optional[Path] = None      # Path to generated audio file
    audio_data: Optional[bytes] = None     # Raw audio bytes (for streaming)

    # Audio properties
    duration: float = 0.0                  # Duration in seconds
    sample_rate: int = 22050               # Sample rate in Hz
    format: AudioFormat = AudioFormat.WAV  # Audio format

    # Source info
    text: str = ""                         # Original text
    engine: str = "unknown"                # Engine that generated this

    # Optional inline visemes (if engine provides them)
    visemes: Optional[VisemeSequence] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_audio_file(self) -> bool:
        """Check if result has an audio file."""
        return self.audio_file is not None and self.audio_file.exists()

    @property
    def has_audio_data(self) -> bool:
        """Check if result has raw audio data."""
        return self.audio_data is not None and len(self.audio_data) > 0

    @property
    def has_visemes(self) -> bool:
        """Check if result includes viseme data."""
        return self.visemes is not None and len(self.visemes.cues) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audio_file": str(self.audio_file) if self.audio_file else None,
            "has_audio_data": self.has_audio_data,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "format": self.format.value,
            "text": self.text,
            "engine": self.engine,
            "has_visemes": self.has_visemes,
            "metadata": self.metadata,
        }
