"""
Core types for Speech-to-Text services.

These types are backend-agnostic and used across all STT engines
and wake word detection engines.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TranscriptionState(Enum):
    """State of a transcription."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranscriptionWord:
    """A single word with timing information."""
    word: str
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text (e.g., a sentence or phrase)."""
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    words: List[TranscriptionWord] = field(default_factory=list)
    language: str = "en"
    is_final: bool = True  # False for partial/interim results

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "is_final": self.is_final,
        }


@dataclass
class STTResult:
    """
    Result of speech-to-text transcription.

    Contains the transcribed text plus metadata about the transcription.
    """
    text: str                          # Full transcribed text
    segments: List[TranscriptionSegment] = field(default_factory=list)
    confidence: float = 1.0            # Overall confidence
    language: str = "en"               # Detected/specified language
    duration: float = 0.0              # Audio duration in seconds
    engine: str = "unknown"            # Engine that produced this result
    is_final: bool = True              # False for streaming partial results

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "confidence": self.confidence,
            "language": self.language,
            "duration": self.duration,
            "engine": self.engine,
            "is_final": self.is_final,
            "metadata": self.metadata,
        }

    @classmethod
    def empty(cls) -> "STTResult":
        """Create an empty result."""
        return cls(text="", engine="none")


@dataclass
class WakeWordResult:
    """
    Result of wake word detection.

    Contains information about the detected wake word.
    """
    detected: bool                     # Whether wake word was detected
    wake_word: str = ""                # The wake word that was detected
    model_name: str = ""               # Model that detected it
    confidence: float = 0.0            # Detection confidence
    timestamp: float = 0.0             # When it was detected

    # All scores from detection (for debugging)
    all_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "wake_word": self.wake_word,
            "model_name": self.model_name,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "all_scores": self.all_scores,
        }


@dataclass
class STTCapabilities:
    """Capabilities of an STT engine."""
    name: str
    version: str = "unknown"

    # Recognition capabilities
    supports_streaming: bool = False
    supports_word_timestamps: bool = False
    supports_confidence_scores: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["en"])

    # Audio requirements
    required_sample_rate: Optional[int] = 16000
    supported_sample_rates: List[int] = field(default_factory=lambda: [16000])
    requires_mono: bool = True

    # Resource requirements
    requires_internet: bool = False
    requires_api_key: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supports_streaming": self.supports_streaming,
            "supports_word_timestamps": self.supports_word_timestamps,
            "supports_confidence_scores": self.supports_confidence_scores,
            "supported_languages": self.supported_languages,
            "required_sample_rate": self.required_sample_rate,
            "supported_sample_rates": self.supported_sample_rates,
            "requires_mono": self.requires_mono,
            "requires_internet": self.requires_internet,
            "requires_api_key": self.requires_api_key,
        }


@dataclass
class WakeWordCapabilities:
    """Capabilities of a wake word engine."""
    name: str
    version: str = "unknown"

    # Detection capabilities
    supported_wake_words: List[str] = field(default_factory=list)
    supports_custom_wake_words: bool = False
    supports_multiple_wake_words: bool = True

    # Audio requirements
    required_sample_rate: int = 16000
    chunk_size_samples: int = 1280  # Recommended chunk size

    # Resource requirements
    requires_internet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supported_wake_words": self.supported_wake_words,
            "supports_custom_wake_words": self.supports_custom_wake_words,
            "supports_multiple_wake_words": self.supports_multiple_wake_words,
            "required_sample_rate": self.required_sample_rate,
            "chunk_size_samples": self.chunk_size_samples,
            "requires_internet": self.requires_internet,
        }
