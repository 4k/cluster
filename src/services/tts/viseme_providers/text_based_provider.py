"""
Text-Based Viseme Provider.

Generates viseme timing from text without audio analysis.
This is a zero-latency fallback when Rhubarb is not available
or when immediate animation is needed.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.audio.viseme_provider import (
    VisemeProvider,
    VisemeExtractionError,
    CHAR_TO_VISEME,
)
from src.core.audio.types import (
    VisemeCue,
    VisemeSequence,
    VisemeShape,
)

logger = logging.getLogger(__name__)


# Digraph patterns that map to specific visemes
DIGRAPH_TO_VISEME: Dict[str, VisemeShape] = {
    "th": VisemeShape.TH,
    "ch": VisemeShape.LNT,
    "sh": VisemeShape.LNT,
    "ph": VisemeShape.FV,
    "wh": VisemeShape.OO,
    "oo": VisemeShape.OO,
    "ee": VisemeShape.EE,
    "ea": VisemeShape.EE,
    "ou": VisemeShape.OH,
    "ow": VisemeShape.OH,
    "ai": VisemeShape.EE,
    "ay": VisemeShape.EE,
    "oi": VisemeShape.OH,
    "oy": VisemeShape.OH,
}


@dataclass
class TextBasedProviderConfig:
    """Configuration for text-based viseme provider."""
    # Duration per character in seconds
    char_duration: float = 0.08

    # Duration for word boundaries (spaces)
    space_duration: float = 0.1

    # Minimum viseme duration
    min_viseme_duration: float = 0.04

    # Enable digraph processing (th, ch, sh, etc.)
    process_digraphs: bool = True

    # Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
    speed_factor: float = 1.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TextBasedProviderConfig":
        """Create from dictionary."""
        return cls(
            char_duration=config.get("char_duration", cls.char_duration),
            space_duration=config.get("space_duration", cls.space_duration),
            min_viseme_duration=config.get("min_viseme_duration", cls.min_viseme_duration),
            process_digraphs=config.get("process_digraphs", cls.process_digraphs),
            speed_factor=config.get("speed_factor", cls.speed_factor),
        )


class TextBasedVisemeProvider(VisemeProvider):
    """
    Text-based viseme provider.

    Generates viseme timing directly from text using character-to-viseme
    mapping. This provides zero-latency lip sync without audio analysis.

    Pros:
    - No latency (can start immediately with TTS)
    - No external dependencies
    - Works with any TTS engine

    Cons:
    - Less accurate than audio-based analysis
    - Timing is estimated, not synchronized to actual speech
    """

    def __init__(self, config: Optional[TextBasedProviderConfig] = None):
        """
        Initialize the text-based viseme provider.

        Args:
            config: Provider configuration
        """
        self.config = config or TextBasedProviderConfig()

    async def extract_visemes(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> VisemeSequence:
        """
        Generate visemes from text.

        Args:
            audio_path: Ignored (not needed for text-based)
            text: The text to generate visemes for (required)
            audio_duration: Optional known duration (for timing adjustment)

        Returns:
            VisemeSequence with timing cues

        Raises:
            VisemeExtractionError: If text is not provided
        """
        if not text:
            raise VisemeExtractionError("Text-based provider requires text input")

        # Process text into viseme units
        units = self._text_to_units(text.lower())

        # Calculate base duration
        base_duration = self._calculate_duration(units)

        # If audio duration is known, scale timing to match
        if audio_duration and audio_duration > 0:
            scale_factor = audio_duration / base_duration if base_duration > 0 else 1.0
        else:
            scale_factor = 1.0

        # Generate cues
        cues = self._generate_cues(units, scale_factor)

        # Calculate final duration
        duration = cues[-1].end_time if cues else 0.0
        if audio_duration:
            duration = audio_duration

        return VisemeSequence(
            duration=duration,
            cues=cues,
            source_text=text,
            provider=self.name,
            metadata={
                "char_count": len(text),
                "unit_count": len(units),
                "scale_factor": scale_factor,
            },
        )

    def _text_to_units(self, text: str) -> List[Tuple[str, VisemeShape, float]]:
        """
        Convert text to viseme units.

        Returns list of (text, shape, duration) tuples.
        """
        units = []
        i = 0
        text = text.lower()

        while i < len(text):
            # Check for digraphs first
            if self.config.process_digraphs and i + 1 < len(text):
                digraph = text[i:i+2]
                if digraph in DIGRAPH_TO_VISEME:
                    units.append((
                        digraph,
                        DIGRAPH_TO_VISEME[digraph],
                        self.config.char_duration * 1.5  # Digraphs slightly longer
                    ))
                    i += 2
                    continue

            char = text[i]

            # Handle spaces and punctuation
            if char == ' ':
                units.append((char, VisemeShape.SILENCE, self.config.space_duration))
            elif char in '.,!?;:':
                # Punctuation = longer pause
                units.append((char, VisemeShape.SILENCE, self.config.space_duration * 1.5))
            elif char in '\n\r\t':
                units.append((char, VisemeShape.SILENCE, self.config.space_duration))
            elif char.isalpha():
                shape = CHAR_TO_VISEME.get(char, VisemeShape.AH)
                units.append((char, shape, self.config.char_duration))
            else:
                # Other characters (numbers, symbols) - brief silence
                units.append((char, VisemeShape.SILENCE, self.config.char_duration * 0.5))

            i += 1

        return units

    def _calculate_duration(
        self,
        units: List[Tuple[str, VisemeShape, float]]
    ) -> float:
        """Calculate total duration from units."""
        total = sum(u[2] for u in units)
        return total / self.config.speed_factor

    def _generate_cues(
        self,
        units: List[Tuple[str, VisemeShape, float]],
        scale_factor: float = 1.0,
    ) -> List[VisemeCue]:
        """Generate viseme cues from units."""
        cues = []
        current_time = 0.0

        # Merge consecutive same-shape cues
        merged_units = []
        for text, shape, duration in units:
            adjusted_duration = (duration / self.config.speed_factor) * scale_factor
            adjusted_duration = max(adjusted_duration, self.config.min_viseme_duration)

            if merged_units and merged_units[-1][1] == shape:
                # Merge with previous
                prev_text, prev_shape, prev_duration = merged_units[-1]
                merged_units[-1] = (prev_text + text, prev_shape, prev_duration + adjusted_duration)
            else:
                merged_units.append((text, shape, adjusted_duration))

        # Generate cues
        for text, shape, duration in merged_units:
            cues.append(VisemeCue(
                start_time=current_time,
                end_time=current_time + duration,
                shape=shape,
            ))
            current_time += duration

        return cues

    def is_available(self) -> bool:
        """Always available (no external dependencies)."""
        return True

    @property
    def name(self) -> str:
        return "text_based"

    @property
    def requires_audio(self) -> bool:
        return False

    @property
    def requires_text(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        """Get provider info."""
        info = super().get_info()
        info["char_duration"] = self.config.char_duration
        info["speed_factor"] = self.config.speed_factor
        return info


# Convenience function for quick viseme generation
async def generate_text_visemes(
    text: str,
    duration: Optional[float] = None,
    speed_factor: float = 1.0,
) -> VisemeSequence:
    """
    Quick viseme generation from text.

    Args:
        text: Text to generate visemes for
        duration: Optional known audio duration for timing adjustment
        speed_factor: Speed multiplier (1.0 = normal)

    Returns:
        VisemeSequence
    """
    config = TextBasedProviderConfig(speed_factor=speed_factor)
    provider = TextBasedVisemeProvider(config)
    return await provider.extract_visemes(text=text, audio_duration=duration)
