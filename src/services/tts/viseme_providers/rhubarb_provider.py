"""
Rhubarb Viseme Provider.

Wraps the Rhubarb Lip Sync tool to implement the VisemeProvider protocol.
Rhubarb analyzes audio files and produces accurate viseme timing.

References:
- https://github.com/DanielSWolf/rhubarb-lip-sync
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.core.audio.viseme_provider import (
    VisemeProvider,
    VisemeExtractionError,
    VisemeConfigurationError,
)
from src.core.audio.types import (
    VisemeCue,
    VisemeSequence,
    VisemeShape,
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
VENDOR_RHUBARB_DIR = PROJECT_ROOT / "vendor" / "rhubarb"


# Mapping from Rhubarb visemes to our standard shapes
RHUBARB_TO_STANDARD: Dict[str, VisemeShape] = {
    "A": VisemeShape.BMP,      # Closed mouth - M, B, P
    "B": VisemeShape.LNT,      # Slightly open - consonants
    "C": VisemeShape.AH,       # Open mouth - AH, EH vowels
    "D": VisemeShape.EE,       # Wide mouth - EE, EY vowels
    "E": VisemeShape.OH,       # Rounded mouth - OH, AO vowels
    "F": VisemeShape.OO,       # Puckered mouth - OO, UW vowels
    "G": VisemeShape.FV,       # F/V shape - upper teeth on lower lip
    "H": VisemeShape.AH,       # L shape - tongue visible, wide open
    "X": VisemeShape.SILENCE,  # Idle/silence
}


@dataclass
class RhubarbProviderConfig:
    """Configuration for Rhubarb viseme provider."""
    # Rhubarb executable path (None = auto-detect)
    rhubarb_path: Optional[str] = None

    # Speech recognizer: "pocketSphinx" (English) or "phonetic" (any language)
    recognizer: str = "pocketSphinx"

    # Use extended shape set (G, H, X)
    extended_shapes: bool = True

    # Timeout for Rhubarb process (seconds)
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RhubarbProviderConfig":
        """Create from dictionary."""
        return cls(
            rhubarb_path=config.get("rhubarb_path"),
            recognizer=config.get("recognizer", cls.recognizer),
            extended_shapes=config.get("extended_shapes", cls.extended_shapes),
            timeout=config.get("timeout", cls.timeout),
        )


class RhubarbVisemeProvider(VisemeProvider):
    """
    Rhubarb-based viseme extraction provider.

    Uses Rhubarb Lip Sync to analyze audio files and extract
    accurate viseme timing based on speech recognition.
    """

    def __init__(self, config: Optional[RhubarbProviderConfig] = None):
        """
        Initialize the Rhubarb viseme provider.

        Args:
            config: Rhubarb provider configuration
        """
        self.config = config or RhubarbProviderConfig()
        self._rhubarb_path: Optional[str] = None
        self._cache: Dict[str, VisemeSequence] = {}

    async def initialize(self) -> None:
        """Initialize the provider by finding Rhubarb executable."""
        self._rhubarb_path = self._find_rhubarb()
        if self._rhubarb_path:
            logger.info(f"Rhubarb provider initialized: {self._rhubarb_path}")
        else:
            logger.warning("Rhubarb executable not found")

    def _find_rhubarb(self) -> Optional[str]:
        """Find the Rhubarb executable."""
        # Check explicit path first
        if self.config.rhubarb_path:
            if os.path.isfile(self.config.rhubarb_path):
                return self.config.rhubarb_path
            logger.warning(f"Configured Rhubarb path not found: {self.config.rhubarb_path}")

        # Determine executable name
        exe_name = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"

        # Check vendor directory
        vendor_path = VENDOR_RHUBARB_DIR / exe_name
        if vendor_path.exists() and os.access(str(vendor_path), os.X_OK):
            return str(vendor_path)

        # Check PATH
        rhubarb_in_path = shutil.which("rhubarb")
        if rhubarb_in_path:
            return rhubarb_in_path

        # Check common locations
        search_paths = [
            "/usr/local/bin/rhubarb",
            "/usr/bin/rhubarb",
            os.path.expanduser("~/.local/bin/rhubarb"),
            os.path.expanduser("~/rhubarb/rhubarb"),
            "/opt/rhubarb/rhubarb",
        ]
        for path in search_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    async def extract_visemes(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> VisemeSequence:
        """
        Extract visemes from audio using Rhubarb.

        Args:
            audio_path: Path to the audio file (required)
            text: Optional transcript for improved accuracy
            audio_duration: Ignored (calculated from audio)

        Returns:
            VisemeSequence with timing cues

        Raises:
            VisemeExtractionError: If extraction fails
        """
        if not audio_path:
            raise VisemeExtractionError("Rhubarb requires an audio file")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise VisemeExtractionError(f"Audio file not found: {audio_path}")

        if not self._rhubarb_path:
            await self.initialize()
            if not self._rhubarb_path:
                raise VisemeExtractionError("Rhubarb executable not found")

        # Check cache
        cache_key = f"{audio_path}:{text or ''}"
        if cache_key in self._cache:
            logger.debug(f"Using cached visemes for {audio_path}")
            return self._cache[cache_key]

        # Run Rhubarb in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_rhubarb, str(audio_path), text
        )

        # Cache result
        self._cache[cache_key] = result
        return result

    def _run_rhubarb(self, audio_path: str, text: Optional[str]) -> VisemeSequence:
        """Run Rhubarb subprocess (blocking)."""
        temp_dialog = None
        try:
            # Build command
            cmd = [
                self._rhubarb_path,
                "-f", "json",
                "-r", self.config.recognizer,
            ]

            if self.config.extended_shapes:
                cmd.extend(["--extendedShapes", "GHX"])

            # Add dialog file if text provided
            if text:
                temp_dialog = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt', delete=False
                )
                temp_dialog.write(text)
                temp_dialog.close()
                cmd.extend(["-d", temp_dialog.name])

            cmd.append(audio_path)

            logger.debug(f"Running Rhubarb: {' '.join(cmd)}")

            # Run Rhubarb
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            if result.returncode != 0:
                raise VisemeExtractionError(f"Rhubarb failed: {result.stderr}")

            # Parse output
            data = json.loads(result.stdout)
            return self._parse_output(data, audio_path, text)

        except subprocess.TimeoutExpired:
            raise VisemeExtractionError(
                f"Rhubarb timed out after {self.config.timeout}s"
            )
        except json.JSONDecodeError as e:
            raise VisemeExtractionError(f"Failed to parse Rhubarb output: {e}")
        except Exception as e:
            raise VisemeExtractionError(f"Rhubarb error: {e}")
        finally:
            if temp_dialog:
                try:
                    os.unlink(temp_dialog.name)
                except OSError:
                    pass

    def _parse_output(
        self,
        data: Dict[str, Any],
        audio_path: str,
        text: Optional[str],
    ) -> VisemeSequence:
        """Parse Rhubarb JSON output."""
        cues = []
        mouth_cues = data.get("mouthCues", [])

        for i, cue in enumerate(mouth_cues):
            start = cue.get("start", 0.0)
            viseme_char = cue.get("value", "X")

            # Calculate end time from next cue
            if i + 1 < len(mouth_cues):
                end = mouth_cues[i + 1].get("start", start + 0.1)
            else:
                end = start + 0.1

            # Map to standard shape
            shape = RHUBARB_TO_STANDARD.get(viseme_char, VisemeShape.SILENCE)

            cues.append(VisemeCue(
                start_time=start,
                end_time=end,
                shape=shape,
            ))

        # Get duration
        duration = cues[-1].end_time if cues else 0.0
        metadata_duration = data.get("metadata", {}).get("duration", duration)

        return VisemeSequence(
            duration=metadata_duration,
            cues=cues,
            source_text=text,
            provider=self.name,
            metadata={
                "audio_file": audio_path,
                "recognizer": self.config.recognizer,
                "rhubarb_metadata": data.get("metadata", {}),
            },
        )

    def is_available(self) -> bool:
        """Check if Rhubarb is available."""
        if self._rhubarb_path is None:
            self._rhubarb_path = self._find_rhubarb()

        if not self._rhubarb_path:
            return False

        try:
            result = subprocess.run(
                [self._rhubarb_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "rhubarb"

    @property
    def requires_audio(self) -> bool:
        return True

    @property
    def requires_text(self) -> bool:
        return False  # Text is optional but improves accuracy

    def clear_cache(self) -> None:
        """Clear the viseme cache."""
        self._cache.clear()

    def get_info(self) -> Dict[str, Any]:
        """Get provider info."""
        info = super().get_info()
        info["rhubarb_path"] = self._rhubarb_path
        info["recognizer"] = self.config.recognizer
        return info
