#!/usr/bin/env python3
"""
Rhubarb Lip Sync Service.

Integrates Rhubarb Lip Sync tool for precise mouth animation timing.
Rhubarb analyzes audio files and produces viseme timings with timestamps.

Input: Audio file (WAV) + optional text transcript
Output: Viseme sequence with precise timestamps

References:
- https://github.com/DanielSWolf/rhubarb-lip-sync
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project paths - lip_sync_service.py is at src/services/lip_sync_service.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
VENDOR_RHUBARB_DIR = PROJECT_ROOT / "vendor" / "rhubarb"


class RhubarbViseme(Enum):
    """
    Rhubarb's viseme set based on Preston Blair's phoneme groups.

    These are the standard mouth shapes used in traditional animation.
    """
    A = "A"  # Closed mouth (silence, M, B, P)
    B = "B"  # Slightly open mouth (most consonants like K, S, T)
    C = "C"  # Open mouth (vowels like EH, AE, AH)
    D = "D"  # Wide mouth (vowels like EY, EE, Y)
    E = "E"  # Rounded mouth (vowels like AO, OW)
    F = "F"  # Puckered mouth (vowels like UW, OO, W)
    G = "G"  # F/V shape (F, V - upper teeth on lower lip)
    H = "H"  # L shape (L - tongue visible, wide open)
    X = "X"  # Idle/Silence (rest position between words)


@dataclass
class VisemeCue:
    """A single viseme timing cue from Rhubarb."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    viseme: RhubarbViseme  # The viseme shape

    @property
    def duration(self) -> float:
        """Get duration of this viseme."""
        return self.end_time - self.start_time


@dataclass
class LipSyncData:
    """Complete lip sync data for an utterance."""
    audio_file: str
    text: Optional[str]
    duration: float
    cues: List[VisemeCue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_viseme_at_time(self, time_seconds: float) -> Optional[VisemeCue]:
        """Get the active viseme at a specific time."""
        for cue in self.cues:
            if cue.start_time <= time_seconds < cue.end_time:
                return cue
        return None


# Mapping from Rhubarb visemes to our internal viseme system
RHUBARB_TO_INTERNAL_VISEME: Dict[RhubarbViseme, str] = {
    RhubarbViseme.A: "BMP",      # Closed mouth - B, M, P
    RhubarbViseme.B: "LNT",      # Slightly open - consonants like K, S, T, N
    RhubarbViseme.C: "AH",       # Open mouth - AH, EH vowels
    RhubarbViseme.D: "EE",       # Wide mouth - EE, EY vowels
    RhubarbViseme.E: "OH",       # Rounded mouth - OH, AO vowels
    RhubarbViseme.F: "OO",       # Puckered mouth - OO, UW vowels
    RhubarbViseme.G: "FV",       # F/V shape - upper teeth on lower lip
    RhubarbViseme.H: "AH",       # L shape - tongue visible, open
    RhubarbViseme.X: "SILENCE",  # Idle/rest position
}


class RhubarbLipSyncService:
    """
    Service for generating lip sync data using Rhubarb Lip Sync.

    Rhubarb is an external tool that analyzes audio and produces
    viseme timings with precise timestamps.
    """

    def __init__(
        self,
        rhubarb_path: Optional[str] = None,
        recognizer: str = "pocketSphinx",
        extended_shapes: bool = True
    ):
        """
        Initialize the Rhubarb Lip Sync service.

        Args:
            rhubarb_path: Path to rhubarb executable. If None, searches PATH.
            recognizer: Speech recognizer to use ('pocketSphinx' or 'phonetic')
            extended_shapes: Whether to use extended shape set (G, H, X)
        """
        self.rhubarb_path = rhubarb_path or self._find_rhubarb()
        self.recognizer = recognizer
        self.extended_shapes = extended_shapes

        # Event bus connection
        self.event_bus = None
        self.is_running = False

        # Cache for lip sync data
        self._cache: Dict[str, LipSyncData] = {}

        logger.info(f"RhubarbLipSyncService initialized (rhubarb: {self.rhubarb_path})")

    def _find_rhubarb(self) -> str:
        """Find the rhubarb executable, checking vendor directory first."""
        import shutil

        # Determine executable name based on platform
        executable_name = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"

        # 1. Check vendor directory first (bundled with application)
        vendor_path = VENDOR_RHUBARB_DIR / executable_name
        if vendor_path.exists() and os.access(str(vendor_path), os.X_OK):
            logger.info(f"Using bundled Rhubarb: {vendor_path}")
            return str(vendor_path)

        # 2. Check if rhubarb is in PATH
        rhubarb_in_path = shutil.which("rhubarb")
        if rhubarb_in_path:
            logger.info(f"Using system Rhubarb: {rhubarb_in_path}")
            return rhubarb_in_path

        # 3. Check common installation locations
        search_paths = [
            "/usr/local/bin/rhubarb",
            "/usr/bin/rhubarb",
            os.path.expanduser("~/.local/bin/rhubarb"),
            os.path.expanduser("~/rhubarb/rhubarb"),
            "/opt/rhubarb/rhubarb",
        ]

        for path in search_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Using Rhubarb at: {path}")
                return path

        # Not found - provide installation instructions
        logger.warning(
            "Rhubarb executable not found. Run 'python scripts/setup_rhubarb.py' "
            "to install it, or download from https://github.com/DanielSWolf/rhubarb-lip-sync"
        )
        return str(vendor_path)  # Return expected path even if not found

    def is_available(self) -> bool:
        """Check if Rhubarb is available and working."""
        try:
            result = subprocess.run(
                [self.rhubarb_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Rhubarb version: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Rhubarb not available: {e}")
        return False

    async def initialize(self) -> None:
        """Initialize event bus connection."""
        try:
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()
            self.is_running = True

            logger.info("RhubarbLipSyncService connected to event bus")

        except ImportError:
            logger.warning("Event bus not available")
        except Exception as e:
            logger.error(f"Failed to initialize event bus: {e}")

    def generate_lip_sync(
        self,
        audio_file: str,
        text: Optional[str] = None,
        dialog_file: Optional[str] = None
    ) -> Optional[LipSyncData]:
        """
        Generate lip sync data from an audio file.

        Args:
            audio_file: Path to the audio file (WAV format recommended)
            text: Optional transcript text for improved accuracy
            dialog_file: Optional path to dialog file (alternative to text)

        Returns:
            LipSyncData with viseme cues, or None if failed
        """
        if not os.path.isfile(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        # Check cache
        cache_key = f"{audio_file}:{text or ''}"
        if cache_key in self._cache:
            logger.debug(f"Using cached lip sync data for {audio_file}")
            return self._cache[cache_key]

        try:
            # Build rhubarb command
            cmd = [
                self.rhubarb_path,
                "-f", "json",  # JSON output format
                "-r", self.recognizer,  # Speech recognizer
            ]

            # Add extended shapes flag
            if self.extended_shapes:
                cmd.extend(["--extendedShapes", "GHX"])

            # Add dialog file if provided
            temp_dialog = None
            if dialog_file and os.path.isfile(dialog_file):
                cmd.extend(["-d", dialog_file])
            elif text:
                # Create temporary dialog file
                temp_dialog = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt', delete=False
                )
                temp_dialog.write(text)
                temp_dialog.close()
                cmd.extend(["-d", temp_dialog.name])

            # Add audio file
            cmd.append(audio_file)

            logger.info(f"Running Rhubarb: {' '.join(cmd)}")

            # Run rhubarb
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Clean up temp file
            if temp_dialog:
                os.unlink(temp_dialog.name)

            if result.returncode != 0:
                logger.error(f"Rhubarb failed: {result.stderr}")
                return None

            # Parse JSON output
            data = json.loads(result.stdout)

            # Convert to LipSyncData
            lip_sync = self._parse_rhubarb_output(data, audio_file, text)

            # Cache the result
            self._cache[cache_key] = lip_sync

            logger.info(
                f"Generated lip sync: {len(lip_sync.cues)} cues, "
                f"{lip_sync.duration:.2f}s duration"
            )

            return lip_sync

        except subprocess.TimeoutExpired:
            logger.error("Rhubarb timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Rhubarb output: {e}")
        except Exception as e:
            logger.error(f"Error generating lip sync: {e}", exc_info=True)

        return None

    async def generate_lip_sync_async(
        self,
        audio_file: str,
        text: Optional[str] = None,
        dialog_file: Optional[str] = None
    ) -> Optional[LipSyncData]:
        """Async wrapper for generate_lip_sync."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_lip_sync, audio_file, text, dialog_file
        )

    def _parse_rhubarb_output(
        self,
        data: Dict[str, Any],
        audio_file: str,
        text: Optional[str]
    ) -> LipSyncData:
        """Parse Rhubarb JSON output into LipSyncData."""
        cues = []

        mouth_cues = data.get("mouthCues", [])

        for i, cue in enumerate(mouth_cues):
            start = cue.get("start", 0.0)
            viseme_char = cue.get("value", "X")

            # Calculate end time from next cue or use a default
            if i + 1 < len(mouth_cues):
                end = mouth_cues[i + 1].get("start", start + 0.1)
            else:
                end = start + 0.1

            try:
                viseme = RhubarbViseme(viseme_char)
            except ValueError:
                viseme = RhubarbViseme.X

            cues.append(VisemeCue(
                start_time=start,
                end_time=end,
                viseme=viseme
            ))

        # Get total duration
        duration = 0.0
        if cues:
            duration = cues[-1].end_time

        # Get metadata
        metadata = {
            "sound_file": data.get("metadata", {}).get("soundFile", audio_file),
            "duration": data.get("metadata", {}).get("duration", duration),
        }

        return LipSyncData(
            audio_file=audio_file,
            text=text,
            duration=duration,
            cues=cues,
            metadata=metadata
        )

    def convert_to_internal_viseme(self, rhubarb_viseme: RhubarbViseme) -> str:
        """Convert a Rhubarb viseme to our internal viseme name."""
        return RHUBARB_TO_INTERNAL_VISEME.get(rhubarb_viseme, "SILENCE")

    def get_cue_schedule(
        self,
        lip_sync_data: LipSyncData,
        start_offset: float = 0.0
    ) -> List[Tuple[float, str]]:
        """
        Get a schedule of (timestamp, viseme_name) tuples for playback.

        Args:
            lip_sync_data: The lip sync data
            start_offset: Time offset to add to all timestamps

        Returns:
            List of (absolute_time, internal_viseme_name) tuples
        """
        schedule = []

        for cue in lip_sync_data.cues:
            internal_viseme = self.convert_to_internal_viseme(cue.viseme)
            timestamp = cue.start_time + start_offset
            schedule.append((timestamp, internal_viseme))

        return schedule

    def clear_cache(self) -> None:
        """Clear the lip sync cache."""
        self._cache.clear()
        logger.debug("Lip sync cache cleared")


def ensure_rhubarb_installed(auto_install: bool = False) -> bool:
    """
    Ensure Rhubarb is installed, optionally auto-installing if not found.

    Args:
        auto_install: If True, automatically download and install Rhubarb

    Returns:
        True if Rhubarb is available
    """
    service = RhubarbLipSyncService()
    if service.is_available():
        return True

    if auto_install:
        print("Rhubarb not found. Installing...")
        try:
            from scripts.setup_rhubarb import setup_rhubarb, verify_installation
            setup_rhubarb()
            return verify_installation()
        except Exception as e:
            print(f"Auto-install failed: {e}")
            return False

    return False


# Standalone testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check for --install flag
    if "--install" in sys.argv:
        sys.argv.remove("--install")
        print("Installing Rhubarb Lip Sync...")
        try:
            from scripts.setup_rhubarb import setup_rhubarb, verify_installation
            setup_rhubarb()
            if verify_installation():
                print("\nRhubarb installed successfully!")
            else:
                print("\nInstallation may have issues.")
                sys.exit(1)
        except Exception as e:
            print(f"Installation failed: {e}")
            sys.exit(1)

        if len(sys.argv) == 1:
            sys.exit(0)

    service = RhubarbLipSyncService()

    if not service.is_available():
        print("Rhubarb is not available.")
        print("\nTo install, run one of:")
        print("  python rhubarb_lip_sync_service.py --install")
        print("  python scripts/setup_rhubarb.py")
        print("\nOr download manually from:")
        print("  https://github.com/DanielSWolf/rhubarb-lip-sync/releases")
        sys.exit(1)

    # Test with a sample audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        text = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"\nAnalyzing: {audio_file}")
        if text:
            print(f"Text: {text}")

        result = service.generate_lip_sync(audio_file, text)

        if result:
            print(f"\nDuration: {result.duration:.2f}s")
            print(f"Cues: {len(result.cues)}")
            print("\nViseme timeline:")
            for cue in result.cues[:20]:  # Show first 20
                internal = service.convert_to_internal_viseme(cue.viseme)
                print(f"  {cue.start_time:6.3f}s - {cue.end_time:6.3f}s: "
                      f"{cue.viseme.value} -> {internal}")
            if len(result.cues) > 20:
                print(f"  ... and {len(result.cues) - 20} more cues")
        else:
            print("Failed to generate lip sync data")
    else:
        print("Rhubarb Lip Sync Service")
        print("Usage: python rhubarb_lip_sync_service.py <audio.wav> [text]")
