"""
Simple TTS Manager - Text-to-Speech interface.

This is a stub implementation that provides the interface
expected by main.py. A full implementation would integrate
with Piper TTS or another TTS engine.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from core.event_bus import EventBus, EventType
from core.types import TTSConfig

logger = logging.getLogger(__name__)


class SimpleTTSManager:
    """Manages text-to-speech synthesis.

    This class provides:
    - Text to audio synthesis
    - Phoneme/viseme events for lip-sync
    - Integration with the event bus
    """

    def __init__(self, config: TTSConfig):
        """Initialize the TTS manager.

        Args:
            config: TTS configuration
        """
        self.config = config
        self.event_bus: Optional[EventBus] = None
        self._initialized = False
        self._mock_mode = config.engine_type == "mock"

    async def initialize(self) -> None:
        """Initialize TTS system."""
        logger.info(f"Initializing SimpleTTSManager (engine: {self.config.engine_type})")

        self.event_bus = await EventBus.get_instance()

        if self._mock_mode:
            logger.info("TTS running in mock mode")
        else:
            # TODO: Initialize actual TTS engine (Piper, etc.)
            logger.info("TTS engine initialization placeholder")

        self._initialized = True
        logger.info("SimpleTTSManager initialized")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes, or None if failed
        """
        if not self._initialized:
            logger.error("TTS not initialized")
            return None

        logger.info(f"Synthesizing: '{text[:50]}...' " if len(text) > 50 else f"Synthesizing: '{text}'")

        # Emit TTS started event
        await self.event_bus.emit(
            EventType.TTS_STARTED,
            {"text": text},
            source="tts_manager"
        )

        if self._mock_mode:
            # Mock synthesis - just delay
            await asyncio.sleep(len(text) * 0.05)  # ~50ms per character
        else:
            # TODO: Actual synthesis
            await asyncio.sleep(0.5)

        # Emit TTS completed event
        await self.event_bus.emit(
            EventType.TTS_COMPLETED,
            {"text": text},
            source="tts_manager"
        )

        return b""  # Placeholder

    async def speak(self, text: str) -> None:
        """Synthesize and play text.

        Args:
            text: Text to speak
        """
        audio = await self.synthesize(text)
        if audio:
            # TODO: Play audio
            pass

    async def cleanup(self) -> None:
        """Clean up TTS resources."""
        logger.info("SimpleTTSManager cleaned up")
        self._initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get current TTS state.

        Returns:
            Dictionary with state info
        """
        return {
            "initialized": self._initialized,
            "engine_type": self.config.engine_type,
            "mock_mode": self._mock_mode,
        }
