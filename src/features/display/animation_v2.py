"""
Animation Service V2 - Backend-Agnostic Version.

Coordinates facial animations with TTS using the new VisemeProvider
abstraction. Works with any viseme provider (Rhubarb, text-based, etc.).

Key changes from V1:
- Uses VisemeProvider protocol instead of direct Rhubarb calls
- Works with VisemeSequence from core.audio.types
- Supports provider switching at runtime
- Better fallback handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.audio.viseme_provider import VisemeProvider, NoneVisemeProvider
from src.core.audio.types import VisemeSequence, VisemeShape

logger = logging.getLogger(__name__)


# Mapping from new VisemeShape to legacy internal viseme names
# This maintains backward compatibility with existing renderers
SHAPE_TO_LEGACY: Dict[VisemeShape, str] = {
    VisemeShape.SILENCE: "SILENCE",
    VisemeShape.BMP: "BMP",
    VisemeShape.FV: "FV",
    VisemeShape.TH: "TH",
    VisemeShape.LNT: "LNT",
    VisemeShape.AH: "AH",
    VisemeShape.EE: "EE",
    VisemeShape.OH: "OH",
    VisemeShape.OO: "OO",
    VisemeShape.WQ: "WQ",
    VisemeShape.REST: "REST",
}


@dataclass
class LipSyncSession:
    """Active lip sync playback session."""
    session_id: str
    viseme_data: VisemeSequence
    start_time: float  # When playback started
    is_active: bool = True
    correlation_id: Optional[str] = None

    def get_elapsed_time(self) -> float:
        """Get elapsed time since playback started."""
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.get_elapsed_time() >= self.viseme_data.duration


class AnimationServiceV2:
    """
    Backend-agnostic animation service.

    Uses VisemeProvider abstraction for lip sync generation,
    allowing easy switching between providers.
    """

    def __init__(
        self,
        viseme_provider: Optional[VisemeProvider] = None,
        fallback_provider: Optional[VisemeProvider] = None,
        viseme_emit_interval: float = 0.033,  # ~30 FPS
    ):
        """
        Initialize the animation service.

        Args:
            viseme_provider: Primary viseme provider
            fallback_provider: Fallback if primary fails
            viseme_emit_interval: How often to emit viseme events
        """
        self.viseme_provider = viseme_provider or NoneVisemeProvider()
        self.fallback_provider = fallback_provider

        self.viseme_emit_interval = viseme_emit_interval

        # Event bus
        self._event_bus = None
        self._is_running = False

        # Active sessions
        self._active_sessions: Dict[str, LipSyncSession] = {}
        self._viseme_task: Optional[asyncio.Task] = None

        # Cached viseme data (correlation_id -> VisemeSequence)
        self._cached_visemes: Dict[str, VisemeSequence] = {}

        # Statistics
        self.stats = {
            "sessions_started": 0,
            "sessions_completed": 0,
            "provider_successes": 0,
            "provider_failures": 0,
            "fallback_used": 0,
            "visemes_emitted": 0,
        }

        logger.info(f"AnimationServiceV2 initialized: provider={self.viseme_provider.name}")

    async def initialize(self) -> bool:
        """Initialize the service."""
        try:
            # Initialize providers
            await self.viseme_provider.initialize()
            if self.fallback_provider:
                await self.fallback_provider.initialize()

            # Connect to event bus
            from src.core.event_bus import EventBus, EventType

            self._event_bus = await EventBus.get_instance()

            # Subscribe to events
            self._event_bus.subscribe(
                EventType.TTS_STARTED,
                self._on_tts_started,
                priority=5,
            )
            self._event_bus.subscribe(
                EventType.AUDIO_PLAYBACK_STARTED,
                self._on_audio_playback_started,
                priority=5,
            )
            self._event_bus.subscribe(
                EventType.AUDIO_PLAYBACK_ENDED,
                self._on_audio_playback_ended,
                priority=5,
            )
            self._event_bus.subscribe(
                EventType.TTS_COMPLETED,
                self._on_tts_completed,
                priority=5,
            )
            self._event_bus.subscribe(
                EventType.SYSTEM_STOPPED,
                self._on_system_stopped,
            )

            self._is_running = True

            # Start viseme emission loop
            self._viseme_task = asyncio.create_task(self._viseme_emission_loop())

            logger.info(
                f"AnimationServiceV2 initialized "
                f"(provider: {self.viseme_provider.name}, "
                f"available: {self.viseme_provider.is_available()})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AnimationServiceV2: {e}", exc_info=True)
            return False

    async def stop(self) -> None:
        """Stop the service."""
        self._is_running = False

        if self._viseme_task:
            self._viseme_task.cancel()
            try:
                await self._viseme_task
            except asyncio.CancelledError:
                pass

        self._active_sessions.clear()
        self._cached_visemes.clear()

        # Shutdown providers
        await self.viseme_provider.shutdown()
        if self.fallback_provider:
            await self.fallback_provider.shutdown()

        logger.info("AnimationServiceV2 stopped")

    async def _on_tts_started(self, event) -> None:
        """Handle TTS started - generate visemes."""
        text = event.data.get("text", "")
        audio_file = event.data.get("audio_file")
        duration = event.data.get("duration", 0.0)
        correlation_id = event.correlation_id

        if not text:
            return

        logger.info(f"TTS started: generating visemes for '{text[:50]}...'")

        # Generate visemes
        visemes = await self._generate_visemes(
            audio_file=audio_file,
            text=text,
            duration=duration,
        )

        if visemes and correlation_id:
            # Cache for when playback starts
            self._cached_visemes[correlation_id] = visemes
            self.stats["provider_successes"] += 1

            # Emit LIP_SYNC_READY
            from src.core.event_bus import EventType, emit_event
            await emit_event(
                EventType.LIP_SYNC_READY,
                {
                    "correlation_id": correlation_id,
                    "duration": visemes.duration,
                    "cue_count": len(visemes.cues),
                    "provider": visemes.provider,
                    "audio_file": audio_file,
                },
                source="animation_service",
                correlation_id=correlation_id,
            )
        else:
            self.stats["provider_failures"] += 1
            # Still emit LIP_SYNC_READY so TTS doesn't wait forever
            from src.core.event_bus import EventType, emit_event
            await emit_event(
                EventType.LIP_SYNC_READY,
                {
                    "correlation_id": correlation_id,
                    "fallback": True,
                },
                source="animation_service",
                correlation_id=correlation_id,
            )

    async def _on_audio_playback_started(self, event) -> None:
        """Handle audio playback started - start viseme session."""
        correlation_id = event.correlation_id
        start_timestamp = event.data.get("start_timestamp")

        if correlation_id and correlation_id in self._cached_visemes:
            visemes = self._cached_visemes.pop(correlation_id)
            await self._start_session(visemes, correlation_id, start_timestamp)
        else:
            logger.warning(f"No cached visemes for {correlation_id}")

    async def _on_audio_playback_ended(self, event) -> None:
        """Handle audio playback ended."""
        await self._stop_session_by_correlation(event.correlation_id)

    async def _on_tts_completed(self, event) -> None:
        """Handle TTS completed."""
        await self._stop_session_by_correlation(event.correlation_id)
        await self._emit_viseme(VisemeShape.SILENCE, event.correlation_id)

    async def _on_system_stopped(self, event) -> None:
        """Handle system stopped."""
        await self.stop()

    async def _generate_visemes(
        self,
        audio_file: Optional[str],
        text: str,
        duration: float,
    ) -> Optional[VisemeSequence]:
        """Generate visemes using provider."""
        try:
            return await self.viseme_provider.extract_visemes(
                audio_path=audio_file,
                text=text,
                audio_duration=duration,
            )
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")

            # Try fallback
            if self.fallback_provider:
                try:
                    self.stats["fallback_used"] += 1
                    return await self.fallback_provider.extract_visemes(
                        audio_path=audio_file,
                        text=text,
                        audio_duration=duration,
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            return None

    async def _start_session(
        self,
        visemes: VisemeSequence,
        correlation_id: Optional[str],
        start_timestamp: Optional[float] = None,
    ) -> None:
        """Start a viseme playback session."""
        session_id = f"lipsync_{time.time()}"
        session_start = start_timestamp or time.time()

        session = LipSyncSession(
            session_id=session_id,
            viseme_data=visemes,
            start_time=session_start,
            correlation_id=correlation_id,
        )

        self._active_sessions[session_id] = session
        self.stats["sessions_started"] += 1

        logger.info(f"Lip sync session started: {session_id}")

        # Emit event
        from src.core.event_bus import EventType, emit_event
        await emit_event(
            EventType.LIP_SYNC_STARTED,
            {
                "session_id": session_id,
                "duration": visemes.duration,
                "cue_count": len(visemes.cues),
                "provider": visemes.provider,
            },
            source="animation_service",
            correlation_id=correlation_id,
        )

    async def _stop_session_by_correlation(self, correlation_id: Optional[str]) -> None:
        """Stop sessions matching a correlation ID."""
        if not correlation_id:
            return

        to_remove = []
        for session_id, session in self._active_sessions.items():
            if session.correlation_id == correlation_id:
                session.is_active = False
                to_remove.append(session_id)

        for session_id in to_remove:
            session = self._active_sessions.pop(session_id, None)
            if session:
                self.stats["sessions_completed"] += 1

                from src.core.event_bus import EventType, emit_event
                await emit_event(
                    EventType.LIP_SYNC_COMPLETED,
                    {
                        "session_id": session_id,
                        "duration": session.viseme_data.duration,
                        "elapsed": session.get_elapsed_time(),
                    },
                    source="animation_service",
                    correlation_id=correlation_id,
                )

    async def _viseme_emission_loop(self) -> None:
        """Main loop for emitting viseme events."""
        logger.info("Viseme emission loop started")

        while self._is_running:
            try:
                await asyncio.sleep(self.viseme_emit_interval)

                completed = []

                for session_id, session in self._active_sessions.items():
                    if not session.is_active:
                        completed.append(session_id)
                        continue

                    elapsed = session.get_elapsed_time()
                    cue = session.viseme_data.get_viseme_at_time(elapsed)

                    if cue:
                        await self._emit_viseme(cue.shape, session.correlation_id)

                    if session.is_complete():
                        session.is_active = False
                        completed.append(session_id)

                for session_id in completed:
                    if session_id in self._active_sessions:
                        del self._active_sessions[session_id]
                        self.stats["sessions_completed"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in viseme loop: {e}")

        logger.info("Viseme emission loop stopped")

    async def _emit_viseme(
        self,
        shape: VisemeShape,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit a viseme update event."""
        if not self._event_bus:
            return

        # Convert to legacy name for backward compatibility
        legacy_name = SHAPE_TO_LEGACY.get(shape, "SILENCE")

        from src.core.event_bus import EventType, emit_event
        await emit_event(
            EventType.MOUTH_SHAPE_UPDATE,
            {"viseme": legacy_name, "shape": shape.value},
            source="animation_service",
            correlation_id=correlation_id,
        )
        self.stats["visemes_emitted"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "active_sessions": len(self._active_sessions),
            "provider": self.viseme_provider.name,
            "provider_available": self.viseme_provider.is_available(),
            "is_running": self._is_running,
        }

    def set_provider(self, provider: VisemeProvider) -> None:
        """
        Switch viseme provider at runtime.

        Args:
            provider: New viseme provider to use
        """
        old_name = self.viseme_provider.name
        self.viseme_provider = provider
        logger.info(f"Switched viseme provider: {old_name} -> {provider.name}")
