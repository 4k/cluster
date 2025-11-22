#!/usr/bin/env python3
"""
Animation Service for coordinating facial animations.

This service watches the event bus and coordinates lip sync, expressions,
and other animations with the display system.

Key responsibilities:
1. Monitor TTS events and generate lip sync data using Rhubarb
2. Schedule and emit viseme events synchronized with audio playback
3. Coordinate with the display manager for smooth animations
"""

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Local imports
from .lip_sync import (
    RhubarbLipSyncService,
    LipSyncData,
    VisemeCue,
    RhubarbViseme,
    RHUBARB_TO_INTERNAL_VISEME
)

logger = logging.getLogger(__name__)


@dataclass
class LipSyncSession:
    """Active lip sync playback session."""
    session_id: str
    lip_sync_data: LipSyncData
    start_time: float  # When playback started
    is_active: bool = True
    correlation_id: Optional[str] = None

    def get_elapsed_time(self) -> float:
        """Get elapsed time since playback started."""
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        """Check if the lip sync session is complete."""
        return self.get_elapsed_time() >= self.lip_sync_data.duration


class AnimationService:
    """
    Service that coordinates facial animations with the event bus.

    Watches for TTS events and generates synchronized lip sync animations
    using Rhubarb Lip Sync.
    """

    def __init__(
        self,
        rhubarb_path: Optional[str] = None,
        temp_dir: Optional[str] = None,
        viseme_emit_interval: float = 0.033,  # ~30 FPS
        enable_fallback: bool = True
    ):
        """
        Initialize the animation service.

        Args:
            rhubarb_path: Path to rhubarb executable
            temp_dir: Directory for temporary audio files
            viseme_emit_interval: How often to emit viseme events (seconds)
            enable_fallback: Fall back to character-based lip sync if Rhubarb fails
        """
        # Rhubarb lip sync service
        self.lip_sync_service = RhubarbLipSyncService(rhubarb_path=rhubarb_path)
        self.rhubarb_available = False

        # Temp directory for audio files
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Configuration
        self.viseme_emit_interval = viseme_emit_interval
        self.enable_fallback = enable_fallback

        # Event bus
        self.event_bus = None
        self.is_running = False

        # Active sessions
        self._active_sessions: Dict[str, LipSyncSession] = {}
        self._viseme_task: Optional[asyncio.Task] = None

        # Pending lip sync requests (audio file -> text mapping)
        self._pending_requests: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "sessions_started": 0,
            "sessions_completed": 0,
            "rhubarb_successes": 0,
            "rhubarb_failures": 0,
            "fallback_used": 0,
            "visemes_emitted": 0,
        }

        logger.info("AnimationService initialized")

    async def initialize(self) -> bool:
        """
        Initialize the animation service.

        Returns:
            True if initialized successfully
        """
        try:
            # Check if Rhubarb is available
            self.rhubarb_available = self.lip_sync_service.is_available()
            if not self.rhubarb_available:
                logger.warning(
                    "Rhubarb not available. Lip sync will use fallback method. "
                    "Install from: https://github.com/DanielSWolf/rhubarb-lip-sync"
                )

            # Connect to event bus
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()

            # Subscribe to TTS events
            self.event_bus.subscribe(
                EventType.TTS_STARTED,
                self._on_tts_started,
                priority=5  # Higher priority for animation
            )
            self.event_bus.subscribe(
                EventType.AUDIO_PLAYBACK_STARTED,
                self._on_audio_playback_started,
                priority=5
            )
            self.event_bus.subscribe(
                EventType.AUDIO_PLAYBACK_ENDED,
                self._on_audio_playback_ended,
                priority=5
            )
            self.event_bus.subscribe(
                EventType.TTS_COMPLETED,
                self._on_tts_completed,
                priority=5
            )
            self.event_bus.subscribe(
                EventType.SYSTEM_STOPPED,
                self._on_system_stopped
            )

            self.is_running = True

            # Start viseme emission loop
            self._viseme_task = asyncio.create_task(self._viseme_emission_loop())

            logger.info(
                f"AnimationService initialized (Rhubarb: "
                f"{'available' if self.rhubarb_available else 'not available'})"
            )
            return True

        except ImportError:
            logger.error("Event bus not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize AnimationService: {e}", exc_info=True)
            return False

    async def stop(self) -> None:
        """Stop the animation service."""
        self.is_running = False

        # Cancel viseme task
        if self._viseme_task:
            self._viseme_task.cancel()
            try:
                await self._viseme_task
            except asyncio.CancelledError:
                pass

        # Clear active sessions
        self._active_sessions.clear()

        logger.info("AnimationService stopped")

    async def _on_tts_started(self, event) -> None:
        """Handle TTS started event - prepare lip sync data."""
        text = event.data.get("text", "")
        audio_file = event.data.get("audio_file")  # Optional
        correlation_id = event.correlation_id

        if not text:
            logger.debug("TTS started without text, skipping lip sync prep")
            return

        logger.info(f"TTS started: preparing lip sync for '{text[:50]}...'")

        # If audio file is provided, generate lip sync immediately
        if audio_file and os.path.isfile(audio_file):
            await self._prepare_lip_sync(audio_file, text, correlation_id)
        else:
            # Store pending request - will process when audio playback starts
            if correlation_id:
                self._pending_requests[correlation_id] = text

    async def _on_audio_playback_started(self, event) -> None:
        """Handle audio playback started - start lip sync session."""
        audio_file = event.data.get("audio_file")
        correlation_id = event.correlation_id
        text = event.data.get("text", "")

        # Check for pending text from TTS_STARTED
        if correlation_id and correlation_id in self._pending_requests:
            text = self._pending_requests.pop(correlation_id)

        # Generate lip sync if we have audio and text
        if audio_file and os.path.isfile(audio_file):
            lip_sync_data = await self._generate_lip_sync(audio_file, text)

            if lip_sync_data:
                await self._start_lip_sync_session(
                    lip_sync_data, correlation_id
                )
            elif self.enable_fallback and text:
                # Fall back to text-based animation via display manager
                await self._emit_fallback_start(text, correlation_id)
        elif text:
            # No audio file, use text-based fallback
            await self._emit_fallback_start(text, correlation_id)

    async def _on_audio_playback_ended(self, event) -> None:
        """Handle audio playback ended - stop lip sync session."""
        correlation_id = event.correlation_id

        # Stop any active session for this correlation
        await self._stop_session_by_correlation(correlation_id)

    async def _on_tts_completed(self, event) -> None:
        """Handle TTS completed event - ensure lip sync is stopped."""
        correlation_id = event.correlation_id
        await self._stop_session_by_correlation(correlation_id)

        # Emit silence viseme
        await self._emit_viseme("SILENCE", correlation_id)

    async def _on_system_stopped(self, event) -> None:
        """Handle system stopped event."""
        await self.stop()

    async def _generate_lip_sync(
        self,
        audio_file: str,
        text: Optional[str]
    ) -> Optional[LipSyncData]:
        """Generate lip sync data for an audio file."""
        if not self.rhubarb_available:
            return None

        try:
            lip_sync_data = await self.lip_sync_service.generate_lip_sync_async(
                audio_file, text
            )

            if lip_sync_data:
                self.stats["rhubarb_successes"] += 1
                logger.info(
                    f"Lip sync generated: {len(lip_sync_data.cues)} cues, "
                    f"{lip_sync_data.duration:.2f}s"
                )
            else:
                self.stats["rhubarb_failures"] += 1

            return lip_sync_data

        except Exception as e:
            logger.error(f"Error generating lip sync: {e}")
            self.stats["rhubarb_failures"] += 1
            return None

    async def _prepare_lip_sync(
        self,
        audio_file: str,
        text: str,
        correlation_id: Optional[str]
    ) -> None:
        """Pre-generate lip sync data before playback starts."""
        # Generate lip sync in background
        asyncio.create_task(
            self._generate_and_cache_lip_sync(audio_file, text, correlation_id)
        )

    async def _generate_and_cache_lip_sync(
        self,
        audio_file: str,
        text: str,
        correlation_id: Optional[str]
    ) -> None:
        """Generate and cache lip sync data."""
        lip_sync_data = await self._generate_lip_sync(audio_file, text)
        # The lip_sync_service caches internally, so data will be available
        # when audio playback starts

    async def _start_lip_sync_session(
        self,
        lip_sync_data: LipSyncData,
        correlation_id: Optional[str]
    ) -> None:
        """Start a new lip sync playback session."""
        session_id = f"lipsync_{time.time()}"

        session = LipSyncSession(
            session_id=session_id,
            lip_sync_data=lip_sync_data,
            start_time=time.time(),
            correlation_id=correlation_id
        )

        self._active_sessions[session_id] = session
        self.stats["sessions_started"] += 1

        logger.info(f"Lip sync session started: {session_id}")

        # Emit lip sync started event
        if self.event_bus:
            from src.core.event_bus import EventType, emit_event
            await emit_event(
                EventType.LIP_SYNC_STARTED,
                {
                    "session_id": session_id,
                    "duration": lip_sync_data.duration,
                    "cue_count": len(lip_sync_data.cues),
                    "audio_file": lip_sync_data.audio_file
                },
                source="animation_service",
                correlation_id=correlation_id
            )

    async def _stop_session_by_correlation(
        self,
        correlation_id: Optional[str]
    ) -> None:
        """Stop sessions matching a correlation ID."""
        if not correlation_id:
            return

        to_remove = []
        for session_id, session in self._active_sessions.items():
            if session.correlation_id == correlation_id:
                session.is_active = False
                to_remove.append((session_id, session))

        for session_id, session in to_remove:
            del self._active_sessions[session_id]
            self.stats["sessions_completed"] += 1
            logger.debug(f"Lip sync session stopped: {session_id}")

            # Emit lip sync completed event
            if self.event_bus:
                from src.core.event_bus import EventType, emit_event
                await emit_event(
                    EventType.LIP_SYNC_COMPLETED,
                    {
                        "session_id": session_id,
                        "duration": session.lip_sync_data.duration,
                        "elapsed": session.get_elapsed_time()
                    },
                    source="animation_service",
                    correlation_id=correlation_id
                )

    async def _viseme_emission_loop(self) -> None:
        """Main loop for emitting viseme events based on active sessions."""
        logger.info("Viseme emission loop started")

        while self.is_running:
            try:
                await asyncio.sleep(self.viseme_emit_interval)

                # Process active sessions
                completed_sessions = []

                for session_id, session in self._active_sessions.items():
                    if not session.is_active:
                        completed_sessions.append(session_id)
                        continue

                    # Get current viseme based on elapsed time
                    elapsed = session.get_elapsed_time()
                    cue = session.lip_sync_data.get_viseme_at_time(elapsed)

                    if cue:
                        # Convert to internal viseme and emit
                        internal_viseme = self.lip_sync_service.convert_to_internal_viseme(
                            cue.viseme
                        )
                        await self._emit_viseme(
                            internal_viseme,
                            session.correlation_id
                        )

                    # Check if session is complete
                    if session.is_complete():
                        session.is_active = False
                        completed_sessions.append(session_id)

                # Clean up completed sessions
                for session_id in completed_sessions:
                    if session_id in self._active_sessions:
                        del self._active_sessions[session_id]
                        self.stats["sessions_completed"] += 1
                        logger.debug(f"Lip sync session completed: {session_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in viseme emission loop: {e}")

        logger.info("Viseme emission loop stopped")

    async def _emit_viseme(
        self,
        viseme_name: str,
        correlation_id: Optional[str] = None
    ) -> None:
        """Emit a viseme update event."""
        if not self.event_bus:
            return

        from src.core.event_bus import EventType, emit_event

        await emit_event(
            EventType.MOUTH_SHAPE_UPDATE,
            {"viseme": viseme_name},
            source="animation_service",
            correlation_id=correlation_id
        )
        self.stats["visemes_emitted"] += 1

    async def _emit_fallback_start(
        self,
        text: str,
        correlation_id: Optional[str]
    ) -> None:
        """Emit event to trigger fallback text-based lip sync."""
        if not self.event_bus:
            return

        self.stats["fallback_used"] += 1
        logger.debug(f"Using fallback lip sync for: {text[:30]}...")

        # The display manager's TTS_STARTED handler already handles
        # text-based lip sync, so we don't need to do anything special here.
        # The speak_text() method in the display manager will animate
        # based on character-to-viseme mapping.

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "active_sessions": len(self._active_sessions),
            "rhubarb_available": self.rhubarb_available,
            "is_running": self.is_running
        }

    async def process_audio_file(
        self,
        audio_file: str,
        text: Optional[str] = None
    ) -> Optional[LipSyncData]:
        """
        Process an audio file and return lip sync data.

        This can be called directly for testing or pre-processing.

        Args:
            audio_file: Path to audio file
            text: Optional transcript for better accuracy

        Returns:
            LipSyncData or None if processing failed
        """
        return await self._generate_lip_sync(audio_file, text)


async def main():
    """Main entry point for the animation service."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Animation Service')
    parser.add_argument(
        '--rhubarb-path',
        help='Path to rhubarb executable'
    )
    parser.add_argument(
        '--test-audio',
        help='Test with an audio file'
    )
    parser.add_argument(
        '--test-text',
        help='Text transcript for test audio'
    )
    args = parser.parse_args()

    # Create service
    service = AnimationService(rhubarb_path=args.rhubarb_path)

    # Test mode
    if args.test_audio:
        if not service.lip_sync_service.is_available():
            print("Rhubarb is not available. Cannot test.")
            return

        print(f"\nProcessing: {args.test_audio}")
        result = service.lip_sync_service.generate_lip_sync(
            args.test_audio,
            args.test_text
        )

        if result:
            print(f"\nDuration: {result.duration:.2f}s")
            print(f"Cues: {len(result.cues)}")
            print("\nViseme timeline:")
            for cue in result.cues[:30]:
                internal = service.lip_sync_service.convert_to_internal_viseme(
                    cue.viseme
                )
                print(
                    f"  {cue.start_time:6.3f}s: {cue.viseme.value} -> {internal}"
                )
            if len(result.cues) > 30:
                print(f"  ... and {len(result.cues) - 30} more cues")
        else:
            print("Failed to generate lip sync data")
        return

    # Event-driven mode
    from src.core.event_bus import EventBus, emit_event, EventType

    bus = await EventBus.get_instance()
    await bus.start()

    if not await service.initialize():
        print("Failed to initialize animation service")
        return

    await emit_event(EventType.SYSTEM_STARTED, {"service": "animation"})

    print("\n" + "=" * 60)
    print("Animation Service Running")
    print("=" * 60)
    print(f"Rhubarb available: {service.rhubarb_available}")
    print("Listening for TTS events...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping animation service...")
    finally:
        await emit_event(EventType.SYSTEM_STOPPED, {"service": "animation"})
        await service.stop()
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(main())
