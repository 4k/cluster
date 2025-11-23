"""
Backend-Agnostic TTS Service (V2).

This service provides text-to-speech functionality with support for
multiple TTS engines and viseme extraction providers. It is completely
decoupled from specific implementations.

Key features:
- Engine-agnostic: Works with Piper, Azure, OpenAI, etc.
- Provider-agnostic: Works with Rhubarb, text-based, Azure inline, etc.
- Event-driven: Integrates with the application event bus
- Configurable: Extensive YAML configuration support
- Fallback support: Graceful degradation when providers fail
"""

import asyncio
import logging
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import pyaudio

from src.core.audio.tts_engine import TTSEngine, TTSSynthesisError
from src.core.audio.viseme_provider import VisemeProvider, VisemeExtractionError
from src.core.audio.types import TTSResult, VisemeSequence
from src.core.event_bus import EventBus, EventType, emit_event

from .factory import TTSConfig, TTSEngineFactory, VisemeProviderFactory

logger = logging.getLogger(__name__)


class TTSServiceV2:
    """
    Backend-agnostic Text-to-Speech service.

    This service orchestrates TTS synthesis and viseme extraction without
    knowing the specific implementations being used. All backend-specific
    logic is delegated to the engine and provider instances.
    """

    def __init__(
        self,
        engine: Optional[TTSEngine] = None,
        viseme_provider: Optional[VisemeProvider] = None,
        fallback_provider: Optional[VisemeProvider] = None,
        config: Optional[TTSConfig] = None,
    ):
        """
        Initialize the TTS service.

        Args:
            engine: TTS engine instance (or created from config)
            viseme_provider: Viseme provider instance (or created from config)
            fallback_provider: Fallback viseme provider (or created from config)
            config: TTS configuration (or loaded from file)
        """
        # Load configuration
        self.config = config or TTSConfig.load()

        # Create engine and providers from config if not provided
        self.engine = engine or TTSEngineFactory.create(self.config)
        self.viseme_provider = viseme_provider or VisemeProviderFactory.create(self.config)
        self.fallback_provider = fallback_provider or VisemeProviderFactory.create_fallback(self.config)

        # Audio playback
        self._audio = pyaudio.PyAudio()

        # Event bus
        self._event_bus: Optional[EventBus] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Queue for TTS requests
        self._queue: asyncio.Queue = asyncio.Queue()
        self._is_processing = False
        self._is_running = False

        # Audio file management
        if self.config.audio_output_dir:
            self._audio_dir = Path(self.config.audio_output_dir)
        else:
            self._audio_dir = Path(tempfile.gettempdir()) / "cluster_tts_audio"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._audio_files: deque = deque(maxlen=self.config.max_audio_files)

        # Statistics
        self.stats = {
            "syntheses": 0,
            "synthesis_errors": 0,
            "viseme_extractions": 0,
            "viseme_errors": 0,
            "fallback_used": 0,
            "total_duration": 0.0,
        }

        logger.info(
            f"TTSServiceV2 initialized: engine={self.engine.name}, "
            f"viseme_provider={self.viseme_provider.name}"
        )

    async def initialize(self) -> None:
        """Initialize the service and connect to event bus."""
        logger.info("Initializing TTS service...")

        # Initialize engine
        await self.engine.initialize()

        # Initialize viseme providers
        await self.viseme_provider.initialize()
        if self.fallback_provider:
            await self.fallback_provider.initialize()

        # Connect to event bus
        self._event_bus = await EventBus.get_instance()
        self._loop = asyncio.get_running_loop()

        # Subscribe to events
        self._event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)
        self._event_bus.subscribe(EventType.SYSTEM_STOPPED, self._on_system_stopped)

        # Start queue processor
        self._is_running = True
        asyncio.create_task(self._process_queue())

        logger.info("TTS service initialized and listening for events")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        logger.info("Shutting down TTS service...")
        self._is_running = False

        # Shutdown engine and providers
        await self.engine.shutdown()
        await self.viseme_provider.shutdown()
        if self.fallback_provider:
            await self.fallback_provider.shutdown()

        # Cleanup audio
        self._audio.terminate()

        logger.info("TTS service shutdown complete")

    async def _on_response_generated(self, event) -> None:
        """Handle response generated event."""
        response = event.data.get("response", "")
        correlation_id = event.correlation_id

        if response:
            logger.info(f"Queueing TTS: {response[:50]}...")
            await self._queue.put({
                "text": response,
                "correlation_id": correlation_id,
                "timestamp": time.time(),
            })

    async def _on_system_stopped(self, event) -> None:
        """Handle system stopped event."""
        await self.shutdown()

    async def _process_queue(self) -> None:
        """Process TTS queue."""
        logger.info("TTS queue processor started")

        while self._is_running:
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                self._is_processing = True
                await self.speak(
                    request["text"],
                    correlation_id=request["correlation_id"],
                )
                self._queue.task_done()
            except Exception as e:
                logger.error(f"TTS processing error: {e}", exc_info=True)
            finally:
                self._is_processing = False

        logger.info("TTS queue processor stopped")

    async def speak(
        self,
        text: str,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Synthesize and play text.

        This is the main entry point for TTS. It:
        1. Synthesizes text to audio using the configured engine
        2. Extracts visemes using the configured provider
        3. Emits events for animation synchronization
        4. Plays the audio

        Args:
            text: Text to synthesize
            correlation_id: Correlation ID for event tracing

        Returns:
            True if successful
        """
        if not text or not text.strip():
            logger.warning("Empty text, nothing to speak")
            return False

        try:
            # Determine output path
            output_path = None
            if self.config.keep_audio_files or self.viseme_provider.requires_audio:
                timestamp = int(time.time() * 1000)
                output_path = self._audio_dir / f"tts_{timestamp}.wav"

            # Synthesize audio
            result = await self.engine.synthesize(text, output_path=output_path)
            self.stats["syntheses"] += 1
            self.stats["total_duration"] += result.duration

            # Track audio file
            if result.audio_file:
                self._audio_files.append(result.audio_file)
                self._cleanup_old_files()

            # Emit TTS_STARTED
            await emit_event(
                EventType.TTS_STARTED,
                {
                    "text": text,
                    "audio_file": str(result.audio_file) if result.audio_file else None,
                    "duration": result.duration,
                    "engine": self.engine.name,
                    "timestamp": time.time(),
                },
                correlation_id=correlation_id,
                source="tts",
            )

            # Extract visemes (if not provided by engine)
            visemes = result.visemes
            if not visemes:
                visemes = await self._extract_visemes(result, text)

            # Emit LIP_SYNC_READY
            await emit_event(
                EventType.LIP_SYNC_READY,
                {
                    "duration": visemes.duration if visemes else result.duration,
                    "cue_count": len(visemes.cues) if visemes else 0,
                    "provider": visemes.provider if visemes else "none",
                    "audio_file": str(result.audio_file) if result.audio_file else None,
                },
                correlation_id=correlation_id,
                source="tts",
            )

            # Wait for visemes if configured
            if self.config.wait_for_visemes and visemes:
                # Small delay to ensure animation service received the event
                await asyncio.sleep(0.01)

            # Play audio
            success = await self._play_audio(result, text, correlation_id, visemes)

            # Emit TTS_COMPLETED
            await emit_event(
                EventType.TTS_COMPLETED,
                {"text": text, "success": success},
                correlation_id=correlation_id,
                source="tts",
            )

            return success

        except TTSSynthesisError as e:
            logger.error(f"Synthesis failed: {e}")
            self.stats["synthesis_errors"] += 1
            await emit_event(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "service": "tts", "operation": "synthesis"},
                correlation_id=correlation_id,
                source="tts",
            )
            return False
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return False

    async def _extract_visemes(
        self,
        result: TTSResult,
        text: str,
    ) -> Optional[VisemeSequence]:
        """Extract visemes from synthesis result."""
        try:
            # Try primary provider
            visemes = await self.viseme_provider.extract_visemes(
                audio_path=result.audio_file,
                text=text,
                audio_duration=result.duration,
            )
            self.stats["viseme_extractions"] += 1
            return visemes

        except VisemeExtractionError as e:
            logger.warning(f"Viseme extraction failed: {e}")
            self.stats["viseme_errors"] += 1

            # Try fallback
            if self.fallback_provider:
                try:
                    logger.info(f"Using fallback provider: {self.fallback_provider.name}")
                    visemes = await self.fallback_provider.extract_visemes(
                        audio_path=result.audio_file,
                        text=text,
                        audio_duration=result.duration,
                    )
                    self.stats["fallback_used"] += 1
                    return visemes
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            return None

    async def _play_audio(
        self,
        result: TTSResult,
        text: str,
        correlation_id: Optional[str],
        visemes: Optional[VisemeSequence],
    ) -> bool:
        """Play audio file."""
        if not result.audio_file or not result.audio_file.exists():
            logger.error("No audio file to play")
            return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._play_audio_sync,
            str(result.audio_file),
            text,
            correlation_id,
            result.duration,
        )

    def _play_audio_sync(
        self,
        audio_path: str,
        text: str,
        correlation_id: Optional[str],
        duration: float,
    ) -> bool:
        """Synchronous audio playback (runs in executor)."""
        import wave

        try:
            with wave.open(audio_path, 'rb') as wav:
                stream = self._audio.open(
                    format=self._audio.get_format_from_width(wav.getsampwidth()),
                    channels=wav.getnchannels(),
                    rate=wav.getframerate(),
                    output=True,
                    frames_per_buffer=1024,
                )

                # Capture start timestamp
                start_timestamp = time.time()

                # Emit playback started
                self._emit_event_sync(
                    EventType.AUDIO_PLAYBACK_STARTED,
                    {
                        "text": text,
                        "audio_file": audio_path,
                        "duration": duration,
                        "start_timestamp": start_timestamp,
                    },
                    correlation_id,
                )

                # Play audio
                audio_data = wav.readframes(wav.getnframes())
                for i in range(0, len(audio_data), 4096):
                    stream.write(audio_data[i:i + 4096])

                stream.stop_stream()
                stream.close()

                # Emit playback ended
                self._emit_event_sync(
                    EventType.AUDIO_PLAYBACK_ENDED,
                    {
                        "success": True,
                        "audio_file": audio_path,
                    },
                    correlation_id,
                )

            return True

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False

    def _emit_event_sync(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        correlation_id: Optional[str],
    ) -> None:
        """Emit event from sync context."""
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                emit_event(event_type, data, correlation_id=correlation_id, source="tts"),
                self._loop,
            )
            future.add_done_callback(
                lambda f: logger.error(f"Event emit error: {f.exception()}")
                if f.exception() else None
            )

    def _cleanup_old_files(self) -> None:
        """Cleanup old audio files beyond max limit."""
        while len(self._audio_files) > self.config.max_audio_files:
            old_file = self._audio_files.popleft()
            try:
                if old_file.exists():
                    old_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {old_file}: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "engine": self.engine.get_info(),
            "viseme_provider": self.viseme_provider.get_info(),
            "fallback_provider": self.fallback_provider.get_info() if self.fallback_provider else None,
            "config": {
                "wait_for_visemes": self.config.wait_for_visemes,
                "keep_audio_files": self.config.keep_audio_files,
            },
            "stats": self.stats,
            "is_running": self._is_running,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.stats.copy()
