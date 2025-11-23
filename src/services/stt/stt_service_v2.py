"""
Backend-Agnostic STT Service (V2).

This service provides speech-to-text functionality with support for
multiple STT engines and wake word detection engines. It is completely
decoupled from specific implementations.

Key features:
- Engine-agnostic: Works with Vosk, Whisper, Azure, Google, etc.
- Wake word-agnostic: Works with OpenWakeWord, Porcupine, etc.
- Event-driven: Integrates with the application event bus
- Configurable: Extensive YAML configuration support
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import pyaudio

from src.core.audio.stt_engine import STTEngine
from src.core.audio.wake_word_engine import WakeWordEngine
from src.core.audio.stt_types import STTResult, WakeWordResult
from src.core.event_bus import EventBus, EventType, emit_event

from .factory import STTConfig, STTEngineFactory, WakeWordEngineFactory

logger = logging.getLogger(__name__)


class STTServiceV2:
    """
    Backend-agnostic Speech-to-Text service.

    This service orchestrates wake word detection and speech recognition
    without knowing the specific implementations being used.
    """

    def __init__(
        self,
        stt_engine: Optional[STTEngine] = None,
        wake_word_engine: Optional[WakeWordEngine] = None,
        config: Optional[STTConfig] = None,
    ):
        """
        Initialize the STT service.

        Args:
            stt_engine: STT engine instance (or created from config)
            wake_word_engine: Wake word engine instance (or created from config)
            config: STT configuration (or loaded from file)
        """
        # Load configuration
        self.config = config or STTConfig.load()

        # Create engines from config if not provided
        self.stt_engine = stt_engine or STTEngineFactory.create(self.config)
        self.wake_word_engine = wake_word_engine or WakeWordEngineFactory.create(self.config)

        # Audio settings
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._format = pyaudio.paInt16
        self._channels = 1

        # Actual audio parameters (may differ from config due to device)
        self._actual_sample_rate = self.config.sample_rate
        self._actual_chunk_size = self.config.chunk_size
        self._input_channels = 1

        # Event bus
        self._event_bus: Optional[EventBus] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # State
        self._is_running = False

        # Statistics
        self.stats = {
            "wake_words_detected": 0,
            "transcriptions": 0,
            "transcription_errors": 0,
            "total_audio_seconds": 0.0,
        }

        logger.info(
            f"STTServiceV2 initialized: stt_engine={self.stt_engine.name}, "
            f"wake_word_engine={self.wake_word_engine.name}"
        )

    async def initialize(self) -> None:
        """Initialize the service and connect to event bus."""
        logger.info("Initializing STT service...")

        # Initialize engines
        await self.stt_engine.initialize()
        await self.wake_word_engine.initialize()

        # Connect to event bus
        self._event_bus = await EventBus.get_instance()
        self._loop = asyncio.get_running_loop()

        # Subscribe to events
        self._event_bus.subscribe(EventType.SYSTEM_STOPPED, self._on_system_stopped)

        logger.info("STT service initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        logger.info("Shutting down STT service...")
        self._is_running = False

        # Close audio stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass

        if self._audio:
            self._audio.terminate()

        # Shutdown engines
        await self.stt_engine.shutdown()
        await self.wake_word_engine.shutdown()

        logger.info("STT service shutdown complete")

    async def _on_system_stopped(self, event) -> None:
        """Handle system stopped event."""
        await self.shutdown()

    def _emit_event_sync(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit event from sync context."""
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                emit_event(event_type, data, correlation_id=correlation_id, source="stt"),
                self._loop,
            )
            future.add_done_callback(
                lambda f: logger.error(f"Event emit error: {f.exception()}")
                if f.exception() else None
            )

    def _setup_audio_stream(self) -> bool:
        """Set up the audio input stream."""
        self._audio = pyaudio.PyAudio()

        try:
            # Get device info
            if self.config.device_index is not None:
                device_info = self._audio.get_device_info_by_index(self.config.device_index)
                device_index = self.config.device_index
            else:
                device_info = self._audio.get_default_input_device_info()
                device_index = device_info['index']

            logger.info(f"Using audio device [{device_index}]: {device_info['name']}")

            # Try different configurations
            sample_rates = [16000, int(device_info['defaultSampleRate'])]
            channel_options = [1, 2] if device_info['maxInputChannels'] >= 2 else [1]
            buffer_sizes = [1280, 1024, 2048, 4096]

            for rate in sample_rates:
                for channels in channel_options:
                    for buffer_size in buffer_sizes:
                        try:
                            self._stream = self._audio.open(
                                format=self._format,
                                channels=channels,
                                rate=rate,
                                input=True,
                                input_device_index=device_index,
                                frames_per_buffer=buffer_size,
                            )
                            self._actual_sample_rate = rate
                            self._actual_chunk_size = buffer_size
                            self._input_channels = channels
                            logger.info(
                                f"Audio stream opened: {channels}ch, {rate}Hz, "
                                f"buffer={buffer_size}"
                            )
                            return True
                        except OSError:
                            continue

            logger.error("Could not open audio stream with any configuration")
            return False

        except Exception as e:
            logger.error(f"Failed to setup audio: {e}")
            return False

    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data for wake word detection."""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Convert stereo to mono if needed
        if self._input_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample to 16kHz if needed
        if self._actual_sample_rate != 16000:
            try:
                from scipy import signal
                num_samples = int(len(audio_array) * 16000 / self._actual_sample_rate)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
            except ImportError:
                logger.warning("scipy not available, skipping resampling")

        return audio_array

    async def _transcribe_speech(self, correlation_id: str) -> Optional[STTResult]:
        """Transcribe speech after wake word detection."""
        logger.info("Listening for speech...")
        print("\n🎤 Listening... (speak now)")

        # Create recognizer for streaming
        recognizer = self.stt_engine.create_recognizer(self._actual_sample_rate)

        # Record audio
        frames = []
        silent_chunks = 0
        max_frames = int(
            self._actual_sample_rate / self._actual_chunk_size *
            self.config.max_recording_seconds
        )
        max_silent = int(
            self._actual_sample_rate / self._actual_chunk_size *
            self.config.silence_duration_seconds
        )

        final_text = ""

        for _ in range(max_frames):
            data = self._stream.read(self._actual_chunk_size, exception_on_overflow=False)
            frames.append(data)

            # Process with recognizer (Vosk-style API)
            if hasattr(recognizer, 'AcceptWaveform'):
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        final_text += " " + result["text"]

            # Check for silence
            audio_array = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

            if rms < self.config.silence_threshold_rms:
                silent_chunks += 1
                if silent_chunks > max_silent and final_text:
                    break
            else:
                silent_chunks = 0

        # Get final result
        if hasattr(recognizer, 'FinalResult'):
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get("text"):
                final_text += " " + final_result["text"]

        final_text = final_text.strip()

        if final_text:
            self.stats["transcriptions"] += 1

            # Calculate duration
            audio_data = b''.join(frames)
            duration = len(audio_data) / (2 * self._actual_sample_rate)
            self.stats["total_audio_seconds"] += duration

            result = STTResult(
                text=final_text,
                confidence=0.95,
                duration=duration,
                engine=self.stt_engine.name,
            )

            # Emit events
            await emit_event(
                EventType.SPEECH_DETECTED,
                {
                    "text": final_text,
                    "confidence": result.confidence,
                    "timestamp": time.time(),
                    "language": "en",
                },
                correlation_id=correlation_id,
                source="stt",
            )

            await emit_event(
                EventType.SPEECH_ENDED,
                {"text": final_text},
                correlation_id=correlation_id,
                source="stt",
            )

            return result
        else:
            self.stats["transcription_errors"] += 1
            logger.warning("No speech detected")
            return None

    async def start(self) -> None:
        """Start the STT service with wake word detection."""
        self._is_running = True

        # Setup audio
        if not self._setup_audio_stream():
            raise RuntimeError("Failed to setup audio stream")

        print(f"\n{'='*60}")
        print(f"👂 Listening for wake word: '{self.config.wake_word.upper()}'")
        print(f"   STT Engine: {self.stt_engine.name}")
        print(f"   Wake Word Engine: {self.wake_word_engine.name}")
        print(f"   Sample Rate: {self._actual_sample_rate} Hz")
        print(f"   Detection Threshold: {self.config.threshold}")
        print(f"   Say '{self.config.wake_word}' to activate")
        print(f"   (Press Ctrl+C to stop)")
        print(f"{'='*60}\n")

        try:
            while self._is_running:
                # Read audio chunk
                audio_data = self._stream.read(
                    self._actual_chunk_size,
                    exception_on_overflow=False,
                )

                # Preprocess
                audio_array = self._preprocess_audio(audio_data)

                # Detect wake word
                result = self.wake_word_engine.detect(
                    audio_array,
                    wake_words=[self.config.wake_word],
                )

                if result.detected:
                    print(f"\n✅ Wake word detected! (confidence: {result.confidence:.3f})")
                    self.stats["wake_words_detected"] += 1

                    # Generate correlation ID
                    correlation_id = f"stt-{uuid.uuid4().hex[:12]}"

                    # Emit wake word event
                    await emit_event(
                        EventType.WAKE_WORD_DETECTED,
                        {
                            "wake_word": result.wake_word,
                            "model": result.model_name,
                            "confidence": result.confidence,
                            "timestamp": result.timestamp,
                        },
                        correlation_id=correlation_id,
                        source="stt",
                    )

                    # Emit audio started
                    await emit_event(
                        EventType.AUDIO_STARTED,
                        {"device_index": self.config.device_index},
                        correlation_id=correlation_id,
                        source="stt",
                    )

                    # Transcribe speech
                    stt_result = await self._transcribe_speech(correlation_id)

                    if stt_result:
                        print(f"\n📝 Transcribed: \"{stt_result.text}\"\n")

                    # Emit audio stopped
                    await emit_event(
                        EventType.AUDIO_STOPPED,
                        {},
                        correlation_id=correlation_id,
                        source="stt",
                    )

                    print(f"👂 Listening for wake word: '{self.config.wake_word.upper()}'...")

                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n👋 Stopping STT service...")
        finally:
            await self.shutdown()

    def stop(self) -> None:
        """Stop the STT service."""
        self._is_running = False

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "stt_engine": self.stt_engine.get_info(),
            "wake_word_engine": self.wake_word_engine.get_info(),
            "config": {
                "wake_word": self.config.wake_word,
                "threshold": self.config.threshold,
                "sample_rate": self._actual_sample_rate,
            },
            "stats": self.stats,
            "is_running": self._is_running,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.stats.copy()
