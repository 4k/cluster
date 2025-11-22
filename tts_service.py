#!/usr/bin/env python3
"""
Text-to-Speech Service using Piper-TTS 1.3.0.
Converts text to speech and plays it through loudspeakers.
Event-driven version with queuing support to prevent message interruption.
"""
import sys
import logging
import argparse
import asyncio
import time
import wave
import io
import tempfile
from pathlib import Path
from collections import deque
from typing import Optional
import numpy as np
from piper.voice import PiperVoice
import pyaudio

# Event bus imports
from src.core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service using Piper-TTS for high-quality voice synthesis. Event-driven with async queuing."""

    def __init__(self, model_path=None):
        """
        Initialize the TTS service.

        Args:
            model_path: Path to Piper voice model (.onnx file)
        """
        self.model_path = model_path or self._get_default_model()

        # Initialize Piper voice
        logger.info(f"Loading Piper voice model: {self.model_path}")
        try:
            self.voice = PiperVoice.load(str(self.model_path))
            logger.info("TTS Service initialized successfully")
            logger.info(f"Sample rate: {self.voice.config.sample_rate} Hz")
        except Exception as e:
            logger.error(f"Failed to load Piper voice model: {e}", exc_info=True)
            raise

        # Initialize PyAudio for playback
        self.audio = pyaudio.PyAudio()

        # Event bus and queuing
        self.event_bus = None
        self.tts_queue = asyncio.Queue()
        self.is_processing = False
        self.is_running = False

    async def initialize(self):
        """Initialize event bus connection and subscribe to events."""
        logger.info("Initializing TTS service with event bus...")
        self.event_bus = await EventBus.get_instance()

        # Subscribe to response generated events
        self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)
        self.event_bus.subscribe(EventType.SYSTEM_STOPPED, self._on_system_stopped)

        # Start queue processor
        self.is_running = True
        asyncio.create_task(self._process_queue())

        logger.info("TTS service initialized with event bus and queue processor started")

    async def _on_system_stopped(self, event):
        """Handle system stopped event."""
        logger.info("Received SYSTEM_STOPPED event, stopping TTS service...")
        self.is_running = False

    async def _on_response_generated(self, event):
        """Handle response generated event by queuing TTS request."""
        response = event.data.get("response", "")
        correlation_id = event.correlation_id

        if response:
            logger.info(f"Queueing TTS request: {response[:50]}... (correlation_id: {correlation_id})")
            await self.tts_queue.put({
                "text": response,
                "correlation_id": correlation_id,
                "timestamp": time.time()
            })

    async def _process_queue(self):
        """Process TTS queue to prevent message interruption."""
        logger.info("TTS queue processor started")

        while self.is_running:
            try:
                # Get next item from queue (wait with timeout to check is_running)
                try:
                    request = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                text = request["text"]
                correlation_id = request["correlation_id"]

                logger.info(f"Processing TTS request from queue: {text[:50]}...")

                # Mark as processing
                self.is_processing = True

                # Speak the text
                await self.speak_async(text, correlation_id=correlation_id)

                # Mark as done
                self.is_processing = False

                # Mark task as done
                self.tts_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing TTS queue: {e}", exc_info=True)
                self.is_processing = False

        logger.info("TTS queue processor stopped")

    async def speak_async(self, text: str, correlation_id: Optional[str] = None) -> bool:
        """
        Async version of speak that integrates with event bus.

        Args:
            text: Text to convert to speech
            correlation_id: Correlation ID for request tracing

        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to speak")
            return False

        try:
            # Emit TTS started event
            if self.event_bus:
                await emit_event(
                    EventType.TTS_STARTED,
                    {"text": text, "timestamp": time.time()},
                    correlation_id=correlation_id,
                    source="tts"
                )

            # Emit audio playback started event
            if self.event_bus:
                await emit_event(
                    EventType.AUDIO_PLAYBACK_STARTED,
                    {"text": text},
                    correlation_id=correlation_id,
                    source="tts"
                )

            # Run blocking speak in executor
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self.speak, text)

            # Emit audio playback ended event
            if self.event_bus:
                await emit_event(
                    EventType.AUDIO_PLAYBACK_ENDED,
                    {"success": success},
                    correlation_id=correlation_id,
                    source="tts"
                )

            # Emit TTS completed event
            if self.event_bus:
                await emit_event(
                    EventType.TTS_COMPLETED,
                    {"text": text, "success": success},
                    correlation_id=correlation_id,
                    source="tts"
                )

            return success

        except Exception as e:
            logger.error(f"Error in async speak: {e}", exc_info=True)

            # Emit error event
            if self.event_bus:
                await emit_event(
                    EventType.ERROR_OCCURRED,
                    {
                        "error": str(e),
                        "service": "tts",
                        "operation": "synthesis"
                    },
                    correlation_id=correlation_id,
                    source="tts"
                )
            return False

    def _get_default_model(self):
        """Get default model path or provide download instructions."""
        # Default model directory
        models_dir = Path.home() / ".local" / "share" / "piper" / "voices"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Look for any existing .onnx model
        existing_models = list(models_dir.glob("*.onnx"))
        if existing_models:
            logger.info(f"Using existing model: {existing_models[0]}")
            return existing_models[0]

        # If no model exists, inform user
        logger.warning("No Piper voice model found.")
        logger.info(f"Please download a model to: {models_dir}")
        logger.info("Download models from: https://huggingface.co/rhasspy/piper-voices")
        logger.info("Or run: python download_voice.py")

        raise FileNotFoundError(
            f"No Piper voice model found in {models_dir}. "
            f"Please run 'python download_voice.py' to download a voice model."
        )

    def speak(self, text):
        """
        Convert text to speech and play through loudspeakers.

        Args:
            text: Text to convert to speech

        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to speak")
            return False

        try:
            print(f"\n🔊 Speaking: \"{text}\"")
            logger.info(f"Synthesizing: {text}")

            # Try streaming method first (returns AudioChunk objects in 1.3.0)
            if hasattr(self.voice, 'synthesize'):
                return self._speak_streaming(text)
            else:
                # Fallback to WAV file method
                logger.info("Using WAV file method")
                return self._speak_via_wav(text)

        except Exception as e:
            logger.error(f"Failed to speak text: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def _speak_streaming(self, text):
        """Stream audio directly using synthesize() method (returns AudioChunk objects)."""
        try:
            stream = None
            sample_rate = None

            # Synthesize returns AudioChunk objects
            for chunk in self.voice.synthesize(text):
                # Initialize stream on first chunk (we get sample rate from chunk)
                if stream is None:
                    sample_rate = chunk.sample_rate
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(chunk.sample_width),
                        channels=chunk.sample_channels,
                        rate=sample_rate,
                        output=True
                    )
                    logger.info(f"Audio stream opened: {chunk.sample_rate}Hz, {chunk.sample_channels}ch, {chunk.sample_width}bytes")

                # Play audio chunk
                stream.write(chunk.audio_int16_bytes)

            # Clean up
            if stream:
                stream.stop_stream()
                stream.close()

            logger.info("Speech completed successfully (streaming)")
            return True

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            # Fall back to WAV method
            logger.info("Falling back to WAV file method")
            return self._speak_via_wav(text)

    def _speak_via_wav(self, text):
        """Generate WAV file and play it (fallback method)."""
        temp_file = None
        try:
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Synthesize speech to WAV file using piper-tts 1.3.0 API
            logger.info(f"Calling synthesize_wav with text: '{text}'")

            # Use context manager - synthesize_wav handles all WAV parameters
            with wave.open(temp_path, 'wb') as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            logger.info("Synthesis completed")

            # Check if file has data
            file_size = Path(temp_path).stat().st_size
            logger.info(f"WAV file size: {file_size} bytes")

            # Read and play the WAV file
            with wave.open(temp_path, 'rb') as wav_reader:
                # Get WAV parameters
                sample_width = wav_reader.getsampwidth()
                channels = wav_reader.getnchannels()
                framerate = wav_reader.getframerate()

                logger.info(f"Playing audio: {channels} channel(s), {framerate} Hz, {sample_width} bytes/sample")

                # Open PyAudio stream with explicit parameters
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(sample_width),
                    channels=channels,
                    rate=framerate,
                    output=True,
                    frames_per_buffer=1024
                )

                # Read all audio data
                audio_data = wav_reader.readframes(wav_reader.getnframes())

                # Play audio in chunks to avoid buffer issues
                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    stream.write(chunk)

                # Clean up stream
                stream.stop_stream()
                stream.close()

            logger.info("Speech completed successfully (WAV file)")
            return True

        except Exception as e:
            logger.error(f"WAV playback failed: {e}")
            raise

        finally:
            # Clean up temporary file
            if temp_file and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except:
                    pass

    def speak_to_file(self, text, output_path):
        """
        Convert text to speech and save to WAV file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the WAV file

        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to synthesize")
            return False

        try:
            logger.info(f"Synthesizing to file: {output_path}")

            # Use piper-tts 1.3.0 API - synthesize_wav handles all WAV parameters
            with wave.open(str(output_path), 'wb') as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            logger.info(f"Audio saved to: {output_path}")
            print(f"✅ Audio saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def get_voice_info(self):
        """Get information about the current voice."""
        # Debug: list all available methods
        voice_methods = [m for m in dir(self.voice) if not m.startswith('_')]
        logger.info(f"Available voice methods: {voice_methods}")

        info = {
            'sample_rate': self.voice.config.sample_rate,
            'num_speakers': self.voice.config.num_speakers if hasattr(self.voice.config, 'num_speakers') else 1,
            'has_synthesize': hasattr(self.voice, 'synthesize'),
            'has_synthesize_wav': hasattr(self.voice, 'synthesize_wav'),
            'available_methods': voice_methods,
        }
        return info

    def stop(self):
        """Stop the TTS engine and clean up resources."""
        try:
            self.audio.terminate()
            logger.info("TTS engine stopped")
        except Exception as e:
            logger.error(f"Error stopping TTS engine: {e}")


async def main_async():
    """Async main entry point for TTS service with event bus."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Piper Text-to-Speech Service (Event-Driven)')
    parser.add_argument(
        'text',
        nargs='*',
        help='Text to convert to speech'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to Piper voice model (.onnx file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save audio to file instead of playing'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show voice information and exit'
    )
    parser.add_argument(
        '--event-driven',
        action='store_true',
        help='Run in event-driven mode (listens for RESPONSE_GENERATED events)'
    )

    args = parser.parse_args()

    try:
        # Initialize TTS service
        tts = TTSService(model_path=args.model)

        # Show voice info if requested
        if args.info:
            info = tts.get_voice_info()
            print("\n📢 Voice Information:")
            for key, value in info.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            print()
            return 0

        # Event-driven mode
        if args.event_driven or not args.text:
            # Initialize event bus
            bus = await EventBus.get_instance()
            await bus.start()

            # Emit system started event
            await emit_event(EventType.SYSTEM_STARTED, {"service": "tts"}, source="tts")

            # Initialize event bus integration
            await tts.initialize()

            print("\n" + "="*60)
            print("TTS Service Ready (Event-Driven with Queue)")
            print("="*60)
            print("Listening for RESPONSE_GENERATED events...")
            print("Queue prevents message interruption")
            print("Press Ctrl+C to stop\n")

            try:
                # Keep service running
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping TTS service...")
            finally:
                # Emit system stopped event and stop bus
                await emit_event(EventType.SYSTEM_STOPPED, {"service": "tts"}, source="tts")
                await bus.stop()
                tts.stop()

            return 0

        # Direct mode (for testing)
        text = ' '.join(args.text)

        # Save to file or play audio
        if args.output:
            success = tts.speak_to_file(text, args.output)
        else:
            success = tts.speak(text)

        # Clean up
        tts.stop()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"TTS service error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main entry point for the TTS service."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run async main
    result = asyncio.run(main_async())
    sys.exit(result)


if __name__ == "__main__":
    main()
