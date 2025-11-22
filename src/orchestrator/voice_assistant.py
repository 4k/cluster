#!/usr/bin/env python3
"""
Event-Driven Voice Assistant Orchestrator
Coordinates STT, LLM, and TTS services through the event bus.

This orchestrator:
- Initializes the event bus
- Starts all services (STT, LLM, TTS)
- Coordinates service communication via events
- Tracks requests with correlation IDs
- Provides monitoring and statistics
"""

import sys
import asyncio
import logging
import argparse
import signal
from pathlib import Path

# Import services
from src.services.stt_service import STTService
from src.services.llm_service import LLMService
from src.services.tts_service import TTSService

# Import event bus
from src.core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Event-driven voice assistant orchestrator."""

    def __init__(self, config: dict):
        """
        Initialize the voice assistant.

        Args:
            config: Configuration dictionary with service settings
        """
        self.config = config
        self.event_bus = None
        self.stt_service = None
        self.llm_service = None
        self.tts_service = None
        self.is_running = False
        self.stats = {
            "requests_processed": 0,
            "responses_generated": 0,
            "speech_synthesized": 0,
            "errors": 0
        }

    async def initialize(self):
        """Initialize all services and event bus."""
        logger.info("Initializing Event-Driven Voice Assistant...")

        # Initialize event bus
        self.event_bus = await EventBus.get_instance()
        await self.event_bus.start()
        logger.info("Event bus started")

        # Subscribe to events for monitoring
        self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word_detected)
        self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
        self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)
        self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

        # Initialize STT service
        logger.info("Initializing STT service...")
        self.stt_service = STTService(
            wake_word=self.config.get("wake_word", "jarvis"),
            threshold=self.config.get("threshold", 0.5),
            device_index=self.config.get("device_index"),
            verbose=self.config.get("verbose", False),
            vosk_model_path=self.config.get("vosk_model_path")
        )
        await self.stt_service.initialize()
        logger.info("STT service initialized")

        # Initialize LLM service
        logger.info("Initializing LLM service...")
        self.llm_service = LLMService(
            base_url=self.config.get("llm_base_url", "http://192.168.1.144:11434"),
            default_model=self.config.get("llm_model", "qwen3-coder:30b")
        )
        await self.llm_service.initialize()
        logger.info("LLM service initialized")

        # Initialize TTS service
        logger.info("Initializing TTS service...")
        self.tts_service = TTSService(
            model_path=self.config.get("tts_model_path")
        )
        await self.tts_service.initialize()
        logger.info("TTS service initialized")

        # Emit system started event
        await emit_event(
            EventType.SYSTEM_STARTED,
            {
                "services": ["stt", "llm", "tts"],
                "version": "1.0.0",
                "config": {
                    "wake_word": self.config.get("wake_word", "jarvis"),
                    "llm_model": self.config.get("llm_model", "qwen3-coder:30b")
                }
            },
            source="orchestrator"
        )

        logger.info("All services initialized successfully")

    async def _on_wake_word_detected(self, event):
        """Handle wake word detected event."""
        wake_word = event.data.get("wake_word")
        confidence = event.data.get("confidence")
        logger.info(f"🎤 Wake word '{wake_word}' detected (confidence: {confidence:.2f})")

    async def _on_speech_detected(self, event):
        """Handle speech detected event."""
        text = event.data.get("text")
        correlation_id = event.correlation_id
        self.stats["requests_processed"] += 1
        logger.info(f"📝 Speech detected: '{text}' (correlation_id: {correlation_id})")

    async def _on_response_generated(self, event):
        """Handle response generated event."""
        response = event.data.get("response")
        elapsed_time = event.data.get("elapsed_time", 0)
        correlation_id = event.correlation_id
        self.stats["responses_generated"] += 1
        logger.info(f"🧠 Response generated in {elapsed_time:.2f}s: '{response[:50]}...' (correlation_id: {correlation_id})")

    async def _on_tts_completed(self, event):
        """Handle TTS completed event."""
        text = event.data.get("text")
        correlation_id = event.correlation_id
        self.stats["speech_synthesized"] += 1
        logger.info(f"🔊 Speech synthesized: '{text[:50]}...' (correlation_id: {correlation_id})")

    async def _on_error(self, event):
        """Handle error events."""
        error = event.data.get("error")
        service = event.data.get("service")
        operation = event.data.get("operation")
        self.stats["errors"] += 1
        logger.error(f"❌ Error in {service}/{operation}: {error}")

    async def run(self):
        """Run the voice assistant."""
        self.is_running = True

        # Print startup message
        print("\n" + "="*70)
        print("🎙️  EVENT-DRIVEN VOICE ASSISTANT")
        print("="*70)
        print(f"Wake Word: {self.config.get('wake_word', 'jarvis').upper()}")
        print(f"LLM Model: {self.config.get('llm_model', 'qwen3-coder:30b')}")
        print(f"Event Bus: Active")
        print(f"Services: STT → LLM → TTS (via event bus)")
        print("="*70)
        print("\n✅ All services ready!")
        print(f"\n👂 Say 'Hey {self.config.get('wake_word', 'jarvis').title()}' to start...")
        print("📊 Stats will be shown every 60 seconds")
        print("🛑 Press Ctrl+C to stop\n")

        # Start STT service in background
        stt_task = asyncio.create_task(
            asyncio.to_thread(self.stt_service.start)
        )

        # Monitor stats
        stats_task = asyncio.create_task(self._print_stats_periodically())

        try:
            # Wait for STT to complete or KeyboardInterrupt
            await stt_task
        except asyncio.CancelledError:
            logger.info("Voice assistant shutting down...")
        finally:
            # Cancel stats task
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass

    async def _print_stats_periodically(self):
        """Print statistics periodically."""
        while self.is_running:
            try:
                await asyncio.sleep(60)
                self.print_stats()
            except asyncio.CancelledError:
                break

    def print_stats(self):
        """Print current statistics."""
        print("\n" + "="*70)
        print("📊 VOICE ASSISTANT STATISTICS")
        print("="*70)
        print(f"Requests Processed:    {self.stats['requests_processed']}")
        print(f"Responses Generated:   {self.stats['responses_generated']}")
        print(f"Speech Synthesized:    {self.stats['speech_synthesized']}")
        print(f"Errors:                {self.stats['errors']}")

        # Event bus stats
        if self.event_bus:
            bus_stats = self.event_bus.get_stats()
            metrics = bus_stats.get("metrics", {})
            print(f"\nEvent Bus:")
            print(f"  Total Events:        {metrics.get('total_events', 0)}")
            print(f"  Total Errors:        {metrics.get('total_errors', 0)}")
            print(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.6f}s")

        print("="*70 + "\n")

    async def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Shutting down voice assistant...")
        self.is_running = False

        print("\n" + "="*70)
        print("🛑 Shutting down services...")
        print("="*70)

        # Print final stats
        self.print_stats()

        # Stop STT service
        if self.stt_service:
            self.stt_service.stop()
            logger.info("STT service stopped")

        # Emit system stopped event
        if self.event_bus:
            await emit_event(
                EventType.SYSTEM_STOPPED,
                {"services": ["stt", "llm", "tts"]},
                source="orchestrator"
            )

            # Wait a bit for final events to process
            await asyncio.sleep(0.5)

            # Stop event bus
            await self.event_bus.stop()
            logger.info("Event bus stopped")

        print("\n✅ All services stopped gracefully")
        print("👋 Goodbye!\n")


async def main_async():
    """Async main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Event-Driven Voice Assistant (STT → LLM → TTS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python voice_assistant.py

  # Custom wake word and LLM
  python voice_assistant.py --wake-word alexa --llm-model llama3:8b

  # Custom audio device
  python voice_assistant.py --device 2 --verbose

  # List available audio devices
  python voice_assistant.py --list-devices
"""
    )

    parser.add_argument(
        '--wake-word',
        type=str,
        default='jarvis',
        help='Wake word to detect (default: jarvis)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Wake word detection threshold 0.0-1.0 (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        type=int,
        help='Audio input device index'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detection scores in real-time'
    )
    parser.add_argument(
        '--llm-base-url',
        type=str,
        default='http://192.168.1.144:11434',
        help='Ollama base URL (default: http://192.168.1.144:11434)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='qwen3-coder:30b',
        help='LLM model name (default: qwen3-coder:30b)'
    )
    parser.add_argument(
        '--tts-model',
        type=str,
        help='Path to TTS model (.onnx file)'
    )
    parser.add_argument(
        '--vosk-model',
        type=str,
        help='Path to Vosk STT model directory'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List audio devices and exit'
    )

    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        from src.services.stt_service import list_audio_devices
        list_audio_devices()
        return 0

    # Build configuration
    config = {
        "wake_word": args.wake_word,
        "threshold": args.threshold,
        "device_index": args.device,
        "verbose": args.verbose,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "tts_model_path": args.tts_model,
        "vosk_model_path": args.vosk_model
    }

    # Create and initialize assistant
    assistant = VoiceAssistant(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(assistant.shutdown())

    # Register signal handlers (Unix only - Windows uses KeyboardInterrupt)
    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        # Initialize services
        await assistant.initialize()

        # Run assistant
        await assistant.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        await assistant.shutdown()

    return 0


def main():
    """Main entry point."""
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
