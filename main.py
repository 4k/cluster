#!/usr/bin/env python3
"""
Cluster Voice Assistant - Main Entry Point

An event-driven voice assistant with pluggable features.

Core Services:
- STT: Speech-to-Text with wake word detection
- LLM: Language model integration (Ollama)
- TTS: Text-to-Speech synthesis (Piper)

Features (optional):
- display: Animated face with lip sync

Usage:
    python main.py                          # Run with default features
    python main.py --features display       # Enable display feature
    python main.py --no-features            # Core services only
    python main.py --list-features          # Show available features
"""
import os
import sys
import signal
import logging
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Event-driven voice assistant with pluggable features."""

    def __init__(self, enabled_features: Optional[List[str]] = None):
        self.enabled_features = enabled_features or []
        self.event_bus = None
        self.feature_loader = None

        # Core services
        self.stt_service = None
        self.llm_service = None
        self.tts_service = None

        self.is_running = False

    async def initialize(self) -> None:
        """Initialize the voice assistant."""
        logger.info("Initializing Voice Assistant...")

        # Initialize event bus
        from src.core.event_bus import EventBus
        self.event_bus = await EventBus.get_instance()
        await self.event_bus.start()
        logger.info("Event bus started")

        # Initialize core services
        await self._init_services()

        # Initialize features
        if self.enabled_features:
            await self._init_features()

        logger.info("Voice Assistant initialized")

    async def _init_services(self) -> None:
        """Initialize core services."""
        from src.services import STTService, LLMService, TTSService

        # STT Service
        logger.info("Initializing STT service...")
        self.stt_service = STTService()
        await self.stt_service.initialize()

        # LLM Service
        logger.info("Initializing LLM service...")
        self.llm_service = LLMService()
        await self.llm_service.initialize()

        # TTS Service
        logger.info("Initializing TTS service...")
        self.tts_service = TTSService()
        await self.tts_service.initialize()

        logger.info("Core services initialized")

    async def _init_features(self) -> None:
        """Initialize enabled features."""
        from src.features import FeatureLoader

        self.feature_loader = FeatureLoader()
        self.feature_loader.load_features(self.enabled_features)
        await self.feature_loader.initialize_all()
        logger.info(f"Features initialized: {self.enabled_features}")

    async def start(self) -> None:
        """Start the voice assistant."""
        logger.info("Starting Voice Assistant...")

        # Start STT (this runs the main listening loop)
        asyncio.create_task(asyncio.to_thread(self.stt_service.start))

        # Start features
        if self.feature_loader:
            await self.feature_loader.start_all()

        self.is_running = True
        logger.info("Voice Assistant running. Say the wake word to begin...")

    async def stop(self) -> None:
        """Stop the voice assistant."""
        logger.info("Stopping Voice Assistant...")
        self.is_running = False

        # Stop STT
        if self.stt_service:
            self.stt_service.stop()

        # Stop features
        if self.feature_loader:
            await self.feature_loader.stop_all()

        # Stop event bus
        if self.event_bus:
            await self.event_bus.stop()

        logger.info("Voice Assistant stopped")

    async def run(self) -> None:
        """Run the voice assistant until interrupted."""
        await self.initialize()
        await self.start()

        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )


def list_features() -> None:
    """List available features."""
    from src.features import get_available_features
    print("\nAvailable Features:")
    print("-" * 40)
    for name in get_available_features():
        print(f"  - {name}")
    print()


async def async_main(features: List[str]) -> None:
    """Async main entry point."""
    assistant = VoiceAssistant(enabled_features=features)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(assistant.stop())

    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        await assistant.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await assistant.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cluster Voice Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with no features
  python main.py --features display       # Enable display feature
  python main.py --list-features          # Show available features
"""
    )
    parser.add_argument(
        '--features',
        nargs='*',
        default=[],
        help='Features to enable (e.g., display)'
    )
    parser.add_argument(
        '--list-features',
        action='store_true',
        help='List available features and exit'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # List features if requested
    if args.list_features:
        list_features()
        return

    logger.info("=" * 60)
    logger.info("CLUSTER VOICE ASSISTANT")
    logger.info("=" * 60)
    logger.info(f"Features: {args.features or 'none'}")
    logger.info("=" * 60)

    try:
        asyncio.run(async_main(args.features))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
