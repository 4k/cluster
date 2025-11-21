#!/usr/bin/env python3
"""
Demo script for the dual-window facial animation system.

This script demonstrates:
1. Creating a DisplayManager with dual-window configuration
2. Subscribing windows to different content types (eyes, mouth)
3. Animating the face through the event bus
4. The centralized DisplayDecisionModule routing content

Run this script to see two windows - one showing eyes and one showing mouth,
synchronized through the event bus.

Usage:
    python src/demo_dual_window.py

Environment variables:
    DISPLAY_EYES_POSITION=100,100     # Position of eyes window
    DISPLAY_MOUTH_POSITION=100,600    # Position of mouth window
    DISPLAY_RESOLUTION=800,400        # Default window size
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.event_bus import EventBus, EventType
from core.types import (
    ContentType, WindowConfig, EmotionType, GazeDirection, AnimationState
)
from display.simple_display_manager import DisplayManager, DisplayConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_animations(event_bus: EventBus):
    """Run demo animations through the event bus."""

    logger.info("Starting demo animations...")

    # Demo sequence
    animations = [
        # Wake up
        ("Wake word detected", EventType.WAKE_WORD_DETECTED, {"wake_word": "hey assistant"}),
        (2.0, None, None),

        # Listen and process
        ("Speech detected", EventType.SPEECH_DETECTED, {"text": "Hello, how are you?", "confidence": 0.95}),
        (1.5, None, None),

        # Start speaking
        ("TTS started", EventType.TTS_STARTED, {"text": "I'm doing great, thank you for asking!"}),
        (3.0, None, None),

        # Emotion changes while speaking
        ("Express happiness", EventType.EMOTION_CHANGED, {"emotion": "happy"}),
        (1.5, None, None),

        # Gaze changes
        ("Look left", EventType.GAZE_UPDATE, {"direction": "left"}),
        (1.0, None, None),
        ("Look right", EventType.GAZE_UPDATE, {"direction": "right"}),
        (1.0, None, None),
        ("Look forward", EventType.GAZE_UPDATE, {"direction": "forward"}),
        (1.0, None, None),

        # Finish speaking
        ("TTS completed", EventType.TTS_COMPLETED, {}),
        (1.5, None, None),

        # Show different emotions
        ("Express surprise", EventType.EMOTION_CHANGED, {"emotion": "surprised"}),
        (1.5, None, None),
        ("Express thinking", EventType.EMOTION_CHANGED, {"emotion": "thinking"}),
        (1.5, None, None),
        ("Express confused", EventType.EMOTION_CHANGED, {"emotion": "confused"}),
        (1.5, None, None),

        # Trigger blink
        ("Trigger blink", EventType.BLINK_TRIGGERED, {}),
        (0.5, None, None),

        # Return to neutral
        ("Return to neutral", EventType.EMOTION_CHANGED, {"emotion": "neutral"}),
        (2.0, None, None),

        # Show error state briefly
        ("Show error", EventType.ERROR_OCCURRED, {"error": "Demo error", "component": "demo"}),
        (2.0, None, None),

        # Final idle
        ("Return to idle", EventType.EMOTION_CHANGED, {"emotion": "neutral"}),
    ]

    for item in animations:
        if isinstance(item[0], float):
            # Delay
            await asyncio.sleep(item[0])
        else:
            # Event
            description, event_type, data = item
            logger.info(f"Demo: {description}")
            await event_bus.emit(event_type, data, source="demo")
            await asyncio.sleep(0.1)

    logger.info("Demo animations complete!")


async def main():
    """Main demo entry point."""
    logger.info("=" * 60)
    logger.info("Dual-Window Facial Animation Demo")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This demo shows two synchronized windows:")
    logger.info("  - Eyes window: Shows animated eyes with gaze and blinking")
    logger.info("  - Mouth window: Shows animated mouth with lip-sync")
    logger.info("")
    logger.info("The windows are synchronized through the event bus.")
    logger.info("The DisplayDecisionModule routes content to each window")
    logger.info("based on what content type each window subscribed to.")
    logger.info("")
    logger.info("Press Ctrl+C to exit")
    logger.info("=" * 60)

    # Initialize event bus
    event_bus = await EventBus.get_instance()
    await event_bus.start()

    # Create display configuration for dual windows
    config = DisplayConfig(
        mode="dual_display",
        resolution=(800, 400),
        fps=60,
        development_mode=True,
        window_positions={
            'eyes': (100, 100),
            'mouth': (100, 550),
        },
        fullscreen=False,
        borderless=False,
    )

    # You can also create custom window configurations:
    # config = DisplayConfig(
    #     mode="multi",
    #     windows=[
    #         WindowConfig(
    #             name="left_eye",
    #             title="Left Eye",
    #             content_type=ContentType.EYES,
    #             position=(100, 100),
    #             size=(400, 400),
    #         ),
    #         WindowConfig(
    #             name="right_eye",
    #             title="Right Eye",
    #             content_type=ContentType.EYES,
    #             position=(550, 100),
    #             size=(400, 400),
    #         ),
    #         WindowConfig(
    #             name="mouth",
    #             title="Mouth",
    #             content_type=ContentType.MOUTH,
    #             position=(100, 550),
    #             size=(850, 300),
    #         ),
    #     ]
    # )

    # Create and initialize display manager
    display_manager = DisplayManager(config)
    await display_manager.initialize()
    await display_manager.start()

    logger.info("Display windows created. Starting demo animations...")

    try:
        # Run demo animations
        await demo_animations(event_bus)

        # Keep running until Ctrl+C
        logger.info("")
        logger.info("Demo complete. Windows will stay open.")
        logger.info("Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")

    finally:
        # Cleanup
        await display_manager.cleanup()
        await event_bus.stop()

    logger.info("Demo finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
