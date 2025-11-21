#!/usr/bin/env python3
"""
Demo script for the Multi-Window Emotion Display System.

This demonstrates the dual-window facial animation with:
- Eyes window with blinking, gaze tracking, and emotional expressions
- Mouth window with lip-sync animation and visemes
- Synchronized via event bus and centralized decision module

Usage:
    python -m emotion_display_v2.demo
    python -m emotion_display_v2.demo --single    # Single combined window
    python -m emotion_display_v2.demo --debug     # Include debug window
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_display_v2 import (
    DisplayManager,
    DisplaySettings,
    WindowSettings,
    ContentType,
    WindowType
)


async def run_demo(mode: str = 'dual', connect_event_bus: bool = False):
    """
    Run the emotion display demo.

    Args:
        mode: Display mode ('dual', 'single', 'debug')
        connect_event_bus: Whether to connect to the event bus
    """
    # Create settings based on mode
    if mode == 'single':
        windows = DisplaySettings.get_single_window_config()
        print("Running in single-window mode")
    elif mode == 'debug':
        windows = DisplaySettings.get_debug_window_config()
        print("Running with debug window")
    else:
        windows = DisplaySettings.get_default_dual_window_config()
        print("Running in dual-window mode (eyes + mouth)")

    settings = DisplaySettings(
        windows=windows,
        connect_event_bus=connect_event_bus,
        show_fps=True
    )

    # Create and start display manager
    manager = DisplayManager(settings)

    try:
        await manager.initialize()
        await manager.start()

        print("\n" + "=" * 50)
        print("Multi-Window Emotion Display Demo")
        print("=" * 50)
        print("\nControls (in any window):")
        print("  SPACE - Trigger blink")
        print("  S     - Start speaking animation")
        print("  X     - Stop speaking animation")
        print("  T     - Test text animation ('Hello, how are you today?')")
        print("  A     - Angry emotion")
        print("  N     - Neutral emotion")
        print("  ESC   - Close window")
        print("\n" + "=" * 50)
        print(f"Active windows: {manager.get_window_ids()}")
        print("=" * 50 + "\n")

        # Demo sequence (optional)
        if mode == 'demo':
            await run_demo_sequence(manager)

        # Wait for windows to close
        while manager.any_window_alive():
            await asyncio.sleep(0.1)

        print("\nAll windows closed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await manager.stop()
        print("Demo ended.")


async def run_demo_sequence(manager: DisplayManager):
    """Run an automated demo sequence."""
    print("\nStarting demo sequence...")

    # Wait for windows to initialize
    await asyncio.sleep(1.0)

    # Demo emotions
    emotions = ['HAPPY', 'SAD', 'SURPRISED', 'THINKING', 'NEUTRAL']
    for emotion in emotions:
        print(f"  Setting emotion: {emotion}")
        manager.set_emotion(emotion)
        await asyncio.sleep(2.0)

    # Demo speaking
    print("  Starting speaking animation...")
    manager.speak_text("Hello! I am the emotion display demo. I can show various emotions and lip-sync to text.")
    await asyncio.sleep(5.0)
    manager.stop_speaking()

    # Demo blinking
    print("  Triggering blinks...")
    for _ in range(3):
        manager.trigger_blink()
        await asyncio.sleep(0.5)

    # Demo gaze
    print("  Moving gaze...")
    positions = [(0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8), (0.5, 0.5)]
    for x, y in positions:
        manager.set_gaze(x, y)
        await asyncio.sleep(1.0)

    print("Demo sequence complete!\n")


async def interactive_demo(manager: DisplayManager):
    """Run an interactive demo with command input."""
    print("\nInteractive mode. Commands:")
    print("  blink - Trigger blink")
    print("  speak <text> - Speak text")
    print("  stop - Stop speaking")
    print("  emotion <name> - Set emotion")
    print("  gaze <x> <y> - Set gaze (0-1)")
    print("  state <name> - Set animation state")
    print("  quit - Exit")

    while manager.any_window_alive():
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, input, "> "
            )
            parts = line.strip().split()
            if not parts:
                continue

            cmd = parts[0].lower()

            if cmd == 'quit':
                break
            elif cmd == 'blink':
                manager.trigger_blink()
            elif cmd == 'speak' and len(parts) > 1:
                text = ' '.join(parts[1:])
                manager.speak_text(text)
            elif cmd == 'stop':
                manager.stop_speaking()
            elif cmd == 'emotion' and len(parts) > 1:
                manager.set_emotion(parts[1].upper())
            elif cmd == 'gaze' and len(parts) >= 3:
                x, y = float(parts[1]), float(parts[2])
                manager.set_gaze(x, y)
            elif cmd == 'state' and len(parts) > 1:
                manager.set_animation_state(parts[1].upper())
            else:
                print(f"Unknown command: {cmd}")

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Window Emotion Display Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m emotion_display_v2.demo           # Default dual-window mode
  python -m emotion_display_v2.demo --single  # Single combined window
  python -m emotion_display_v2.demo --debug   # Include debug window
  python -m emotion_display_v2.demo --demo    # Run automated demo sequence
        """
    )

    parser.add_argument(
        '--single', action='store_true',
        help='Use single combined window instead of dual windows'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Include debug window'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run automated demo sequence'
    )
    parser.add_argument(
        '--event-bus', action='store_true',
        help='Connect to event bus'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine mode
    if args.single:
        mode = 'single'
    elif args.debug:
        mode = 'debug'
    elif args.demo:
        mode = 'demo'
    else:
        mode = 'dual'

    # Run demo
    asyncio.run(run_demo(mode, connect_event_bus=args.event_bus))


if __name__ == '__main__':
    main()
