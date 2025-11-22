#!/usr/bin/env python3
"""
Rhubarb Lip Sync Visual Demo

Demonstrates the Rhubarb lip sync integration with animated mouth rendering.
Shows all Preston Blair mouth shapes (A-F, G-H-X) with smooth transitions.

Run with: python demo_rhubarb.py
"""

import sys
import time

try:
    import pygame
    pygame.init()
except ImportError:
    print("pygame not installed - run: pip install pygame")
    sys.exit(1)

from emotion_display_v2.renderers.mouth_renderer import MouthRenderer
from emotion_display_v2.rhubarb_controller import RhubarbVisemeController, RhubarbControllerConfig


def main():
    # Create window
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("Rhubarb Lip Sync Demo - Press SPACE to replay, Q to quit")

    # Setup renderer and controller
    renderer = MouthRenderer(800, 400)
    config = RhubarbControllerConfig(
        lookahead_ms=50.0,
        transition_duration_ms=60.0,
        enable_coarticulation=True,
        intensity_scale=1.0
    )
    controller = RhubarbVisemeController(config)

    # Simulated Rhubarb output for "Hello World"
    # These timings simulate what Rhubarb would generate from audio
    hello_world_cues = [
        {'start': 0.00, 'value': 'X'},   # Rest
        {'start': 0.08, 'value': 'C'},   # H - open mouth
        {'start': 0.18, 'value': 'C'},   # E - mid vowel
        {'start': 0.30, 'value': 'H'},   # L - tongue
        {'start': 0.42, 'value': 'H'},   # L - tongue
        {'start': 0.54, 'value': 'E'},   # O - rounded
        {'start': 0.70, 'value': 'X'},   # (pause)
        {'start': 0.85, 'value': 'F'},   # W - puckered
        {'start': 0.98, 'value': 'E'},   # O - rounded
        {'start': 1.12, 'value': 'B'},   # R - slightly open
        {'start': 1.25, 'value': 'H'},   # L - tongue
        {'start': 1.38, 'value': 'B'},   # D - slightly open
        {'start': 1.50, 'value': 'X'},   # Rest
    ]

    # Alternative: Demo all shapes
    all_shapes_cues = [
        {'start': 0.0, 'value': 'X'},   # Rest
        {'start': 0.4, 'value': 'A'},   # Closed (M, B, P)
        {'start': 0.8, 'value': 'B'},   # Slightly open
        {'start': 1.2, 'value': 'C'},   # Open (EH vowel)
        {'start': 1.6, 'value': 'D'},   # Wide open (AH)
        {'start': 2.0, 'value': 'E'},   # Rounded (OH)
        {'start': 2.4, 'value': 'F'},   # Puckered (OO, W)
        {'start': 2.8, 'value': 'G'},   # F/V (teeth on lip)
        {'start': 3.2, 'value': 'H'},   # L (tongue)
        {'start': 3.6, 'value': 'X'},   # Rest
    ]

    # Callback to update renderer
    def on_viseme(name, params):
        renderer.set_rhubarb_shape(name, params.get('intensity', 1.0), params)

    controller.set_viseme_callback(on_viseme)

    # State
    current_demo = 0
    demos = [
        ("Hello World", hello_world_cues, 1.8),
        ("All Shapes (A-H, X)", all_shapes_cues, 4.0),
    ]

    def start_demo(index):
        nonlocal current_demo
        current_demo = index % len(demos)
        name, cues, duration = demos[current_demo]
        print(f"\nPlaying: {name}")
        controller.load_lip_sync_data(cues, name, duration)
        renderer.start_rhubarb_session(name)
        controller.start_session()

    # Start first demo
    start_demo(0)

    clock = pygame.time.Clock()
    running = True
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    print("\nControls:")
    print("  SPACE - Replay current demo")
    print("  N     - Next demo")
    print("  Q     - Quit")

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    start_demo(current_demo)
                elif event.key == pygame.K_n:
                    start_demo(current_demo + 1)

        # Update animation
        controller.update(dt)
        renderer.update(dt)
        renderer.render()

        # Draw to screen
        screen.blit(renderer.surface, (0, 0))

        # Draw info overlay
        demo_name = demos[current_demo][0]
        shape = renderer.mouth_state.mouth.rhubarb_shape or "X"
        active = "Playing" if controller.is_active() else "Stopped"

        # Status text
        status_text = font.render(f"{demo_name} - {active}", True, (200, 200, 200))
        screen.blit(status_text, (20, 20))

        # Current shape
        shape_text = font.render(f"Shape: {shape}", True, (100, 200, 100))
        screen.blit(shape_text, (20, 60))

        # Mouth params
        mouth = renderer.mouth_state.mouth
        params_text = small_font.render(
            f"open={mouth.open_amount:.2f} width={mouth.width:.2f} "
            f"pucker={mouth.pucker:.2f} stretch={mouth.stretch:.2f}",
            True, (150, 150, 150)
        )
        screen.blit(params_text, (20, 370))

        # Controls hint
        hint_text = small_font.render("SPACE=replay  N=next  Q=quit", True, (100, 100, 100))
        screen.blit(hint_text, (580, 370))

        pygame.display.flip()

        # Auto-advance after demo completes
        if not controller.is_active():
            time.sleep(0.5)

    pygame.quit()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
