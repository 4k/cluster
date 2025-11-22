#!/usr/bin/env python3
"""
Test suite for Rhubarb lip sync integration with emotion_display_v2.

This test file verifies:
1. RhubarbVisemeController functionality
2. MouthRenderer Rhubarb shape handling
3. DisplayManager Rhubarb integration
4. Settings serialization
5. End-to-end lip sync simulation

Run with: python -m pytest tests/test_rhubarb_integration.py -v
Or standalone: python tests/test_rhubarb_integration.py
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_display_v2.rhubarb_controller import (
    RhubarbVisemeController,
    RhubarbControllerConfig,
    RhubarbShape,
    RhubarbVisemeCue,
    InterpolatedViseme,
    EasingFunction,
    apply_easing,
    RHUBARB_SHAPE_PARAMS,
    RHUBARB_TO_VISEME,
    COARTICULATION_PAIRS
)
from emotion_display_v2.renderers.mouth_renderer import (
    MouthRenderer,
    Viseme,
    RHUBARB_SHAPE_TO_VISEME,
    VISEME_SHAPES,
    EasingType,
    apply_easing as mouth_apply_easing
)
from emotion_display_v2.settings import (
    DisplaySettings,
    RhubarbSettings,
    RendererSettings
)
from emotion_display_v2.display_manager import DisplayManager


class TestRhubarbShapes(unittest.TestCase):
    """Test Rhubarb shape definitions and mappings."""

    def test_all_shapes_defined(self):
        """Verify all 9 Rhubarb shapes are defined."""
        expected_shapes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']
        for shape_letter in expected_shapes:
            shape = RhubarbShape(shape_letter)
            self.assertIn(shape, RHUBARB_SHAPE_PARAMS)
            self.assertIn(shape_letter, RHUBARB_SHAPE_TO_VISEME)

    def test_shape_params_have_required_fields(self):
        """Verify each shape has required mouth parameters."""
        required_fields = ['open', 'width', 'pucker', 'stretch']
        for shape, params in RHUBARB_SHAPE_PARAMS.items():
            for field in required_fields:
                self.assertIn(field, params, f"Shape {shape} missing {field}")

    def test_rhubarb_to_viseme_mapping(self):
        """Test Rhubarb shape to internal viseme mapping."""
        # Verify known mappings
        self.assertEqual(RHUBARB_SHAPE_TO_VISEME['A'], Viseme.BMP)
        self.assertEqual(RHUBARB_SHAPE_TO_VISEME['D'], Viseme.AH)
        self.assertEqual(RHUBARB_SHAPE_TO_VISEME['F'], Viseme.OO)
        self.assertEqual(RHUBARB_SHAPE_TO_VISEME['X'], Viseme.SILENCE)

    def test_shape_param_ranges(self):
        """Verify shape parameters are in valid ranges."""
        for shape, params in RHUBARB_SHAPE_PARAMS.items():
            self.assertGreaterEqual(params['open'], 0.0, f"{shape} open too low")
            self.assertLessEqual(params['open'], 1.0, f"{shape} open too high")
            self.assertGreaterEqual(params['width'], 0.0, f"{shape} width too low")
            self.assertLessEqual(params['width'], 1.0, f"{shape} width too high")


class TestEasingFunctions(unittest.TestCase):
    """Test easing function implementations."""

    def test_linear_easing(self):
        """Test linear easing."""
        self.assertAlmostEqual(apply_easing(0.0, EasingFunction.LINEAR), 0.0)
        self.assertAlmostEqual(apply_easing(0.5, EasingFunction.LINEAR), 0.5)
        self.assertAlmostEqual(apply_easing(1.0, EasingFunction.LINEAR), 1.0)

    def test_ease_in_out(self):
        """Test smooth step easing (most common for lip sync)."""
        # Ease in/out should be slower at edges, faster in middle
        self.assertAlmostEqual(apply_easing(0.0, EasingFunction.EASE_IN_OUT), 0.0)
        self.assertAlmostEqual(apply_easing(1.0, EasingFunction.EASE_IN_OUT), 1.0)
        # Midpoint should still be 0.5 for symmetric easing
        self.assertAlmostEqual(apply_easing(0.5, EasingFunction.EASE_IN_OUT), 0.5)

    def test_clamping(self):
        """Test that values are clamped to 0-1."""
        self.assertAlmostEqual(apply_easing(-0.5, EasingFunction.LINEAR), 0.0)
        self.assertAlmostEqual(apply_easing(1.5, EasingFunction.LINEAR), 1.0)

    def test_mouth_renderer_easing(self):
        """Test mouth renderer easing matches."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            controller_result = apply_easing(t, EasingFunction.EASE_IN_OUT)
            renderer_result = mouth_apply_easing(t, EasingType.EASE_IN_OUT)
            self.assertAlmostEqual(controller_result, renderer_result, places=5)


class TestRhubarbVisemeController(unittest.TestCase):
    """Test RhubarbVisemeController functionality."""

    def setUp(self):
        """Create controller for each test."""
        self.config = RhubarbControllerConfig(
            lookahead_ms=50.0,
            transition_duration_ms=60.0,
            enable_coarticulation=True,
            coarticulation_strength=0.3
        )
        self.controller = RhubarbVisemeController(self.config)

    def test_controller_creation(self):
        """Test controller initializes correctly."""
        self.assertFalse(self.controller.is_active())
        self.assertEqual(self.controller.get_progress(), 0.0)

    def test_load_lip_sync_data(self):
        """Test loading lip sync cue data."""
        cues = [
            {'start': 0.0, 'value': 'X'},
            {'start': 0.2, 'value': 'D'},
            {'start': 0.4, 'value': 'X'},
        ]
        self.controller.load_lip_sync_data(cues, 'test_session', 0.5)

        stats = self.controller.get_stats()
        self.assertEqual(stats['cue_count'], 3)
        self.assertEqual(stats['current_session'], 'test_session')

    def test_session_lifecycle(self):
        """Test session start/stop lifecycle."""
        cues = [{'start': 0.0, 'value': 'X'}]
        self.controller.load_lip_sync_data(cues, 'test', 1.0)

        self.assertFalse(self.controller.is_active())

        self.controller.start_session()
        self.assertTrue(self.controller.is_active())

        self.controller.stop_session()
        self.assertFalse(self.controller.is_active())

    def test_viseme_callback(self):
        """Test viseme callback is invoked."""
        callback_results = []

        def callback(viseme, params):
            callback_results.append((viseme, params))

        self.controller.set_viseme_callback(callback)

        cues = [
            {'start': 0.0, 'value': 'X'},
            {'start': 0.05, 'value': 'D'},
        ]
        self.controller.load_lip_sync_data(cues, 'test', 0.2)
        self.controller.start_session()

        # Run a few update cycles
        for _ in range(5):
            self.controller.update(0.033)
            time.sleep(0.01)

        # Should have received some callbacks
        self.assertGreater(len(callback_results), 0)

    def test_interpolated_viseme_params(self):
        """Test InterpolatedViseme produces valid params."""
        viseme = InterpolatedViseme(
            primary_shape=RhubarbShape.D,
            primary_weight=0.7,
            secondary_shape=RhubarbShape.C,
            secondary_weight=0.3,
            intensity=1.0
        )

        params = viseme.to_mouth_params()

        self.assertIn('open', params)
        self.assertIn('width', params)
        self.assertGreaterEqual(params['open'], 0.0)
        self.assertLessEqual(params['open'], 1.0)

    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        cues = [{'start': 0.0, 'value': 'D'}]
        self.controller.load_lip_sync_data(cues, 'test', 0.1)
        self.controller.start_session()

        stats = self.controller.get_stats()
        self.assertEqual(stats['sessions_processed'], 1)


class TestMouthRenderer(unittest.TestCase):
    """Test MouthRenderer Rhubarb integration."""

    def setUp(self):
        """Create renderer for each test."""
        self.renderer = MouthRenderer(800, 300)

    def test_renderer_creation(self):
        """Test renderer initializes with Rhubarb support."""
        self.assertFalse(self.renderer.mouth_state.rhubarb_active)
        self.assertIsNone(self.renderer.mouth_state.rhubarb_session_id)

    def test_set_rhubarb_shape(self):
        """Test setting Rhubarb shapes directly."""
        for shape in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']:
            self.renderer.set_rhubarb_shape(shape, intensity=1.0)
            self.assertEqual(self.renderer.mouth_state.mouth.rhubarb_shape, shape)
            self.assertTrue(self.renderer.mouth_state.rhubarb_active)

    def test_rhubarb_session_management(self):
        """Test Rhubarb session start/stop."""
        self.renderer.start_rhubarb_session('test_session')

        self.assertTrue(self.renderer.mouth_state.rhubarb_active)
        self.assertEqual(self.renderer.mouth_state.rhubarb_session_id, 'test_session')

        self.renderer.stop_rhubarb_session()

        self.assertFalse(self.renderer.mouth_state.rhubarb_active)
        self.assertIsNone(self.renderer.mouth_state.rhubarb_session_id)

    def test_viseme_with_params(self):
        """Test setting viseme with detailed parameters."""
        params = {
            'open': 0.8,
            'width': 0.6,
            'pucker': 0.0,
            'stretch': 0.1,
            'teeth_visible': True,
            'tongue_visible': False
        }

        self.renderer.set_rhubarb_shape('D', intensity=1.0, params=params)

        # After update, values should approach targets
        self.renderer.update(0.1)

        # Check that targets were set
        self.assertAlmostEqual(self.renderer._target_open, 0.8, places=1)

    def test_handle_rhubarb_command(self):
        """Test handling Rhubarb commands from decision module."""
        command = {
            'event': 'rhubarb_shape',
            'data': {
                'shape': 'D',
                'intensity': 0.9
            }
        }

        self.renderer.handle_command(command)

        self.assertEqual(self.renderer.mouth_state.mouth.rhubarb_shape, 'D')
        self.assertAlmostEqual(self.renderer.mouth_state.mouth.rhubarb_intensity, 0.9)

    def test_viseme_interpolation(self):
        """Test viseme interpolation over time."""
        self.renderer.set_rhubarb_shape('D', intensity=1.0)  # Wide open

        # Capture initial state
        initial_open = self.renderer.mouth_state.mouth.open_amount

        # Run several update cycles
        for _ in range(10):
            self.renderer.update(0.016)  # ~60 FPS

        # Should have progressed toward target
        final_open = self.renderer.mouth_state.mouth.open_amount
        self.assertGreater(final_open, initial_open)

    def test_statistics(self):
        """Test renderer statistics tracking."""
        self.renderer.set_rhubarb_shape('D')
        self.renderer.set_rhubarb_shape('C')
        self.renderer.set_rhubarb_shape('X')

        stats = self.renderer.get_stats()
        self.assertEqual(stats['rhubarb_updates'], 3)


class TestDisplayManager(unittest.TestCase):
    """Test DisplayManager Rhubarb integration."""

    def test_manager_creation_with_config(self):
        """Test creating DisplayManager with Rhubarb config."""
        config = RhubarbControllerConfig(
            lookahead_ms=75.0,
            enable_coarticulation=False
        )

        dm = DisplayManager(rhubarb_config=config)

        self.assertEqual(dm._rhubarb_config.lookahead_ms, 75.0)
        self.assertFalse(dm._rhubarb_config.enable_coarticulation)

    def test_rhubarb_stats_initialization(self):
        """Test Rhubarb stats are initialized."""
        dm = DisplayManager()

        self.assertEqual(dm._rhubarb_stats['sessions_started'], 0)
        self.assertEqual(dm._rhubarb_stats['sessions_completed'], 0)
        self.assertEqual(dm._rhubarb_stats['visemes_received'], 0)

    def test_get_state_includes_rhubarb(self):
        """Test get_state includes Rhubarb information."""
        dm = DisplayManager()
        state = dm.get_state()

        self.assertIn('rhubarb_lip_sync_active', state)
        self.assertIn('rhubarb_session_id', state)
        self.assertIn('rhubarb_stats', state)
        self.assertIn('rhubarb_controller_stats', state)

    def test_update_rhubarb_config(self):
        """Test updating Rhubarb config at runtime."""
        dm = DisplayManager()

        original_lookahead = dm._rhubarb_config.lookahead_ms
        dm.update_rhubarb_config(lookahead_ms=100.0)

        self.assertEqual(dm._rhubarb_config.lookahead_ms, 100.0)
        self.assertNotEqual(dm._rhubarb_config.lookahead_ms, original_lookahead)

    def test_get_rhubarb_config(self):
        """Test getting Rhubarb config."""
        config = RhubarbControllerConfig(intensity_scale=0.8)
        dm = DisplayManager(rhubarb_config=config)

        retrieved_config = dm.get_rhubarb_config()
        self.assertEqual(retrieved_config.intensity_scale, 0.8)


class TestRhubarbSettings(unittest.TestCase):
    """Test Rhubarb settings serialization."""

    def test_default_settings(self):
        """Test default Rhubarb settings values."""
        settings = RhubarbSettings()

        self.assertEqual(settings.lookahead_ms, 50.0)
        self.assertEqual(settings.transition_duration_ms, 60.0)
        self.assertTrue(settings.enable_coarticulation)
        self.assertEqual(settings.coarticulation_strength, 0.3)
        self.assertTrue(settings.use_extended_shapes)

    def test_display_settings_includes_rhubarb(self):
        """Test DisplaySettings includes RhubarbSettings."""
        settings = DisplaySettings()

        self.assertIsInstance(settings.rhubarb, RhubarbSettings)
        self.assertEqual(settings.rhubarb.lookahead_ms, 50.0)

    def test_settings_serialization(self):
        """Test settings can be serialized and deserialized."""
        original = DisplaySettings()
        original.rhubarb.lookahead_ms = 75.0
        original.rhubarb.coarticulation_strength = 0.5

        # Serialize
        data = original.to_dict()

        # Verify rhubarb section exists
        self.assertIn('rhubarb', data)
        self.assertEqual(data['rhubarb']['lookahead_ms'], 75.0)

        # Deserialize
        restored = DisplaySettings.from_dict(data)

        self.assertEqual(restored.rhubarb.lookahead_ms, 75.0)
        self.assertEqual(restored.rhubarb.coarticulation_strength, 0.5)

    def test_renderer_settings_rhubarb_fields(self):
        """Test RendererSettings includes Rhubarb fields."""
        settings = RendererSettings()

        self.assertEqual(settings.rhubarb_transition_speed, 18.0)
        self.assertTrue(settings.enable_coarticulation)


class TestCoarticulation(unittest.TestCase):
    """Test coarticulation (viseme blending) functionality."""

    def test_coarticulation_pairs_defined(self):
        """Test coarticulation pairs are defined."""
        self.assertGreater(len(COARTICULATION_PAIRS), 0)

        # Check some expected smooth transitions
        self.assertIn((RhubarbShape.B, RhubarbShape.C), COARTICULATION_PAIRS)
        self.assertIn((RhubarbShape.C, RhubarbShape.D), COARTICULATION_PAIRS)

    def test_coarticulation_weights(self):
        """Test coarticulation weights are in valid range."""
        for pair, weight in COARTICULATION_PAIRS.items():
            self.assertGreaterEqual(weight, 0.0, f"Pair {pair} weight too low")
            self.assertLessEqual(weight, 1.0, f"Pair {pair} weight too high")


class TestEndToEndSimulation(unittest.TestCase):
    """End-to-end simulation tests."""

    def test_hello_world_lip_sync(self):
        """Simulate lip sync for 'Hello World'."""
        controller = RhubarbVisemeController(RhubarbControllerConfig())

        # Simulated Rhubarb output for "Hello World" - shorter duration for test
        cues = [
            {'start': 0.00, 'value': 'X'},  # Rest
            {'start': 0.02, 'value': 'H'},  # H
            {'start': 0.04, 'value': 'C'},  # E
            {'start': 0.06, 'value': 'H'},  # L
            {'start': 0.08, 'value': 'E'},  # O
            {'start': 0.10, 'value': 'X'},  # Space
            {'start': 0.12, 'value': 'F'},  # W
            {'start': 0.14, 'value': 'E'},  # O
            {'start': 0.16, 'value': 'B'},  # R
            {'start': 0.18, 'value': 'H'},  # L
            {'start': 0.20, 'value': 'X'},  # End
        ]

        controller.load_lip_sync_data(cues, 'hello_world', 0.25)
        controller.start_session()

        viseme_changes = []
        def capture_viseme(name, params):
            viseme_changes.append((name, params.get('open', 0)))

        controller.set_viseme_callback(capture_viseme)

        # Simulate playback - run enough iterations to complete
        max_iterations = 100
        for _ in range(max_iterations):
            if not controller.is_active():
                break
            controller.update(0.016)
            time.sleep(0.005)

        # Verify we got viseme updates
        self.assertGreater(len(viseme_changes), 0, "Should have received viseme updates")

        stats = controller.get_stats()
        self.assertGreater(stats['visemes_displayed'], 0)

    def test_renderer_shape_sequence(self):
        """Test renderer handles rapid shape changes."""
        renderer = MouthRenderer(800, 300)
        renderer.start_rhubarb_session('sequence_test')

        # Rapid shape sequence
        shapes = ['X', 'D', 'C', 'B', 'F', 'E', 'D', 'X']

        for shape in shapes:
            renderer.set_rhubarb_shape(shape, intensity=1.0)
            # Simulate a few frames
            for _ in range(3):
                renderer.update(0.016)

        renderer.stop_rhubarb_session()

        stats = renderer.get_stats()
        self.assertEqual(stats['rhubarb_updates'], len(shapes))


def run_interactive_test():
    """Run an interactive test with visual output (requires pygame)."""
    print("\n" + "="*60)
    print("INTERACTIVE RHUBARB INTEGRATION TEST")
    print("="*60)

    # Test 1: Shape definitions
    print("\n1. Testing Rhubarb shape definitions...")
    for shape in RhubarbShape:
        params = RHUBARB_SHAPE_PARAMS[shape]
        viseme = RHUBARB_TO_VISEME[shape]
        print(f"   {shape.value}: open={params['open']:.2f}, width={params['width']:.2f} -> {viseme}")
    print("   PASS")

    # Test 2: Controller
    print("\n2. Testing RhubarbVisemeController...")
    config = RhubarbControllerConfig(
        lookahead_ms=50.0,
        enable_coarticulation=True
    )
    controller = RhubarbVisemeController(config)

    cues = [
        {'start': 0.0, 'value': 'X'},
        {'start': 0.1, 'value': 'D'},
        {'start': 0.3, 'value': 'C'},
        {'start': 0.5, 'value': 'X'},
    ]
    controller.load_lip_sync_data(cues, 'test', 0.6)
    print(f"   Loaded {len(cues)} cues")

    controller.start_session()
    print(f"   Session active: {controller.is_active()}")

    updates = 0
    while controller.is_active():
        result = controller.update(0.033)
        if result:
            updates += 1
        time.sleep(0.01)

    print(f"   Generated {updates} viseme updates")
    print(f"   Stats: {controller.get_stats()}")
    print("   PASS")

    # Test 3: MouthRenderer
    print("\n3. Testing MouthRenderer...")
    renderer = MouthRenderer(800, 300)

    renderer.start_rhubarb_session('render_test')
    for shape in ['A', 'B', 'C', 'D', 'E', 'F']:
        renderer.set_rhubarb_shape(shape)
        renderer.update(0.05)
    renderer.stop_rhubarb_session()

    print(f"   Stats: {renderer.get_stats()}")
    print("   PASS")

    # Test 4: DisplayManager
    print("\n4. Testing DisplayManager...")
    dm = DisplayManager(rhubarb_config=config)
    print(f"   Rhubarb config: lookahead={dm._rhubarb_config.lookahead_ms}ms")
    print(f"   State: {dm.get_state()['rhubarb_stats']}")
    print("   PASS")

    # Test 5: Settings serialization
    print("\n5. Testing settings serialization...")
    settings = DisplaySettings()
    settings.rhubarb.lookahead_ms = 75.0
    data = settings.to_dict()
    restored = DisplaySettings.from_dict(data)
    assert restored.rhubarb.lookahead_ms == 75.0
    print(f"   Serialization round-trip: OK")
    print("   PASS")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Rhubarb integration')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive test with visual output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose unittest output')

    args = parser.parse_args()

    if args.interactive:
        run_interactive_test()
    else:
        # Run unittest suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])

        verbosity = 2 if args.verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)
