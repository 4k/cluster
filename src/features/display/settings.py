"""
Settings and configuration for the multi-window emotion display system.
Defines window sizes, positions, content subscriptions, and rendering parameters.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple
import logging

from src.core.service_config import load_display_config

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be rendered in windows."""
    EYES = "eyes"
    MOUTH = "mouth"
    FACE_FULL = "face_full"
    STATUS = "status"
    DEBUG = "debug"


class WindowType(Enum):
    """Types of display windows."""
    EYE_WINDOW = "eye_window"
    MOUTH_WINDOW = "mouth_window"
    COMBINED_WINDOW = "combined_window"
    STATUS_WINDOW = "status_window"
    DEBUG_WINDOW = "debug_window"


@dataclass
class WindowSettings:
    """Settings for a single display window."""
    window_type: WindowType
    title: str = "Display Window"
    width: int = 800
    height: int = 400
    position_x: int = 100
    position_y: int = 100
    fps: int = 60
    background_color: Tuple[int, int, int] = (20, 20, 25)
    subscriptions: List[ContentType] = field(default_factory=list)
    enabled: bool = True
    always_on_top: bool = False
    borderless: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            'window_type': self.window_type.value,
            'title': self.title,
            'width': self.width,
            'height': self.height,
            'position_x': self.position_x,
            'position_y': self.position_y,
            'fps': self.fps,
            'background_color': self.background_color,
            'subscriptions': [s.value for s in self.subscriptions],
            'enabled': self.enabled,
            'always_on_top': self.always_on_top,
            'borderless': self.borderless
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowSettings':
        """Create settings from dictionary."""
        data = data.copy()
        data['window_type'] = WindowType(data['window_type'])
        data['subscriptions'] = [ContentType(s) for s in data.get('subscriptions', [])]
        data['background_color'] = tuple(data.get('background_color', (20, 20, 25)))
        return cls(**data)


@dataclass
class RhubarbSettings:
    """
    Rhubarb Lip Sync integration settings.

    Based on best practices from Rhubarb documentation and animation industry standards.
    """
    # Timing adjustments
    lookahead_ms: float = 50.0      # Show visemes slightly early (ms) for natural look
    hold_minimum_ms: float = 40.0   # Minimum viseme hold time (ms)

    # Interpolation settings
    transition_duration_ms: float = 60.0  # Time to transition between visemes
    easing_function: str = "ease_in_out"  # linear, ease_in, ease_out, ease_in_out

    # Coarticulation settings (blending adjacent visemes)
    enable_coarticulation: bool = True
    coarticulation_window_ms: float = 100.0  # Time window for coarticulation
    coarticulation_strength: float = 0.3     # Blend strength (0-1)

    # Extended shapes
    use_extended_shapes: bool = True  # Use Rhubarb G, H, X shapes

    # Animation quality
    update_rate_hz: float = 60.0     # Update frequency for smooth animation

    # Intensity
    intensity_scale: float = 1.0     # Overall mouth movement intensity (0-1)
    prefer_wide_mouth: bool = True   # Use shape D more often (more lively)


@dataclass
class RendererSettings:
    """Settings for content renderers."""
    # Eye renderer settings
    eye_iris_color: Tuple[int, int, int] = (100, 150, 200)
    eye_pupil_color: Tuple[int, int, int] = (20, 20, 30)
    eye_sclera_color: Tuple[int, int, int] = (240, 240, 245)
    eye_blink_duration: float = 0.15
    eye_blink_interval_min: float = 2.0
    eye_blink_interval_max: float = 6.0
    eye_gaze_smoothing: float = 0.1
    eye_max_gaze_offset: float = 0.3

    # Mouth renderer settings
    mouth_lip_color: Tuple[int, int, int] = (180, 100, 100)
    mouth_interior_color: Tuple[int, int, int] = (60, 30, 40)
    mouth_teeth_color: Tuple[int, int, int] = (240, 240, 235)
    mouth_transition_speed: float = 12.0
    mouth_idle_movement: bool = True

    # Rhubarb-specific mouth settings
    rhubarb_transition_speed: float = 18.0  # Faster for Rhubarb
    enable_coarticulation: bool = True

    # General animation settings
    animation_smoothing: float = 0.15
    emotion_transition_speed: float = 2.0


@dataclass
class DisplaySettings:
    """Complete settings for the display system with Rhubarb integration."""
    # Window configurations
    windows: Dict[str, WindowSettings] = field(default_factory=dict)

    # Renderer settings
    renderer: RendererSettings = field(default_factory=RendererSettings)

    # Rhubarb lip sync settings
    rhubarb: RhubarbSettings = field(default_factory=RhubarbSettings)

    # Event bus connection
    connect_event_bus: bool = True
    event_bus_retry_attempts: int = 3
    event_bus_retry_delay: float = 1.0

    # Performance settings
    target_fps: int = 60
    enable_vsync: bool = True

    # Debug settings
    show_fps: bool = False
    show_debug_info: bool = False
    log_events: bool = False

    def __post_init__(self):
        """Initialize default windows if none provided."""
        if not self.windows:
            self.windows = self.get_default_dual_window_config()

    @staticmethod
    def get_default_dual_window_config() -> Dict[str, WindowSettings]:
        """Get default configuration for dual window setup (eyes + mouth)."""
        return {
            'eyes': WindowSettings(
                window_type=WindowType.EYE_WINDOW,
                title="Emotion Display - Eyes",
                width=800,
                height=400,
                position_x=100,
                position_y=100,
                subscriptions=[ContentType.EYES]
            ),
            'mouth': WindowSettings(
                window_type=WindowType.MOUTH_WINDOW,
                title="Emotion Display - Mouth",
                width=800,
                height=300,
                position_x=100,
                position_y=550,
                subscriptions=[ContentType.MOUTH]
            )
        }

    @staticmethod
    def get_single_window_config() -> Dict[str, WindowSettings]:
        """Get configuration for single combined window."""
        return {
            'combined': WindowSettings(
                window_type=WindowType.COMBINED_WINDOW,
                title="Emotion Display",
                width=800,
                height=700,
                position_x=100,
                position_y=100,
                subscriptions=[ContentType.EYES, ContentType.MOUTH]
            )
        }

    @staticmethod
    def get_debug_window_config() -> Dict[str, WindowSettings]:
        """Get configuration with debug window."""
        config = DisplaySettings.get_default_dual_window_config()
        config['debug'] = WindowSettings(
            window_type=WindowType.DEBUG_WINDOW,
            title="Emotion Display - Debug",
            width=400,
            height=300,
            position_x=920,
            position_y=100,
            subscriptions=[ContentType.DEBUG]
        )
        return config

    def get_enabled_windows(self) -> Dict[str, WindowSettings]:
        """Get only enabled windows."""
        return {
            name: settings
            for name, settings in self.windows.items()
            if settings.enabled
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            'windows': {
                name: settings.to_dict()
                for name, settings in self.windows.items()
            },
            'renderer': {
                'eye_iris_color': self.renderer.eye_iris_color,
                'eye_pupil_color': self.renderer.eye_pupil_color,
                'eye_sclera_color': self.renderer.eye_sclera_color,
                'eye_blink_duration': self.renderer.eye_blink_duration,
                'eye_blink_interval_min': self.renderer.eye_blink_interval_min,
                'eye_blink_interval_max': self.renderer.eye_blink_interval_max,
                'eye_gaze_smoothing': self.renderer.eye_gaze_smoothing,
                'eye_max_gaze_offset': self.renderer.eye_max_gaze_offset,
                'mouth_lip_color': self.renderer.mouth_lip_color,
                'mouth_interior_color': self.renderer.mouth_interior_color,
                'mouth_teeth_color': self.renderer.mouth_teeth_color,
                'mouth_transition_speed': self.renderer.mouth_transition_speed,
                'mouth_idle_movement': self.renderer.mouth_idle_movement,
                'rhubarb_transition_speed': self.renderer.rhubarb_transition_speed,
                'enable_coarticulation': self.renderer.enable_coarticulation,
                'animation_smoothing': self.renderer.animation_smoothing,
                'emotion_transition_speed': self.renderer.emotion_transition_speed
            },
            'rhubarb': {
                'lookahead_ms': self.rhubarb.lookahead_ms,
                'hold_minimum_ms': self.rhubarb.hold_minimum_ms,
                'transition_duration_ms': self.rhubarb.transition_duration_ms,
                'easing_function': self.rhubarb.easing_function,
                'enable_coarticulation': self.rhubarb.enable_coarticulation,
                'coarticulation_window_ms': self.rhubarb.coarticulation_window_ms,
                'coarticulation_strength': self.rhubarb.coarticulation_strength,
                'use_extended_shapes': self.rhubarb.use_extended_shapes,
                'update_rate_hz': self.rhubarb.update_rate_hz,
                'intensity_scale': self.rhubarb.intensity_scale,
                'prefer_wide_mouth': self.rhubarb.prefer_wide_mouth
            },
            'connect_event_bus': self.connect_event_bus,
            'event_bus_retry_attempts': self.event_bus_retry_attempts,
            'event_bus_retry_delay': self.event_bus_retry_delay,
            'target_fps': self.target_fps,
            'enable_vsync': self.enable_vsync,
            'show_fps': self.show_fps,
            'show_debug_info': self.show_debug_info,
            'log_events': self.log_events
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisplaySettings':
        """Create settings from dictionary including Rhubarb configuration."""
        windows = {
            name: WindowSettings.from_dict(settings_data)
            for name, settings_data in data.get('windows', {}).items()
        }

        renderer_data = data.get('renderer', {})
        renderer = RendererSettings(
            eye_iris_color=tuple(renderer_data.get('eye_iris_color', (100, 150, 200))),
            eye_pupil_color=tuple(renderer_data.get('eye_pupil_color', (20, 20, 30))),
            eye_sclera_color=tuple(renderer_data.get('eye_sclera_color', (240, 240, 245))),
            eye_blink_duration=renderer_data.get('eye_blink_duration', 0.15),
            eye_blink_interval_min=renderer_data.get('eye_blink_interval_min', 2.0),
            eye_blink_interval_max=renderer_data.get('eye_blink_interval_max', 6.0),
            eye_gaze_smoothing=renderer_data.get('eye_gaze_smoothing', 0.1),
            eye_max_gaze_offset=renderer_data.get('eye_max_gaze_offset', 0.3),
            mouth_lip_color=tuple(renderer_data.get('mouth_lip_color', (180, 100, 100))),
            mouth_interior_color=tuple(renderer_data.get('mouth_interior_color', (60, 30, 40))),
            mouth_teeth_color=tuple(renderer_data.get('mouth_teeth_color', (240, 240, 235))),
            mouth_transition_speed=renderer_data.get('mouth_transition_speed', 12.0),
            mouth_idle_movement=renderer_data.get('mouth_idle_movement', True),
            rhubarb_transition_speed=renderer_data.get('rhubarb_transition_speed', 18.0),
            enable_coarticulation=renderer_data.get('enable_coarticulation', True),
            animation_smoothing=renderer_data.get('animation_smoothing', 0.15),
            emotion_transition_speed=renderer_data.get('emotion_transition_speed', 2.0)
        )

        # Parse Rhubarb settings
        rhubarb_data = data.get('rhubarb', {})
        rhubarb = RhubarbSettings(
            lookahead_ms=rhubarb_data.get('lookahead_ms', 50.0),
            hold_minimum_ms=rhubarb_data.get('hold_minimum_ms', 40.0),
            transition_duration_ms=rhubarb_data.get('transition_duration_ms', 60.0),
            easing_function=rhubarb_data.get('easing_function', 'ease_in_out'),
            enable_coarticulation=rhubarb_data.get('enable_coarticulation', True),
            coarticulation_window_ms=rhubarb_data.get('coarticulation_window_ms', 100.0),
            coarticulation_strength=rhubarb_data.get('coarticulation_strength', 0.3),
            use_extended_shapes=rhubarb_data.get('use_extended_shapes', True),
            update_rate_hz=rhubarb_data.get('update_rate_hz', 60.0),
            intensity_scale=rhubarb_data.get('intensity_scale', 1.0),
            prefer_wide_mouth=rhubarb_data.get('prefer_wide_mouth', True)
        )

        return cls(
            windows=windows,
            renderer=renderer,
            rhubarb=rhubarb,
            connect_event_bus=data.get('connect_event_bus', True),
            event_bus_retry_attempts=data.get('event_bus_retry_attempts', 3),
            event_bus_retry_delay=data.get('event_bus_retry_delay', 1.0),
            target_fps=data.get('target_fps', 60),
            enable_vsync=data.get('enable_vsync', True),
            show_fps=data.get('show_fps', False),
            show_debug_info=data.get('show_debug_info', False),
            log_events=data.get('log_events', False)
        )

    @classmethod
    def load(cls) -> 'DisplaySettings':
        """
        Load display settings from config/display.yaml.

        Returns:
            DisplaySettings instance with values from config file,
            or defaults if file not found.
        """
        try:
            config_data = load_display_config()
            if config_data:
                logger.info("Loaded display settings from config/display.yaml")
                return cls.from_dict(config_data)
            else:
                logger.info("No display config found, using defaults")
                return cls()
        except Exception as e:
            logger.warning(f"Error loading display config: {e}, using defaults")
            return cls()
