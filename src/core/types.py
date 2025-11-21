"""
Core type definitions for the voice assistant system.
Includes emotion types, animation states, and configuration dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class EmotionType(Enum):
    """Emotion states for the assistant."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    INTERESTED = "interested"
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"
    ERROR = "error"


class GazeDirection(Enum):
    """Eye gaze directions."""
    FORWARD = "forward"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class MouthShape(Enum):
    """Mouth shapes for lip-sync and expressions."""
    CLOSED = "closed"
    SLIGHTLY_OPEN = "slightly_open"
    OPEN = "open"
    WIDE_OPEN = "wide_open"
    SMILE = "smile"
    FROWN = "frown"
    O_SHAPE = "o_shape"
    # Viseme shapes for phonemes
    VISEME_AA = "viseme_aa"  # "ah" sound
    VISEME_EE = "viseme_ee"  # "ee" sound
    VISEME_OO = "viseme_oo"  # "oo" sound
    VISEME_OH = "viseme_oh"  # "oh" sound
    VISEME_TH = "viseme_th"  # "th" sound
    VISEME_FF = "viseme_ff"  # "f/v" sound
    VISEME_MM = "viseme_mm"  # "m/b/p" sound


class AnimationState(Enum):
    """Animation state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    SLEEPING = "sleeping"


class ContentType(Enum):
    """Types of content that can be displayed in windows.

    Windows subscribe to these content types to receive updates
    from the DisplayDecisionModule.
    """
    EYES = "eyes"
    MOUTH = "mouth"
    FULL_FACE = "full_face"
    STATUS = "status"
    DEBUG = "debug"
    WAVEFORM = "waveform"
    EMOTION_INDICATOR = "emotion_indicator"


@dataclass
class WindowConfig:
    """Configuration for a display window.

    Attributes:
        name: Unique identifier for the window
        title: Window title displayed in title bar
        content_type: What content this window displays
        position: (x, y) position on screen
        size: (width, height) in pixels
        fullscreen: Whether to run in fullscreen mode
        borderless: Whether to remove window decorations
        always_on_top: Whether window stays on top
        background_color: RGB tuple for background
        monitor: Which monitor to display on (0 = primary)
    """
    name: str
    title: str
    content_type: ContentType
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (800, 480)
    fullscreen: bool = False
    borderless: bool = False
    always_on_top: bool = False
    background_color: Tuple[int, int, int] = (0, 0, 0)
    monitor: int = 0


@dataclass
class DisplaySettings:
    """Global display settings with window configurations.

    Attributes:
        windows: List of window configurations
        fps: Target frames per second
        development_mode: Enable debug features
        led_count: Number of NeoPixel LEDs
        led_pin: GPIO pin for LEDs
        touch_enabled: Enable touch input
        calibration_file: Path to touch calibration
    """
    windows: List[WindowConfig] = field(default_factory=list)
    fps: int = 30
    development_mode: bool = True
    led_count: int = 60
    led_pin: int = 18
    touch_enabled: bool = False
    calibration_file: Optional[str] = None

    def __post_init__(self):
        """Set up default windows if none provided."""
        if not self.windows:
            # Default dual-window setup: eyes and mouth
            self.windows = [
                WindowConfig(
                    name="eyes",
                    title="Eyes Display",
                    content_type=ContentType.EYES,
                    position=(100, 100),
                    size=(800, 400)
                ),
                WindowConfig(
                    name="mouth",
                    title="Mouth Display",
                    content_type=ContentType.MOUTH,
                    position=(100, 520),
                    size=(800, 300)
                )
            ]


@dataclass
class EyeState:
    """State for eye rendering.

    Attributes:
        gaze: Current gaze direction
        openness: 0.0 (closed) to 1.0 (fully open)
        pupil_dilation: 0.5 (constricted) to 1.5 (dilated)
        is_blinking: Whether currently in a blink
        blink_progress: 0.0 to 1.0 during blink animation
        emotion: Current emotion affecting eye appearance
    """
    gaze: GazeDirection = GazeDirection.FORWARD
    openness: float = 1.0
    pupil_dilation: float = 1.0
    is_blinking: bool = False
    blink_progress: float = 0.0
    emotion: EmotionType = EmotionType.NEUTRAL


@dataclass
class MouthState:
    """State for mouth rendering.

    Attributes:
        shape: Current mouth shape
        openness: 0.0 (closed) to 1.0 (fully open)
        smile_amount: -1.0 (frown) to 1.0 (smile)
        emotion: Current emotion affecting mouth
        is_speaking: Whether currently speaking
        viseme: Current viseme for lip-sync
    """
    shape: MouthShape = MouthShape.CLOSED
    openness: float = 0.0
    smile_amount: float = 0.0
    emotion: EmotionType = EmotionType.NEUTRAL
    is_speaking: bool = False
    viseme: Optional[MouthShape] = None


@dataclass
class FaceState:
    """Combined face state for rendering.

    Attributes:
        eyes: Current eye state
        mouth: Current mouth state
        emotion: Overall emotion
        animation_state: Current animation state machine state
    """
    eyes: EyeState = field(default_factory=EyeState)
    mouth: MouthState = field(default_factory=MouthState)
    emotion: EmotionType = EmotionType.NEUTRAL
    animation_state: AnimationState = AnimationState.IDLE


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""
    speaker: str  # "user" or "assistant"
    text: str
    timestamp: float
    emotion: EmotionType = EmotionType.NEUTRAL


@dataclass
class ConversationContext:
    """Context for conversation with the LLM."""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_emotion: EmotionType = EmotionType.NEUTRAL
    last_activity: float = 0.0


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    engine_type: str = "piper"
    model_path: str = ""
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion_support: bool = False
    phoneme_output: bool = False


@dataclass
class AudioConfig:
    """Audio input/output configuration."""
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    enable_mock: bool = False


@dataclass
class CameraConfig:
    """Camera configuration for vision system."""
    enabled: bool = False
    device_id: int = 0
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30


# Content update dataclasses for event bus communication

@dataclass
class ContentUpdate:
    """Base class for content updates sent via event bus."""
    content_type: ContentType
    timestamp: float = 0.0


@dataclass
class EyesContentUpdate(ContentUpdate):
    """Update for eyes content."""
    content_type: ContentType = ContentType.EYES
    state: EyeState = field(default_factory=EyeState)


@dataclass
class MouthContentUpdate(ContentUpdate):
    """Update for mouth content."""
    content_type: ContentType = ContentType.MOUTH
    state: MouthState = field(default_factory=MouthState)


@dataclass
class FullFaceContentUpdate(ContentUpdate):
    """Update for full face content."""
    content_type: ContentType = ContentType.FULL_FACE
    state: FaceState = field(default_factory=FaceState)


@dataclass
class StatusContentUpdate(ContentUpdate):
    """Update for status display content."""
    content_type: ContentType = ContentType.STATUS
    animation_state: AnimationState = AnimationState.IDLE
    emotion: EmotionType = EmotionType.NEUTRAL
    message: str = ""
