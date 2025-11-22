"""
Animation states and emotion definitions for the emotion display system.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import time


class AnimationState(Enum):
    """Main animation states for the display system."""
    IDLE = auto()           # Default resting state
    WAITING = auto()        # Waiting for input (subtle movement)
    LISTENING = auto()      # Actively listening to user
    THINKING = auto()       # Processing/generating response
    SPEAKING = auto()       # Speaking/outputting audio
    ERROR = auto()          # Error state
    SLEEPING = auto()       # Low power/inactive state
    SURPRISED = auto()      # Reaction to unexpected input
    ACKNOWLEDGING = auto()  # Quick acknowledgment animation


class EmotionState(Enum):
    """Emotional expressions that modify animations."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    CONFUSED = auto()
    THINKING = auto()
    EXCITED = auto()
    TIRED = auto()
    SKEPTICAL = auto()


@dataclass
class EyeState:
    """State of a single eye."""
    # Position (0-1, relative to socket)
    x: float = 0.5
    y: float = 0.5

    # Eyelid positions (0=fully closed, 1=fully open)
    upper_lid: float = 1.0
    lower_lid: float = 0.0

    # Pupil dilation (0.5-1.5 multiplier)
    pupil_scale: float = 1.0

    # Eye shape modifiers
    squint: float = 0.0  # 0-1, amount of squinting
    wide: float = 0.0    # 0-1, amount of widening

    def copy(self) -> 'EyeState':
        """Create a copy of this state."""
        return EyeState(
            x=self.x, y=self.y,
            upper_lid=self.upper_lid, lower_lid=self.lower_lid,
            pupil_scale=self.pupil_scale,
            squint=self.squint, wide=self.wide
        )


@dataclass
class EyesState:
    """Combined state of both eyes."""
    left: EyeState = field(default_factory=EyeState)
    right: EyeState = field(default_factory=EyeState)

    # Blink state
    blink_progress: float = 0.0  # 0=open, 1=closed
    is_blinking: bool = False

    # Gaze target (screen coordinates or None for center)
    gaze_target: Optional[Tuple[float, float]] = None

    def copy(self) -> 'EyesState':
        """Create a copy of this state."""
        return EyesState(
            left=self.left.copy(),
            right=self.right.copy(),
            blink_progress=self.blink_progress,
            is_blinking=self.is_blinking,
            gaze_target=self.gaze_target
        )


@dataclass
class MouthState:
    """State of the mouth for lip-sync and expressions."""
    # Mouth opening (0=closed, 1=fully open)
    open_amount: float = 0.0

    # Mouth width (0=narrow, 0.5=normal, 1=wide/smile)
    width: float = 0.5

    # Lip positions
    upper_lip: float = 0.0   # -1 to 1 (down to up)
    lower_lip: float = 0.0   # -1 to 1 (down to up)

    # Corner positions for smile/frown
    left_corner: float = 0.0   # -1 to 1 (down to up)
    right_corner: float = 0.0  # -1 to 1 (down to up)

    # Shape modifiers
    pucker: float = 0.0      # 0-1 for O sounds
    stretch: float = 0.0     # 0-1 for EE sounds

    # Current viseme (if any)
    current_viseme: Optional[str] = None

    def copy(self) -> 'MouthState':
        """Create a copy of this state."""
        return MouthState(
            open_amount=self.open_amount,
            width=self.width,
            upper_lip=self.upper_lip,
            lower_lip=self.lower_lip,
            left_corner=self.left_corner,
            right_corner=self.right_corner,
            pucker=self.pucker,
            stretch=self.stretch,
            current_viseme=self.current_viseme
        )


@dataclass
class DisplayState:
    """Complete display state combining eyes and mouth."""
    eyes: EyesState = field(default_factory=EyesState)
    mouth: MouthState = field(default_factory=MouthState)

    animation_state: AnimationState = AnimationState.IDLE
    emotion_state: EmotionState = EmotionState.NEUTRAL

    # Timing
    state_start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def copy(self) -> 'DisplayState':
        """Create a copy of this state."""
        return DisplayState(
            eyes=self.eyes.copy(),
            mouth=self.mouth.copy(),
            animation_state=self.animation_state,
            emotion_state=self.emotion_state,
            state_start_time=self.state_start_time,
            last_update=self.last_update
        )


# Predefined emotion presets
EMOTION_PRESETS: Dict[EmotionState, Dict[str, Any]] = {
    EmotionState.NEUTRAL: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.0},
        'mouth': {'width': 0.5, 'left_corner': 0.0, 'right_corner': 0.0}
    },
    EmotionState.HAPPY: {
        'eyes': {'upper_lid': 0.9, 'squint': 0.2, 'wide': 0.0},
        'mouth': {'width': 0.7, 'left_corner': 0.3, 'right_corner': 0.3}
    },
    EmotionState.SAD: {
        'eyes': {'upper_lid': 0.7, 'squint': 0.0, 'wide': 0.0},
        'mouth': {'width': 0.4, 'left_corner': -0.3, 'right_corner': -0.3}
    },
    EmotionState.ANGRY: {
        'eyes': {'upper_lid': 0.8, 'squint': 0.4, 'wide': 0.0},
        'mouth': {'width': 0.6, 'left_corner': -0.2, 'right_corner': -0.2}
    },
    EmotionState.SURPRISED: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.5},
        'mouth': {'width': 0.4, 'open_amount': 0.3}
    },
    EmotionState.CONFUSED: {
        'eyes': {'upper_lid': 0.9, 'squint': 0.1, 'wide': 0.0},
        'mouth': {'width': 0.45, 'left_corner': -0.1, 'right_corner': 0.1}
    },
    EmotionState.THINKING: {
        'eyes': {'upper_lid': 0.85, 'squint': 0.15, 'wide': 0.0},
        'mouth': {'width': 0.45, 'left_corner': 0.1, 'right_corner': -0.1}
    },
    EmotionState.EXCITED: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.3, 'pupil_scale': 1.2},
        'mouth': {'width': 0.75, 'left_corner': 0.4, 'right_corner': 0.4, 'open_amount': 0.2}
    },
    EmotionState.TIRED: {
        'eyes': {'upper_lid': 0.6, 'squint': 0.1, 'wide': 0.0},
        'mouth': {'width': 0.5, 'left_corner': -0.1, 'right_corner': -0.1}
    },
    EmotionState.SKEPTICAL: {
        'eyes': {'upper_lid': 0.85, 'squint': 0.25, 'wide': 0.0},
        'mouth': {'width': 0.55, 'left_corner': 0.15, 'right_corner': -0.1}
    },
}


def get_emotion_preset(emotion: EmotionState) -> Dict[str, Any]:
    """Get the preset values for an emotion."""
    return EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[EmotionState.NEUTRAL])
