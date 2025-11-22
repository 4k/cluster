"""
Rhubarb Viseme Controller for precise lip-sync animation.

This module provides a controller that integrates Rhubarb Lip Sync data
with the emotion display system, implementing best practices for smooth,
natural-looking mouth animation.

Key features:
- Viseme lookahead (anticipatory animation)
- Smooth easing between visemes
- Coarticulation support (blending adjacent visemes)
- Extended Rhubarb shape support (A-F basic, G-H-X extended)
- Integration with event bus for real-time updates

Based on research and best practices from:
- Rhubarb Lip Sync official documentation
- Animation smoothing techniques for 2D lip sync
- Industry-standard Preston Blair phoneme groups
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class RhubarbShape(Enum):
    """
    Rhubarb's mouth shapes based on Preston Blair's phoneme groups.

    Basic shapes (A-F): Required minimum for animation
    Extended shapes (G, H, X): Optional for improved quality
    """
    # Basic shapes
    A = "A"  # Closed mouth with slight pressure (M, B, P)
    B = "B"  # Slightly open, clenched teeth (most consonants, EE sounds)
    C = "C"  # Open mouth (EH, AE vowels, transition shape)
    D = "D"  # Wide open mouth (AA vowel as in "father")
    E = "E"  # Slightly rounded (AO, ER vowels, transition to F)
    F = "F"  # Puckered lips (OO, UW, W sounds)

    # Extended shapes
    G = "G"  # Upper teeth on lower lip (F, V sounds)
    H = "H"  # Tongue visible behind teeth (L sound)
    X = "X"  # Idle/rest position (silence between words)


@dataclass
class RhubarbVisemeCue:
    """A viseme cue from Rhubarb with timing information."""
    start_time: float
    end_time: float
    shape: RhubarbShape

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2


@dataclass
class InterpolatedViseme:
    """Represents a viseme state with interpolation weights."""
    primary_shape: RhubarbShape
    primary_weight: float = 1.0
    secondary_shape: Optional[RhubarbShape] = None
    secondary_weight: float = 0.0
    intensity: float = 1.0  # Overall intensity (0-1)

    def to_mouth_params(self) -> Dict[str, float]:
        """Convert to mouth shape parameters."""
        primary_params = RHUBARB_SHAPE_PARAMS.get(
            self.primary_shape, RHUBARB_SHAPE_PARAMS[RhubarbShape.X]
        )

        if self.secondary_shape and self.secondary_weight > 0:
            secondary_params = RHUBARB_SHAPE_PARAMS.get(
                self.secondary_shape, RHUBARB_SHAPE_PARAMS[RhubarbShape.X]
            )

            # Blend parameters
            result = {}
            for key in primary_params:
                p_val = primary_params[key] * self.primary_weight
                s_val = secondary_params.get(key, 0) * self.secondary_weight
                result[key] = (p_val + s_val) * self.intensity
            return result
        else:
            return {k: v * self.intensity for k, v in primary_params.items()}


# Rhubarb shape to mouth parameters mapping
# Based on Preston Blair's standard animation mouth shapes
RHUBARB_SHAPE_PARAMS: Dict[RhubarbShape, Dict[str, float]] = {
    RhubarbShape.A: {
        'open': 0.0,
        'width': 0.5,
        'pucker': 0.1,
        'stretch': 0.0,
        'teeth_visible': False,
        'tongue_visible': False
    },
    RhubarbShape.B: {
        'open': 0.2,
        'width': 0.6,
        'pucker': 0.0,
        'stretch': 0.3,
        'teeth_visible': True,
        'tongue_visible': False
    },
    RhubarbShape.C: {
        'open': 0.5,
        'width': 0.6,
        'pucker': 0.0,
        'stretch': 0.1,
        'teeth_visible': True,
        'tongue_visible': False
    },
    RhubarbShape.D: {
        'open': 0.85,
        'width': 0.65,
        'pucker': 0.0,
        'stretch': 0.0,
        'teeth_visible': True,
        'tongue_visible': True
    },
    RhubarbShape.E: {
        'open': 0.55,
        'width': 0.45,
        'pucker': 0.4,
        'stretch': 0.0,
        'teeth_visible': True,
        'tongue_visible': False
    },
    RhubarbShape.F: {
        'open': 0.35,
        'width': 0.3,
        'pucker': 0.8,
        'stretch': 0.0,
        'teeth_visible': False,
        'tongue_visible': False
    },
    RhubarbShape.G: {
        'open': 0.15,
        'width': 0.55,
        'pucker': 0.0,
        'stretch': 0.2,
        'teeth_visible': True,
        'tongue_visible': False,
        'lower_lip_tucked': True
    },
    RhubarbShape.H: {
        'open': 0.4,
        'width': 0.55,
        'pucker': 0.0,
        'stretch': 0.15,
        'teeth_visible': True,
        'tongue_visible': True,
        'tongue_raised': True
    },
    RhubarbShape.X: {
        'open': 0.0,
        'width': 0.5,
        'pucker': 0.0,
        'stretch': 0.0,
        'teeth_visible': False,
        'tongue_visible': False
    },
}

# Mapping from Rhubarb shapes to internal viseme names
RHUBARB_TO_VISEME: Dict[RhubarbShape, str] = {
    RhubarbShape.A: "BMP",      # Closed - bilabial
    RhubarbShape.B: "LNT",      # Slightly open - consonants
    RhubarbShape.C: "AH",       # Open - vowels
    RhubarbShape.D: "AH",       # Wide open - vowels (variant)
    RhubarbShape.E: "OH",       # Rounded - vowels
    RhubarbShape.F: "OO",       # Puckered - vowels
    RhubarbShape.G: "FV",       # F/V shape
    RhubarbShape.H: "LNT",      # L shape (tongue)
    RhubarbShape.X: "SILENCE",  # Rest
}

# Coarticulation rules: which shapes blend naturally
COARTICULATION_PAIRS: Dict[Tuple[RhubarbShape, RhubarbShape], float] = {
    # Smooth transitions (high blend weight)
    (RhubarbShape.A, RhubarbShape.B): 0.7,
    (RhubarbShape.B, RhubarbShape.C): 0.8,
    (RhubarbShape.C, RhubarbShape.D): 0.8,
    (RhubarbShape.D, RhubarbShape.C): 0.8,
    (RhubarbShape.C, RhubarbShape.E): 0.7,
    (RhubarbShape.E, RhubarbShape.F): 0.8,

    # Less smooth transitions (lower blend weight)
    (RhubarbShape.A, RhubarbShape.D): 0.4,
    (RhubarbShape.F, RhubarbShape.D): 0.4,
    (RhubarbShape.G, RhubarbShape.D): 0.5,
}


class EasingFunction(Enum):
    """Easing functions for viseme interpolation."""
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    ANTICIPATE = auto()  # Slight overshoot then settle


def apply_easing(t: float, easing: EasingFunction) -> float:
    """Apply easing function to interpolation value t (0-1)."""
    t = max(0.0, min(1.0, t))

    if easing == EasingFunction.LINEAR:
        return t
    elif easing == EasingFunction.EASE_IN:
        return t * t
    elif easing == EasingFunction.EASE_OUT:
        return 1.0 - (1.0 - t) * (1.0 - t)
    elif easing == EasingFunction.EASE_IN_OUT:
        return t * t * (3.0 - 2.0 * t)
    elif easing == EasingFunction.ANTICIPATE:
        # Slight overshoot then settle
        if t < 0.5:
            return 2.0 * t * t
        else:
            overshoot = 1.1 - 0.2 * (t - 0.5)
            return min(overshoot, 1.0 + 0.05 * (1.0 - t))
    return t


@dataclass
class RhubarbControllerConfig:
    """Configuration for the Rhubarb viseme controller."""
    # Timing adjustments
    lookahead_ms: float = 50.0  # Show visemes slightly early (ms)
    hold_minimum_ms: float = 40.0  # Minimum viseme hold time (ms)

    # Interpolation settings
    transition_duration_ms: float = 60.0  # Time to transition between visemes
    easing_function: EasingFunction = EasingFunction.EASE_IN_OUT

    # Coarticulation settings
    enable_coarticulation: bool = True
    coarticulation_window_ms: float = 100.0  # Time window for coarticulation
    coarticulation_strength: float = 0.3  # Blend strength (0-1)

    # Extended shapes
    use_extended_shapes: bool = True  # Use G, H, X shapes

    # Animation quality
    update_rate_hz: float = 60.0  # Update frequency

    # Rhubarb-specific
    prefer_wide_mouth: bool = True  # Use shape D more often (more lively)
    intensity_scale: float = 1.0  # Overall mouth movement intensity


class RhubarbVisemeController:
    """
    Controller for managing Rhubarb lip-sync animation.

    Handles viseme timing, interpolation, and coarticulation to produce
    smooth, natural-looking mouth animations from Rhubarb data.
    """

    def __init__(self, config: Optional[RhubarbControllerConfig] = None):
        """
        Initialize the Rhubarb viseme controller.

        Args:
            config: Controller configuration
        """
        self.config = config or RhubarbControllerConfig()

        # Current session data
        self._cues: List[RhubarbVisemeCue] = []
        self._session_id: Optional[str] = None
        self._session_start_time: float = 0.0
        self._session_duration: float = 0.0
        self._is_active: bool = False

        # Interpolation state
        self._current_viseme: InterpolatedViseme = InterpolatedViseme(
            primary_shape=RhubarbShape.X
        )
        self._target_viseme: InterpolatedViseme = InterpolatedViseme(
            primary_shape=RhubarbShape.X
        )
        self._transition_progress: float = 1.0
        self._last_update_time: float = 0.0

        # Cue tracking
        self._current_cue_index: int = 0
        self._last_cue: Optional[RhubarbVisemeCue] = None

        # Statistics
        self.stats = {
            'sessions_processed': 0,
            'visemes_displayed': 0,
            'coarticulations_applied': 0,
            'average_transition_time_ms': 0.0,
            'total_playback_time_s': 0.0
        }

        # Callbacks
        self._on_viseme_change: Optional[Callable[[str, Dict[str, float]], None]] = None

        logger.info(f"RhubarbVisemeController initialized with config: "
                   f"lookahead={self.config.lookahead_ms}ms, "
                   f"coarticulation={self.config.enable_coarticulation}")

    def set_viseme_callback(
        self,
        callback: Callable[[str, Dict[str, float]], None]
    ) -> None:
        """
        Set callback for viseme changes.

        Args:
            callback: Function(viseme_name, mouth_params) called on viseme change
        """
        self._on_viseme_change = callback

    def load_lip_sync_data(
        self,
        cues: List[Dict[str, Any]],
        session_id: str,
        duration: float
    ) -> None:
        """
        Load lip sync data from Rhubarb output.

        Args:
            cues: List of cue dictionaries from Rhubarb
            session_id: Unique session identifier
            duration: Total audio duration in seconds
        """
        self._cues = []

        for i, cue in enumerate(cues):
            try:
                shape = RhubarbShape(cue.get('value', 'X'))
                start = cue.get('start', 0.0)

                # Calculate end time from next cue or duration
                if i + 1 < len(cues):
                    end = cues[i + 1].get('start', start + 0.1)
                else:
                    end = duration

                self._cues.append(RhubarbVisemeCue(
                    start_time=start,
                    end_time=end,
                    shape=shape
                ))
            except ValueError:
                # Unknown shape, default to X
                logger.warning(f"Unknown Rhubarb shape: {cue.get('value')}")
                self._cues.append(RhubarbVisemeCue(
                    start_time=cue.get('start', 0.0),
                    end_time=cue.get('start', 0.0) + 0.1,
                    shape=RhubarbShape.X
                ))

        self._session_id = session_id
        self._session_duration = duration
        self._current_cue_index = 0

        logger.info(f"Loaded lip sync data: {len(self._cues)} cues, "
                   f"{duration:.2f}s duration")

    def start_session(self) -> None:
        """Start the lip sync session playback."""
        if not self._cues:
            logger.warning("No lip sync data loaded")
            return

        self._session_start_time = time.time()
        self._is_active = True
        self._current_cue_index = 0
        self._transition_progress = 1.0
        self._last_update_time = time.time()
        self._last_cue = None

        # Reset to initial state
        self._current_viseme = InterpolatedViseme(primary_shape=RhubarbShape.X)
        self._target_viseme = InterpolatedViseme(primary_shape=RhubarbShape.X)

        self.stats['sessions_processed'] += 1

        logger.info(f"Started lip sync session: {self._session_id}")

    def stop_session(self) -> None:
        """Stop the lip sync session."""
        if self._is_active:
            elapsed = time.time() - self._session_start_time
            self.stats['total_playback_time_s'] += elapsed

        self._is_active = False

        # Transition back to rest position
        self._target_viseme = InterpolatedViseme(primary_shape=RhubarbShape.X)
        self._transition_progress = 0.0

        logger.info(f"Stopped lip sync session: {self._session_id}")

    def update(self, dt: float) -> Optional[Tuple[str, Dict[str, float]]]:
        """
        Update the controller and get current viseme state.

        Args:
            dt: Delta time since last update (seconds)

        Returns:
            Tuple of (viseme_name, mouth_params) or None if no change
        """
        if not self._is_active:
            # Still process transitions when stopping
            if self._transition_progress < 1.0:
                return self._update_transition(dt)
            return None

        now = time.time()
        elapsed = now - self._session_start_time

        # Apply lookahead - show visemes slightly early
        adjusted_time = elapsed + (self.config.lookahead_ms / 1000.0)

        # Check if session is complete
        if adjusted_time >= self._session_duration:
            self.stop_session()
            return self._update_transition(dt)

        # Find current cue
        cue = self._get_cue_at_time(adjusted_time)

        if cue and cue != self._last_cue:
            # New cue - start transition
            self._start_viseme_transition(cue, adjusted_time)
            self._last_cue = cue
            self.stats['visemes_displayed'] += 1

        # Update transition
        return self._update_transition(dt)

    def _get_cue_at_time(self, time_seconds: float) -> Optional[RhubarbVisemeCue]:
        """Get the active cue at a specific time."""
        # Binary search would be more efficient for large cue lists
        # but linear search is fine for typical lip sync data
        for cue in self._cues:
            if cue.start_time <= time_seconds < cue.end_time:
                return cue
        return None

    def _start_viseme_transition(
        self,
        cue: RhubarbVisemeCue,
        current_time: float
    ) -> None:
        """Start transitioning to a new viseme."""
        # Create target viseme
        target = InterpolatedViseme(
            primary_shape=cue.shape,
            intensity=self.config.intensity_scale
        )

        # Apply coarticulation if enabled
        if self.config.enable_coarticulation:
            target = self._apply_coarticulation(target, cue, current_time)

        # Store current state before transition
        self._current_viseme = self._get_interpolated_state()
        self._target_viseme = target
        self._transition_progress = 0.0

    def _apply_coarticulation(
        self,
        target: InterpolatedViseme,
        current_cue: RhubarbVisemeCue,
        current_time: float
    ) -> InterpolatedViseme:
        """Apply coarticulation blending with adjacent visemes."""
        window_s = self.config.coarticulation_window_ms / 1000.0

        # Look for the next cue
        next_cue = None
        for cue in self._cues:
            if cue.start_time > current_cue.start_time:
                next_cue = cue
                break

        if next_cue and (next_cue.start_time - current_time) < window_s:
            # Close to next viseme - apply coarticulation
            blend_pair = (current_cue.shape, next_cue.shape)
            reverse_pair = (next_cue.shape, current_cue.shape)

            blend_weight = COARTICULATION_PAIRS.get(
                blend_pair,
                COARTICULATION_PAIRS.get(reverse_pair, 0.3)
            )

            # Calculate how much to blend based on time proximity
            time_to_next = next_cue.start_time - current_time
            proximity = 1.0 - (time_to_next / window_s)
            actual_blend = blend_weight * proximity * self.config.coarticulation_strength

            target.secondary_shape = next_cue.shape
            target.secondary_weight = actual_blend
            target.primary_weight = 1.0 - actual_blend

            self.stats['coarticulations_applied'] += 1

        return target

    def _update_transition(self, dt: float) -> Optional[Tuple[str, Dict[str, float]]]:
        """Update the viseme transition and return current state."""
        if self._transition_progress >= 1.0:
            # No transition in progress
            return None

        # Calculate transition speed
        transition_duration = self.config.transition_duration_ms / 1000.0
        progress_delta = dt / transition_duration if transition_duration > 0 else 1.0

        self._transition_progress = min(1.0, self._transition_progress + progress_delta)

        # Get interpolated state
        interpolated = self._get_interpolated_state()

        # Convert to viseme name and parameters
        viseme_name = RHUBARB_TO_VISEME.get(
            interpolated.primary_shape, "SILENCE"
        )
        mouth_params = interpolated.to_mouth_params()

        # Notify callback
        if self._on_viseme_change:
            self._on_viseme_change(viseme_name, mouth_params)

        return (viseme_name, mouth_params)

    def _get_interpolated_state(self) -> InterpolatedViseme:
        """Get the current interpolated viseme state."""
        if self._transition_progress >= 1.0:
            return self._target_viseme

        # Apply easing
        t = apply_easing(self._transition_progress, self.config.easing_function)

        # Interpolate between current and target
        if self._current_viseme.primary_shape == self._target_viseme.primary_shape:
            # Same shape, just interpolate intensity
            return InterpolatedViseme(
                primary_shape=self._target_viseme.primary_shape,
                primary_weight=1.0,
                intensity=self._lerp(
                    self._current_viseme.intensity,
                    self._target_viseme.intensity,
                    t
                )
            )
        else:
            # Different shapes - blend between them
            return InterpolatedViseme(
                primary_shape=self._target_viseme.primary_shape,
                primary_weight=t,
                secondary_shape=self._current_viseme.primary_shape,
                secondary_weight=1.0 - t,
                intensity=self._lerp(
                    self._current_viseme.intensity,
                    self._target_viseme.intensity,
                    t
                )
            )

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t

    def get_current_viseme(self) -> str:
        """Get the current viseme name."""
        state = self._get_interpolated_state()
        return RHUBARB_TO_VISEME.get(state.primary_shape, "SILENCE")

    def get_current_mouth_params(self) -> Dict[str, float]:
        """Get current mouth parameters."""
        state = self._get_interpolated_state()
        return state.to_mouth_params()

    def get_progress(self) -> float:
        """Get session playback progress (0-1)."""
        if not self._is_active or self._session_duration <= 0:
            return 0.0

        elapsed = time.time() - self._session_start_time
        return min(1.0, elapsed / self._session_duration)

    def get_remaining_time(self) -> float:
        """Get remaining playback time in seconds."""
        if not self._is_active:
            return 0.0

        elapsed = time.time() - self._session_start_time
        return max(0.0, self._session_duration - elapsed)

    def is_active(self) -> bool:
        """Check if a lip sync session is currently active."""
        return self._is_active

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            **self.stats,
            'is_active': self._is_active,
            'current_session': self._session_id,
            'cue_count': len(self._cues),
            'progress': self.get_progress()
        }


class RhubarbLipSyncIntegration:
    """
    Integration layer between Rhubarb controller and event bus.

    Handles event subscriptions and coordinates lip sync playback
    with the display system.
    """

    def __init__(
        self,
        controller: Optional[RhubarbVisemeController] = None,
        config: Optional[RhubarbControllerConfig] = None
    ):
        """
        Initialize the integration layer.

        Args:
            controller: Pre-configured controller instance
            config: Controller configuration (used if controller not provided)
        """
        self.controller = controller or RhubarbVisemeController(config)
        self.event_bus = None
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

        # Pending lip sync data from Rhubarb service
        self._pending_data: Dict[str, Any] = {}

        logger.info("RhubarbLipSyncIntegration initialized")

    async def initialize(self) -> bool:
        """Initialize event bus connection and start update loop."""
        try:
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()

            # Subscribe to lip sync events
            self.event_bus.subscribe(
                EventType.LIP_SYNC_READY,
                self._on_lip_sync_ready,
                priority=10
            )
            self.event_bus.subscribe(
                EventType.LIP_SYNC_STARTED,
                self._on_lip_sync_started,
                priority=10
            )
            self.event_bus.subscribe(
                EventType.LIP_SYNC_COMPLETED,
                self._on_lip_sync_completed,
                priority=10
            )
            self.event_bus.subscribe(
                EventType.AUDIO_PLAYBACK_ENDED,
                self._on_audio_playback_ended,
                priority=5
            )

            self.is_running = True

            # Start update loop
            self._update_task = asyncio.create_task(self._update_loop())

            logger.info("RhubarbLipSyncIntegration connected to event bus")
            return True

        except ImportError:
            logger.warning("Event bus not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    async def stop(self) -> None:
        """Stop the integration."""
        self.is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        self.controller.stop_session()
        logger.info("RhubarbLipSyncIntegration stopped")

    async def _update_loop(self) -> None:
        """Main update loop for viseme emission."""
        update_interval = 1.0 / self.controller.config.update_rate_hz

        while self.is_running:
            try:
                result = self.controller.update(update_interval)

                if result and self.event_bus:
                    viseme_name, mouth_params = result

                    # Emit viseme update event
                    from src.core.event_bus import EventType
                    await self.event_bus.emit(
                        EventType.MOUTH_SHAPE_UPDATE,
                        {
                            'viseme': viseme_name,
                            'params': mouth_params,
                            'source': 'rhubarb_controller'
                        },
                        source='rhubarb_integration'
                    )

                await asyncio.sleep(update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(update_interval)

    async def _on_lip_sync_ready(self, event) -> None:
        """Handle lip sync data ready event."""
        data = event.data
        session_id = data.get('session_id')
        cues = data.get('cues', [])
        duration = data.get('duration', 0.0)

        if session_id and cues:
            self._pending_data[session_id] = {
                'cues': cues,
                'duration': duration
            }
            logger.debug(f"Received lip sync data for session: {session_id}")

    async def _on_lip_sync_started(self, event) -> None:
        """Handle lip sync started event."""
        session_id = event.data.get('session_id')

        # Try to get data from pending or from event
        if session_id and session_id in self._pending_data:
            data = self._pending_data.pop(session_id)
            self.controller.load_lip_sync_data(
                cues=data['cues'],
                session_id=session_id,
                duration=data['duration']
            )
        elif event.data.get('cues'):
            self.controller.load_lip_sync_data(
                cues=event.data['cues'],
                session_id=session_id or 'unknown',
                duration=event.data.get('duration', 0.0)
            )

        self.controller.start_session()

    async def _on_lip_sync_completed(self, event) -> None:
        """Handle lip sync completed event."""
        self.controller.stop_session()

    async def _on_audio_playback_ended(self, event) -> None:
        """Handle audio playback ended - ensure lip sync stops."""
        if self.controller.is_active():
            self.controller.stop_session()
