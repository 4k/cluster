"""
Type definitions and data structures for the voice assistant system.
Provides type safety and clear interfaces between components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


class EmotionType(Enum):
    """Supported emotion types for the assistant."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    CURIOUS = "curious"
    EXCITED = "excited"
    CALM = "calm"
    FOCUSED = "focused"
    CONFUSED = "confused"


class GazeDirection(Enum):
    """Gaze directions for eye display."""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class MouthShape(Enum):
    """Mouth shapes for phoneme visualization."""
    CLOSED = "closed"
    OPEN = "open"
    SMILE = "smile"
    FROWN = "frown"
    O_SHAPE = "o_shape"
    A_SHAPE = "a_shape"
    E_SHAPE = "e_shape"
    I_SHAPE = "i_shape"
    U_SHAPE = "u_shape"
    M_SHAPE = "m_shape"
    F_SHAPE = "f_shape"
    V_SHAPE = "v_shape"
    TH_SHAPE = "th_shape"
    S_SHAPE = "s_shape"
    TALK = "talk"


class SpeakerState(Enum):
    """Speaker identification states."""
    UNKNOWN = "unknown"
    IDENTIFIED = "identified"
    ENROLLING = "enrolling"
    VERIFIED = "verified"


class ConversationState(Enum):
    """Conversation flow states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    WAITING = "waiting"


@dataclass
class Phoneme:
    """Phoneme data for lip-sync animation."""
    symbol: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Viseme:
    """Viseme data for mouth shape animation."""
    shape: MouthShape
    start_time: float
    end_time: float
    intensity: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EmotionMarker:
    """Emotion marker with timing information."""
    emotion: EmotionType
    start_time: float
    end_time: float
    intensity: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TTSOutput:
    """Standardized TTS output format."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    phonemes: Optional[List[Phoneme]] = None
    visemes: Optional[List[Viseme]] = None
    emotion_timeline: Optional[List[EmotionMarker]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.duration <= 0:
            self.duration = len(self.audio) / self.sample_rate


@dataclass
class AudioFrame:
    """Audio frame data for processing."""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    channels: int = 1
    
    @property
    def samples(self) -> int:
        return len(self.data)


@dataclass
class SpeakerProfile:
    """Speaker identification profile."""
    speaker_id: str
    name: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    confidence_threshold: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None
    total_interactions: int = 0
    
    def update_last_seen(self):
        """Update last seen timestamp."""
        self.last_seen = datetime.now()
        self.total_interactions += 1


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker_id: str
    text: str
    timestamp: float
    confidence: float = 1.0
    emotion: Optional[EmotionType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for conversation management."""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_emotion: EmotionType = EmotionType.NEUTRAL
    current_speaker: Optional[str] = None
    last_activity: float = 0.0
    max_turns: int = 10
    max_age_seconds: float = 120.0
    
    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.current_speaker = turn.speaker_id
        self.last_activity = turn.timestamp
        
        # Trim old turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Remove old turns
        current_time = turn.timestamp
        self.turns = [
            t for t in self.turns 
            if current_time - t.timestamp <= self.max_age_seconds
        ]
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        return self.turns[-count:] if self.turns else []
    
    def is_stale(self, current_time: Optional[float] = None) -> bool:
        """Check if conversation is stale."""
        if current_time is None:
            current_time = datetime.now().timestamp()
        return current_time - self.last_activity > self.max_age_seconds


@dataclass
class VisualContext:
    """Visual context from camera module."""
    people_present: List[str] = field(default_factory=list)
    objects_in_view: List[str] = field(default_factory=list)
    scene_description: str = ""
    recent_gestures: List[str] = field(default_factory=list)
    dominant_emotion: Optional[EmotionType] = None
    timestamp: float = 0.0
    
    def add_person(self, person_id: str):
        """Add a person to the scene."""
        if person_id not in self.people_present:
            self.people_present.append(person_id)
    
    def remove_person(self, person_id: str):
        """Remove a person from the scene."""
        if person_id in self.people_present:
            self.people_present.remove(person_id)
    
    def add_gesture(self, gesture: str):
        """Add a recent gesture."""
        self.recent_gestures.append(gesture)
        # Keep only last 5 gestures
        if len(self.recent_gestures) > 5:
            self.recent_gestures = self.recent_gestures[-5:]


@dataclass
class DecisionContext:
    """Context for response decision making."""
    conversation: ConversationContext
    visual: VisualContext
    audio_confidence: float = 0.0
    wake_word_used: bool = False
    direct_address: bool = False
    question_asked: bool = False
    emotion_intensity: float = 0.0
    
    def should_respond(self) -> Tuple[bool, float]:
        """Determine if assistant should respond and with what confidence."""
        confidence = 0.0
        
        # High confidence factors
        if self.wake_word_used:
            confidence += 0.8
        if self.direct_address:
            confidence += 0.6
        if self.question_asked:
            confidence += 0.5
        
        # Medium confidence factors
        if self.audio_confidence > 0.8:
            confidence += 0.3
        if self.emotion_intensity > 0.7:
            confidence += 0.2
        if len(self.visual.people_present) > 0:
            confidence += 0.1
        
        # Low confidence factors
        if len(self.conversation.turns) == 0:
            confidence -= 0.2
        if self.conversation.is_stale():
            confidence -= 0.3
        
        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence > 0.3, confidence


@dataclass
class DisplayState:
    """State for display rendering."""
    eyes_emotion: EmotionType = EmotionType.NEUTRAL
    eyes_gaze: GazeDirection = GazeDirection.CENTER
    eyes_blink_state: float = 0.0  # 0 = open, 1 = closed
    mouth_shape: MouthShape = MouthShape.CLOSED
    mouth_emotion: EmotionType = EmotionType.NEUTRAL
    is_speaking: bool = False
    is_listening: bool = False
    is_processing: bool = False
    
    def update_eyes(self, emotion: EmotionType, gaze: GazeDirection = GazeDirection.CENTER):
        """Update eye state."""
        self.eyes_emotion = emotion
        self.eyes_gaze = gaze
    
    def update_mouth(self, shape: MouthShape, emotion: EmotionType = EmotionType.NEUTRAL):
        """Update mouth state."""
        self.mouth_shape = shape
        self.mouth_emotion = emotion
    
    def set_speaking(self, speaking: bool):
        """Set speaking state."""
        self.is_speaking = speaking
        if not speaking:
            self.mouth_shape = MouthShape.CLOSED
    
    def set_listening(self, listening: bool):
        """Set listening state."""
        self.is_listening = listening
    
    def set_processing(self, processing: bool):
        """Set processing state."""
        self.is_processing = processing


@dataclass
class SystemHealth:
    """System health monitoring data."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 0.0
    disk_usage: float = 0.0
    audio_underruns: int = 0
    display_fps: float = 0.0
    last_error: Optional[str] = None
    timestamp: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return (
            self.cpu_usage < 90.0 and
            self.memory_usage < 80.0 and
            self.temperature < 75.0 and
            self.disk_usage < 90.0 and
            self.audio_underruns < 10 and
            self.display_fps > 20.0
        )


@dataclass
class TTSConfig:
    """Configuration for TTS engines."""
    engine_type: str = "mock"
    model_path: str = "mock"
    voice_id: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion_support: bool = False
    phoneme_output: bool = False
    custom_voice_path: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 512
    device: Optional[str] = None
    wake_word_threshold: float = 0.5
    vad_threshold: float = 0.5
    wake_words: List[str] = field(default_factory=lambda: ["hey assistant", "hey pi"])
    silence_timeout: float = 2.0
    max_audio_length: float = 30.0


@dataclass
# DisplayConfig has been moved to display.display_manager module
# Import from there instead: from display.display_manager import DisplayConfig


@dataclass
class CameraConfig:
    """Camera configuration."""
    enabled: bool = False
    interface: str = "serial"  # serial, network, usb
    port: Optional[str] = None
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    detection_confidence: float = 0.7
    tracking_enabled: bool = True


# Type aliases for common use cases
AudioData = np.ndarray
SpeakerID = str
ConversationID = str
EventData = Dict[str, Any]
ConfigDict = Dict[str, Any]
