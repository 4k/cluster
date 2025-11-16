"""
Wake word detection using Silero-VAD.
Provides real-time wake word detection with integrated VAD.
"""

import logging
import numpy as np
import torch
import asyncio
from typing import Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass
import time

from core.types import AudioFrame
from core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    model_path: str = "silero_vad"
    wake_words: List[str] = None
    threshold: float = 0.5
    vad_threshold: float = 0.5
    min_activation_duration: float = 0.1
    max_activation_duration: float = 2.0
    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms at 16kHz
    energy_threshold: float = 0.05  # Energy threshold for wake word detection
    pattern_threshold: float = 0.3  # Pattern matching threshold
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["hey assistant", "790", "computer"]


class SileroWakeWordDetector:
    """Silero-VAD based wake word detection implementation."""
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.vad_model = None
        self.vad_utils = None
        self.is_initialized = False
        
        # State tracking
        self.activation_start_time = 0.0
        self.is_activated = False
        self.last_confidence = 0.0
        self.activation_history: List[Dict[str, Any]] = []
        
        # Audio pattern matching for wake words
        self.audio_buffer: List[np.ndarray] = []
        self.buffer_max_size = int(self.config.sample_rate * 3.0)  # 3 seconds max
        
        # Callbacks
        self.on_wake_word_detected: Optional[Callable] = None
        self.on_activation_start: Optional[Callable] = None
        self.on_activation_end: Optional[Callable] = None
    
    async def initialize(self) -> None:
        """Initialize the wake word detection model."""
        try:
            # Load Silero VAD model
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.is_initialized = True
            logger.info(f"Silero-VAD wake word detector initialized with wake words: {self.config.wake_words}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Silero-VAD wake word detector: {e}")
            raise
    
    async def process_audio(self, audio_frame: AudioFrame) -> Dict[str, Any]:
        """
        Process audio frame and return wake word detection result.
        
        Args:
            audio_frame: Audio frame to process
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_initialized:
            logger.warning("Wake word detector not initialized")
            return {"detected": False, "confidence": 0.0, "wake_word": None}
        
        try:
            # Add audio to buffer
            self.audio_buffer.append(audio_frame.data.copy())
            
            # Keep buffer size manageable
            if len(self.audio_buffer) * len(audio_frame.data) > self.buffer_max_size:
                self.audio_buffer = self.audio_buffer[-int(self.buffer_max_size / len(audio_frame.data)):]
            
            # Convert to tensor for VAD
            audio_tensor = torch.from_numpy(audio_frame.data).float()
            
            # Get speech probability from VAD
            speech_prob = self.vad_model(audio_tensor, audio_frame.sample_rate).item()
            
            # Calculate audio energy
            energy = np.mean(np.abs(audio_frame.data))
            
            # Simple wake word detection based on VAD + energy + pattern
            is_detected, confidence, detected_wake_word = self._detect_wake_word(
                speech_prob, energy, audio_frame.data
            )
            
            # Update activation state
            await self._update_activation_state(is_detected, confidence, 
                                              detected_wake_word, audio_frame.timestamp)
            
            return {
                "detected": is_detected,
                "confidence": confidence,
                "wake_word": detected_wake_word,
                "timestamp": audio_frame.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error processing audio in wake word detector: {e}")
            return {"detected": False, "confidence": 0.0, "wake_word": None}
    
    def _detect_wake_word(self, speech_prob: float, energy: float, audio_data: np.ndarray) -> tuple[bool, float, Optional[str]]:
        """
        Detect wake word based on VAD, energy, and audio patterns.
        
        Args:
            speech_prob: Speech probability from VAD
            energy: Audio energy level
            audio_data: Raw audio data
            
        Returns:
            Tuple of (is_detected, confidence, wake_word)
        """
        # Check if we have speech and sufficient energy
        has_speech = speech_prob > self.config.vad_threshold
        has_energy = energy > self.config.energy_threshold
        
        if not (has_speech and has_energy):
            return False, 0.0, None
        
        # Calculate confidence based on speech probability and energy
        confidence = min((speech_prob * 0.7 + energy * 10 * 0.3), 1.0)
        
        # Check if confidence exceeds threshold
        if confidence > self.config.threshold:
            # Simple pattern matching - look for specific audio characteristics
            pattern_score = self._analyze_audio_pattern(audio_data)
            
            if pattern_score > self.config.pattern_threshold:
                # Select wake word based on pattern or use default
                detected_wake_word = self._select_wake_word(pattern_score)
                return True, confidence, detected_wake_word
        
        return False, confidence, None
    
    def _analyze_audio_pattern(self, audio_data: np.ndarray) -> float:
        """
        Analyze audio pattern for wake word characteristics.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Pattern score (0.0 to 1.0)
        """
        try:
            # Calculate various audio features
            # 1. Zero crossing rate (indicates speech characteristics)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            zcr = zero_crossings / len(audio_data)
            
            # 2. Spectral centroid (brightness of sound)
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.config.sample_rate)
            magnitude = np.abs(fft)
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
            
            # 3. RMS energy variation
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Combine features into pattern score
            # Higher ZCR and moderate spectral centroid suggest speech-like patterns
            pattern_score = min(
                (zcr * 0.4 + 
                 min(abs(spectral_centroid) / 1000, 1.0) * 0.3 + 
                 min(rms * 20, 1.0) * 0.3), 
                1.0
            )
            
            return pattern_score
            
        except Exception as e:
            logger.debug(f"Error analyzing audio pattern: {e}")
            return 0.0
    
    def _select_wake_word(self, pattern_score: float) -> str:
        """
        Select appropriate wake word based on pattern score.
        
        Args:
            pattern_score: Pattern analysis score
            
        Returns:
            Selected wake word
        """
        # Simple selection based on pattern score
        if pattern_score > 0.8:
            return self.config.wake_words[0]  # Primary wake word
        elif pattern_score > 0.6:
            return self.config.wake_words[1] if len(self.config.wake_words) > 1 else self.config.wake_words[0]
        else:
            return self.config.wake_words[-1] if len(self.config.wake_words) > 2 else self.config.wake_words[0]
    
    async def _update_activation_state(self, is_detected: bool, confidence: float,
                                     wake_word: Optional[str], timestamp: float) -> None:
        """Update wake word activation state."""
        current_time = timestamp
        self.last_confidence = confidence
        
        if is_detected and not self.is_activated:
            # Activation started
            self.is_activated = True
            self.activation_start_time = current_time
            
            if self.on_activation_start:
                await self._call_callback(self.on_activation_start, wake_word, confidence, current_time)
            
        elif not is_detected and self.is_activated:
            # Activation ended
            activation_duration = current_time - self.activation_start_time
            
            if (self.config.min_activation_duration <= activation_duration <= 
                self.config.max_activation_duration):
                
                # Valid activation
                self.is_activated = False
                
                # Record activation
                activation_record = {
                    "wake_word": wake_word,
                    "confidence": confidence,
                    "duration": activation_duration,
                    "timestamp": current_time
                }
                self.activation_history.append(activation_record)
                
                # Keep only last 100 activations
                if len(self.activation_history) > 100:
                    self.activation_history = self.activation_history[-100:]
                
                if self.on_wake_word_detected:
                    await self._call_callback(self.on_wake_word_detected, activation_record)
                
                if self.on_activation_end:
                    await self._call_callback(self.on_activation_end, activation_record)
                
                # Emit event
                await emit_event(EventType.WAKE_WORD_DETECTED, {
                    "wake_word": wake_word,
                    "confidence": confidence,
                    "duration": activation_duration,
                    "timestamp": current_time
                })
                
            else:
                # Invalid activation duration
                logger.debug(f"Invalid activation duration: {activation_duration:.2f}s")
                self.is_activated = False
        
        elif is_detected and self.is_activated:
            # Activation continuing
            activation_duration = current_time - self.activation_start_time
            
            # Check for maximum duration
            if activation_duration > self.config.max_activation_duration:
                logger.debug("Activation duration exceeded maximum, ending activation")
                self.is_activated = False
    
    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in wake word callback: {e}")
    
    def set_callbacks(self, on_wake_word_detected: Optional[Callable] = None,
                     on_activation_start: Optional[Callable] = None,
                     on_activation_end: Optional[Callable] = None) -> None:
        """Set callback functions for wake word events."""
        self.on_wake_word_detected = on_wake_word_detected
        self.on_activation_start = on_activation_start
        self.on_activation_end = on_activation_end
    
    def get_state(self) -> Dict[str, Any]:
        """Get current wake word detector state."""
        return {
            "is_initialized": self.is_initialized,
            "is_activated": self.is_activated,
            "last_confidence": self.last_confidence,
            "activation_start_time": self.activation_start_time,
            "total_activations": len(self.activation_history),
            "recent_activations": self.activation_history[-5:] if self.activation_history else []
        }
    
    def get_activation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activation history."""
        return self.activation_history[-limit:] if self.activation_history else []
    
    def clear_activation_history(self) -> None:
        """Clear activation history."""
        self.activation_history.clear()
        logger.info("Wake word activation history cleared")
    
    def cleanup(self) -> None:
        """Cleanup wake word detector resources."""
        self.is_initialized = False
        self.vad_model = None
        self.vad_utils = None
        self.audio_buffer.clear()
        logger.info("Silero-VAD wake word detector cleaned up")


class MockWakeWordDetector:
    """Mock wake word detector for testing and development."""
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.is_initialized = True
        self.activation_count = 0
        self.callbacks = {}
    
    async def initialize(self) -> None:
        """Mock initialization."""
        logger.info("Mock wake word detector initialized")
    
    async def process_audio(self, audio_frame: AudioFrame) -> Dict[str, Any]:
        """Mock audio processing - occasionally detects wake words."""
        # Simple mock: randomly detect wake word
        import random
        is_detected = random.random() < 0.01  # 1% chance per frame
        confidence = random.random() if is_detected else 0.0
        wake_word = random.choice(self.config.wake_words) if is_detected else None
        
        if is_detected:
            self.activation_count += 1
        
        return {
            "detected": is_detected,
            "confidence": confidence,
            "wake_word": wake_word,
            "timestamp": audio_frame.timestamp
        }
    
    def set_callbacks(self, **kwargs) -> None:
        """Set mock callbacks."""
        self.callbacks.update(kwargs)
    
    def get_state(self) -> Dict[str, Any]:
        """Get mock state."""
        return {
            "is_initialized": True,
            "is_activated": False,
            "last_confidence": 0.0,
            "activation_start_time": 0.0,
            "total_activations": self.activation_count,
            "recent_activations": []
        }
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        logger.info("Mock wake word detector cleaned up")


def create_wake_word_detector(config: WakeWordConfig, mock: bool = False) -> Union[SileroWakeWordDetector, MockWakeWordDetector]:
    """Create wake word detector instance based on configuration."""
    if mock:
        return MockWakeWordDetector(config)
    
    # Try to create Silero-VAD implementation
    try:
        detector = SileroWakeWordDetector(config)
        return detector
    except Exception as e:
        logger.warning(f"Failed to create Silero-VAD wake word detector: {e}, falling back to mock implementation")
        return MockWakeWordDetector(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_wake_word_detector():
        """Test wake word detector functionality."""
        config = WakeWordConfig(
            wake_words=["hey assistant", "hey pi"],
            threshold=0.5
        )
        detector = create_wake_word_detector(config, mock=True)
        
        await detector.initialize()
        
        # Test with mock audio
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Generate test audio (sine wave)
        t = np.linspace(0, duration, samples)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz sine wave
        
        audio_frame = AudioFrame(
            data=audio_data,
            sample_rate=sample_rate,
            timestamp=0.0,
            duration=duration
        )
        
        # Process audio
        result = await detector.process_audio(audio_frame)
        
        print(f"Wake word detected: {result['detected']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Wake word: {result['wake_word']}")
        print(f"Detector state: {detector.get_state()}")
        
        detector.cleanup()
    
    # Run test
    asyncio.run(test_wake_word_detector())
