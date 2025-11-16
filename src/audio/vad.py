"""
Voice Activity Detection (VAD) using Silero-VAD.
Provides real-time speech/silence detection for the audio pipeline.
"""

import logging
import numpy as np
import torch
import asyncio
from typing import Optional, Callable, List, Tuple, Union
from dataclasses import dataclass

from core.types import AudioFrame
from core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    model_path: str = "silero_vad"
    threshold: float = 0.5
    min_silence_duration: float = 0.1
    min_speech_duration: float = 0.1
    window_size: int = 512
    hop_length: int = 160
    sample_rate: int = 16000


class SileroVAD:
    """Silero VAD implementation for real-time voice activity detection."""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.model = None
        self.utils = None
        self.is_initialized = False
        
        # State tracking
        self.is_speech = False
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0
        self.last_confidence = 0.0
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_confidence_change: Optional[Callable] = None
    
    async def initialize(self) -> None:
        """Initialize the VAD model."""
        try:
            # Load Silero VAD model
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.is_initialized = True
            logger.info("Silero VAD initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            raise
    
    async def process_audio(self, audio_frame: AudioFrame) -> Tuple[bool, float]:
        """
        Process audio frame and return speech detection result.
        
        Args:
            audio_frame: Audio frame to process
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        if not self.is_initialized:
            logger.warning("VAD not initialized, returning default values")
            return False, 0.0
        
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_frame.data).float()
            
            # Get speech probability
            speech_prob = self.model(audio_tensor, audio_frame.sample_rate).item()
            
            # Determine if speech is detected
            is_speech = speech_prob > self.config.threshold
            confidence = speech_prob
            
            # Update state and trigger callbacks
            await self._update_state(is_speech, confidence, audio_frame.timestamp)
            
            return is_speech, confidence
            
        except Exception as e:
            logger.error(f"Error processing audio in VAD: {e}")
            return False, 0.0
    
    async def _update_state(self, is_speech: bool, confidence: float, timestamp: float) -> None:
        """Update VAD state and trigger appropriate callbacks."""
        current_time = timestamp
        self.last_confidence = confidence
        
        if is_speech and not self.is_speech:
            # Speech started
            self.is_speech = True
            self.speech_start_time = current_time
            
            if self.on_speech_start:
                await self._call_callback(self.on_speech_start, current_time, confidence)
            
            # Emit event
            await emit_event(EventType.SPEECH_DETECTED, {
                "timestamp": current_time,
                "confidence": confidence,
                "duration": 0.0
            })
            
        elif not is_speech and self.is_speech:
            # Speech ended
            speech_duration = current_time - self.speech_start_time
            
            if speech_duration >= self.config.min_speech_duration:
                self.is_speech = False
                self.silence_start_time = current_time
                
                if self.on_speech_end:
                    await self._call_callback(self.on_speech_end, current_time, speech_duration)
                
                # Emit event
                await emit_event(EventType.SPEECH_ENDED, {
                    "timestamp": current_time,
                    "duration": speech_duration,
                    "confidence": self.last_confidence
                })
            else:
                # Speech too short, ignore
                logger.debug(f"Speech too short ({speech_duration:.2f}s), ignoring")
        
        elif is_speech and self.is_speech:
            # Speech continuing
            speech_duration = current_time - self.speech_start_time
            
            if self.on_confidence_change:
                await self._call_callback(self.on_confidence_change, confidence, speech_duration)
        
        elif not is_speech and not self.is_speech:
            # Silence continuing
            silence_duration = current_time - self.silence_start_time if self.silence_start_time > 0 else 0
            
            if silence_duration >= self.config.min_silence_duration:
                # Emit silence event
                await emit_event(EventType.SILENCE_DETECTED, {
                    "timestamp": current_time,
                    "duration": silence_duration
                })
    
    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in VAD callback: {e}")
    
    def set_callbacks(self, on_speech_start: Optional[Callable] = None,
                     on_speech_end: Optional[Callable] = None,
                     on_confidence_change: Optional[Callable] = None) -> None:
        """Set callback functions for VAD events."""
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_confidence_change = on_confidence_change
    
    def get_state(self) -> dict:
        """Get current VAD state."""
        return {
            "is_speech": self.is_speech,
            "last_confidence": self.last_confidence,
            "speech_start_time": self.speech_start_time,
            "silence_start_time": self.silence_start_time,
            "is_initialized": self.is_initialized
        }
    
    def cleanup(self) -> None:
        """Cleanup VAD resources."""
        self.is_initialized = False
        self.model = None
        self.utils = None
        logger.info("VAD cleaned up")


class MockVAD:
    """Mock VAD for testing and development."""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.is_speech = False
        self.last_confidence = 0.0
        self.callbacks = {}
    
    async def initialize(self) -> None:
        """Mock initialization."""
        logger.info("Mock VAD initialized")
    
    async def process_audio(self, audio_frame: AudioFrame) -> Tuple[bool, float]:
        """Mock audio processing - always returns speech detected."""
        # Simple mock: detect speech if audio energy is above threshold
        energy = np.mean(np.abs(audio_frame.data))
        is_speech = energy > 0.01  # Simple energy threshold
        confidence = min(energy * 10, 1.0)  # Scale to 0-1
        
        self.is_speech = is_speech
        self.last_confidence = confidence
        
        return is_speech, confidence
    
    def set_callbacks(self, **kwargs) -> None:
        """Set mock callbacks."""
        self.callbacks.update(kwargs)
    
    def get_state(self) -> dict:
        """Get mock state."""
        return {
            "is_speech": self.is_speech,
            "last_confidence": self.last_confidence,
            "is_initialized": True
        }
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        logger.info("Mock VAD cleaned up")


def create_vad(config: VADConfig, mock: bool = False) -> Union[SileroVAD, MockVAD]:
    """Create VAD instance based on configuration."""
    if mock:
        return MockVAD(config)
    else:
        return SileroVAD(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_vad():
        """Test VAD functionality."""
        config = VADConfig(threshold=0.5)
        vad = create_vad(config, mock=True)
        
        await vad.initialize()
        
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
        is_speech, confidence = await vad.process_audio(audio_frame)
        
        print(f"Speech detected: {is_speech}, Confidence: {confidence:.3f}")
        print(f"VAD state: {vad.get_state()}")
        
        vad.cleanup()
    
    # Run test
    asyncio.run(test_vad())
