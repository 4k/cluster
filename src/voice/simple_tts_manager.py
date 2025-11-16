"""
Simplified TTS Manager - Piper only with Mock fallback.
Replaces complex multi-engine TTS system with simple Piper + Mock.
"""

import logging
import numpy as np
import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path

from core.types import TTSOutput, TTSConfig, EmotionType

logger = logging.getLogger(__name__)


class SimpleTTSManager:
    """Simplified TTS manager using only Piper with Mock fallback."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.piper_model = None
        self.is_initialized = False
        self.is_processing = False
    
    async def initialize(self) -> None:
        """Initialize the TTS manager."""
        try:
            # Try to initialize Piper
            if await self._init_piper():
                logger.info("TTS manager initialized with Piper")
            else:
                logger.warning("Piper initialization failed, using mock mode")
                self.is_initialized = True  # Mock mode doesn't need initialization
                logger.info("TTS manager initialized with mock mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS manager: {e}")
            # Fallback to mock mode
            self.is_initialized = True
            logger.info("TTS manager initialized with mock mode (fallback)")
    
    async def _init_piper(self) -> bool:
        """Initialize Piper TTS engine."""
        try:
            # Import piper
            import piper
            
            # Load model
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.error(f"Piper model not found: {model_path}")
                return False
            
            self.piper_model = piper.PiperVoice.load(str(model_path))
            logger.info(f"Loaded Piper model: {model_path}")
            
            return True
            
        except ImportError:
            logger.error("Piper TTS not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            return False
    
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None, **kwargs) -> TTSOutput:
        """Generate speech using Piper or mock."""
        if not self.is_initialized:
            raise RuntimeError("TTS manager not initialized")
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            if self.piper_model:
                # Use Piper
                result = await self._generate_piper_speech(text, emotion, **kwargs)
            else:
                # Use mock
                result = await self._generate_mock_speech(text, emotion, **kwargs)
            
            # Add timing information
            result.metadata["generation_time"] = time.time() - start_time
            result.metadata["engine_type"] = "piper" if self.piper_model else "mock"
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def _generate_piper_speech(self, text: str, emotion: Optional[EmotionType] = None, **kwargs) -> TTSOutput:
        """Generate speech using Piper TTS."""
        try:
            # Prepare generation parameters
            generation_params = {
                'text': text,
                'speed': self.config.speed,
                'pitch': self.config.pitch,
                'volume': self.config.volume,
                **kwargs
            }
            
            # Generate speech
            wav = self.piper_model.synthesize(**generation_params)
            
            # Convert to numpy array
            if hasattr(wav, 'numpy'):
                wav = wav.numpy()
            elif hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
            
            # Ensure correct format
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8
            
            # Get sample rate (Piper typically uses 22050 Hz)
            sample_rate = 22050
            
            # Calculate duration
            duration = len(wav) / sample_rate
            
            return TTSOutput(
                audio_data=wav,
                sample_rate=sample_rate,
                duration=duration,
                text=text,
                phonemes=[],  # Piper doesn't provide phonemes by default
                visemes=[],   # Would need phoneme-to-viseme conversion
                emotion_markers=[],  # Would need emotion timeline
                metadata={
                    'engine': 'piper',
                    'model_path': str(self.config.model_path),
                    'speed': self.config.speed,
                    'pitch': self.config.pitch,
                    'volume': self.config.volume
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating Piper speech: {e}")
            raise
    
    async def _generate_mock_speech(self, text: str, emotion: Optional[EmotionType] = None, **kwargs) -> TTSOutput:
        """Generate mock speech for testing."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock audio (silence)
        sample_rate = 22050
        duration = len(text) * 0.1  # Rough estimate: 0.1 seconds per character
        samples = int(sample_rate * duration)
        
        # Create silence with slight noise to simulate audio
        audio_data = np.random.normal(0, 0.01, samples).astype(np.float32)
        
        return TTSOutput(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration=duration,
            text=text,
            phonemes=[],
            visemes=[],
            emotion_markers=[],
            metadata={
                'engine': 'mock',
                'mock_mode': True,
                'text_length': len(text)
            }
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get TTS engine capabilities."""
        return {
            'engine_type': 'piper' if self.piper_model else 'mock',
            'supports_emotion': False,  # Piper doesn't support emotion
            'supports_phonemes': False,  # Would need additional processing
            'supports_visemes': False,  # Would need phoneme-to-viseme conversion
            'sample_rate': 22050,
            'max_text_length': 1000,  # Reasonable limit
            'supports_streaming': False  # Not implemented
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get TTS manager state."""
        return {
            "is_initialized": self.is_initialized,
            "is_processing": self.is_processing,
            "engine_type": "piper" if self.piper_model else "mock",
            "model_path": str(self.config.model_path) if self.piper_model else None,
            "capabilities": self.get_capabilities()
        }
    
    async def cleanup(self) -> None:
        """Cleanup TTS manager resources."""
        self.piper_model = None
        self.is_initialized = False
        logger.info("TTS manager cleaned up")
