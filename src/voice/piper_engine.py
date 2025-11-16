"""
Piper TTS engine implementation.
Provides fast, lightweight speech synthesis optimized for Raspberry Pi.
"""

import logging
import numpy as np
import asyncio
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from .tts_interface import TTSEngine
from core.types import TTSOutput, TTSConfig, Phoneme, Viseme, EmotionMarker, EmotionType

logger = logging.getLogger(__name__)


class PiperTTSEngine(TTSEngine):
    """Piper TTS engine implementation."""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.model = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Piper TTS engine."""
        try:
            # Import piper
            import piper
            
            # Load model
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Piper model not found: {model_path}")
            
            self.model = piper.PiperVoice.load(str(model_path))
            logger.info(f"Loaded Piper model: {model_path}")
            
            self.is_initialized = True
            logger.info("Piper TTS engine initialized successfully")
            
        except ImportError:
            logger.error("Piper TTS not available, using mock implementation")
            self.model = None
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            raise
    
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None,
                            **kwargs) -> TTSOutput:
        """Generate speech using Piper TTS."""
        if not self.is_initialized:
            raise RuntimeError("Piper TTS engine not initialized")
        
        if self.model is None:
            # Mock implementation for testing
            return self._mock_generate_speech(text, emotion, **kwargs)
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            # Prepare generation parameters
            generation_params = {
                'text': text,
                'speed': self.config.speed,
                'pitch': self.config.pitch,
                'volume': self.config.volume,
                **kwargs
            }
            
            # Generate speech
            wav = self.model.synthesize(**generation_params)
            
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
            
            # Generate phonemes if supported
            phonemes = None
            if self.config.phoneme_output:
                phonemes = await self._extract_phonemes(text, wav, sample_rate)
            
            # Generate visemes from phonemes
            visemes = None
            if phonemes:
                from .tts_interface import create_visemes_from_phonemes
                visemes = create_visemes_from_phonemes(phonemes)
            
            # Generate emotion timeline
            emotion_timeline = None
            if emotion:
                from .tts_interface import create_emotion_timeline
                emotion_timeline = create_emotion_timeline(text, emotion, duration)
            
            # Create output
            result = TTSOutput(
                audio=wav,
                sample_rate=sample_rate,
                duration=duration,
                phonemes=phonemes,
                visemes=visemes,
                emotion_timeline=emotion_timeline,
                metadata={
                    'engine': 'piper',
                    'model': str(self.config.model_path),
                    'voice_id': self.config.voice_id,
                    'emotion': emotion.value if emotion else None,
                    'generation_time': time.time() - start_time
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech with Piper TTS: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def _extract_phonemes(self, text: str, audio: np.ndarray, 
                               sample_rate: int) -> List[Phoneme]:
        """Extract phonemes from generated audio."""
        try:
            # Piper doesn't provide phoneme alignment directly
            # This is a simplified phoneme extraction
            # In practice, you would use a forced alignment tool like Montreal Forced Aligner
            
            # For now, create mock phonemes based on text length
            words = text.split()
            phonemes = []
            current_time = 0.0
            
            for word in words:
                # Estimate phoneme duration based on word length
                word_duration = len(word) * 0.08  # 80ms per character
                
                # Create mock phonemes for each character
                for char in word:
                    if char.isalpha():
                        phoneme = Phoneme(
                            symbol=char.upper(),
                            start_time=current_time,
                            end_time=current_time + 0.04,  # 40ms per phoneme
                            confidence=0.9
                        )
                        phonemes.append(phoneme)
                        current_time += 0.04
                    else:
                        current_time += 0.02  # 20ms for non-alphabetic characters
            
            return phonemes
            
        except Exception as e:
            logger.error(f"Error extracting phonemes: {e}")
            return []
    
    def _mock_generate_speech(self, text: str, emotion: Optional[EmotionType] = None,
                            **kwargs) -> TTSOutput:
        """Mock speech generation for testing."""
        # Generate mock audio (sine wave with varying frequency)
        duration = len(text) * 0.08  # 80ms per character
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Create sine wave with frequency based on text
        base_freq = 150 + hash(text) % 150  # 150-300 Hz
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Add some variation
        audio += np.sin(2 * np.pi * base_freq * 1.3 * t) * 0.1
        
        # Apply envelope
        envelope = np.exp(-t * 1.5)  # Decay envelope
        audio *= envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Create mock phonemes
        phonemes = []
        current_time = 0.0
        for char in text:
            if char.isalpha():
                phoneme = Phoneme(
                    symbol=char.upper(),
                    start_time=current_time,
                    end_time=current_time + 0.04,
                    confidence=0.9
                )
                phonemes.append(phoneme)
                current_time += 0.04
            else:
                current_time += 0.02
        
        # Create visemes from phonemes
        from .tts_interface import create_visemes_from_phonemes
        visemes = create_visemes_from_phonemes(phonemes)
        
        # Create emotion timeline
        emotion_timeline = None
        if emotion:
            from .tts_interface import create_emotion_timeline
            emotion_timeline = create_emotion_timeline(text, emotion, duration)
        
        return TTSOutput(
            audio=audio.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            phonemes=phonemes,
            visemes=visemes,
            emotion_timeline=emotion_timeline,
            metadata={
                'engine': 'piper_mock',
                'model': 'mock',
                'voice_id': 'mock_voice',
                'emotion': emotion.value if emotion else None,
                'generation_time': 0.05
            }
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get engine capabilities."""
        return {
            'engine_type': 'piper',
            'emotion_support': False,  # Piper doesn't support emotion
            'phoneme_output': self.config.phoneme_output,
            'custom_voice_support': True,
            'languages': ['en'],  # Add more as needed
            'voices': ['default'],  # Add more as needed
            'speed_range': (0.5, 2.0),
            'pitch_range': (0.5, 2.0),
            'volume_range': (0.0, 1.0),
            'sample_rates': [22050],
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready(),
            'optimized_for_pi': True
        }
    
    def cleanup(self) -> None:
        """Cleanup engine resources."""
        self.is_initialized = False
        self.model = None
        logger.info("Piper TTS engine cleaned up")


# Register the engine with the factory
from .tts_interface import TTSFactory
TTSFactory.register_engine('piper', PiperTTSEngine)


# Example usage and testing
if __name__ == "__main__":
    async def test_piper_engine():
        """Test Piper TTS engine functionality."""
        config = TTSConfig(
            engine_type='piper',
            model_path='models/piper_voice.onnx',
            emotion_support=False,
            phoneme_output=True
        )
        
        engine = PiperTTSEngine(config)
        await engine.initialize()
        
        # Test speech generation
        result = await engine.generate_speech(
            "Hello, this is a test of the Piper TTS engine.",
            emotion=None  # Piper doesn't support emotion
        )
        
        print(f"Generated audio: {len(result.audio)} samples")
        print(f"Sample rate: {result.sample_rate} Hz")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Phonemes: {len(result.phonemes) if result.phonemes else 0}")
        print(f"Visemes: {len(result.visemes) if result.visemes else 0}")
        print(f"Capabilities: {engine.get_capabilities()}")
        
        engine.cleanup()
    
    # Run test
    asyncio.run(test_piper_engine())
