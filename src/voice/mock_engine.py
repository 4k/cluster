"""
Mock TTS engine implementation.
Provides a fallback TTS engine when real engines are not available.
"""

import logging
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
import time

from .tts_interface import TTSEngine
from core.types import TTSOutput, TTSConfig, Phoneme, Viseme, EmotionMarker, EmotionType

logger = logging.getLogger(__name__)


class MockTTSEngine(TTSEngine):
    """Mock TTS engine implementation for testing and fallback."""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.is_initialized = False
        
        # Mock mode ignores all config - use hardcoded audio only
        logger.info("Mock TTS engine initialized - ignoring all engine configuration")
    
    async def initialize(self) -> None:
        """Initialize the mock TTS engine."""
        self.is_initialized = True
        logger.info("Mock TTS engine initialized - using hardcoded audio generation")
    
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None,
                            **kwargs) -> TTSOutput:
        """Generate mock speech from text."""
        if not self.is_initialized:
            raise RuntimeError("Mock TTS engine not initialized")
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            # Try to load sample audio file
            sample_audio = self._load_sample_audio()
            
            if sample_audio is not None:
                # Use sample audio file
                audio = sample_audio
                sample_rate = 22050
                duration = len(audio) / sample_rate
            else:
                # Fallback to generated audio
                audio, sample_rate, duration = self._generate_mock_audio(text, emotion)
            
            # Create mock phonemes
            phonemes = []
            current_time = 0.0
            for char in text:
                if char.isalpha():
                    phoneme = Phoneme(
                        symbol=char.upper(),
                        start_time=current_time,
                        end_time=current_time + 0.05,
                        confidence=0.8
                    )
                    phonemes.append(phoneme)
                    current_time += 0.05
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
            
            result = TTSOutput(
                audio=audio.astype(np.float32),
                sample_rate=sample_rate,
                duration=duration,
                phonemes=phonemes,
                visemes=visemes,
                emotion_timeline=emotion_timeline,
                metadata={
                    'engine': 'mock',
                    'model': 'mock',
                    'voice_id': 'mock_voice',
                    'emotion': emotion.value if emotion else None,
                    'generation_time': time.time() - start_time,
                    'sample_file_used': sample_audio is not None
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating mock speech: {e}")
            raise
        finally:
            self.is_processing = False
    
    def _load_sample_audio(self) -> Optional[np.ndarray]:
        """Load sample audio file if available."""
        try:
            import wave
            from pathlib import Path
            
            sample_file = Path("voices/mock_sample.wav")
            if sample_file.exists():
                with wave.open(str(sample_file), 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    # Convert to float32 and normalize
                    audio = audio.astype(np.float32) / 32767.0
                    return audio
        except Exception as e:
            logger.debug(f"Could not load sample audio file: {e}")
        
        return None
    
    def _generate_mock_audio(self, text: str, emotion: Optional[EmotionType] = None) -> tuple[np.ndarray, int, float]:
        """Generate mock audio (fallback when sample file is not available)."""
        # Generate mock audio (sine wave with varying frequency)
        duration = len(text) * 0.1  # 100ms per character
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Create sine wave with frequency based on text
        base_freq = 200 + hash(text) % 200  # 200-400 Hz
        
        # Adjust frequency based on emotion
        if emotion:
            if emotion == EmotionType.HAPPY:
                base_freq *= 1.2
            elif emotion == EmotionType.SAD:
                base_freq *= 0.8
            elif emotion == EmotionType.ANGRY:
                base_freq *= 1.5
            elif emotion == EmotionType.CALM:
                base_freq *= 0.9
        
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Add some variation
        audio += np.sin(2 * np.pi * base_freq * 1.5 * t) * 0.1
        
        # Apply envelope
        envelope = np.exp(-t * 2)  # Decay envelope
        audio *= envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio, sample_rate, duration
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get engine capabilities."""
        return {
            'engine_type': 'mock',
            'emotion_support': True,
            'phoneme_output': True,
            'custom_voice_support': False,
            'languages': ['en'],
            'voices': ['mock_voice'],
            'speed_range': (0.5, 2.0),
            'pitch_range': (0.5, 2.0),
            'volume_range': (0.0, 1.0),
            'sample_rates': [22050, 44100],
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready()
        }
    
    def cleanup(self) -> None:
        """Cleanup engine resources."""
        self.is_initialized = False
        logger.info("Mock TTS engine cleaned up")


# Register the mock engine with the factory
from .tts_interface import TTSFactory
TTSFactory.register_engine('mock', MockTTSEngine)


# Example usage and testing
if __name__ == "__main__":
    async def test_mock_engine():
        """Test mock TTS engine functionality."""
        config = TTSConfig(
            engine_type='mock',
            model_path='mock',
            emotion_support=True,
            phoneme_output=True
        )
        
        engine = MockTTSEngine(config)
        await engine.initialize()
        
        # Test speech generation
        result = await engine.generate_speech(
            "Hello, this is a test of the mock TTS engine.",
            emotion=EmotionType.HAPPY
        )
        
        print(f"Generated audio: {len(result.audio)} samples")
        print(f"Sample rate: {result.sample_rate} Hz")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Phonemes: {len(result.phonemes) if result.phonemes else 0}")
        print(f"Visemes: {len(result.visemes) if result.visemes else 0}")
        print(f"Capabilities: {engine.get_capabilities()}")
        
        engine.cleanup()
    
    # Run test
    asyncio.run(test_mock_engine())
