"""
Coqui TTS engine implementation.
Provides high-quality speech synthesis with emotion support and phoneme alignment.
"""

import logging
import numpy as np
import asyncio
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from .tts_interface import TTSEngine
from core.types import TTSOutput, TTSConfig, Phoneme, Viseme, EmotionMarker, EmotionType

logger = logging.getLogger(__name__)


class CoquiTTSEngine(TTSEngine):
    """Coqui TTS engine implementation."""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.model = None
        self.vocoder = None
        self.speaker_manager = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Coqui TTS engine."""
        try:
            # Import TTS
            from TTS.api import TTS
            from TTS.utils.manage import ModelManager
            
            # Initialize model manager
            model_manager = ModelManager()
            
            # Load TTS model
            if self.config.custom_voice_path and Path(self.config.custom_voice_path).exists():
                # Load custom voice
                self.model = TTS(model_path=self.config.custom_voice_path)
                logger.info(f"Loaded custom voice from {self.config.custom_voice_path}")
            else:
                # Load default model
                model_name = self.config.model_path or "tts_models/en/ljspeech/tacotron2-DDC"
                self.model = TTS(model_name=model_name)
                logger.info(f"Loaded TTS model: {model_name}")
            
            # Load vocoder if specified
            if hasattr(self.config, 'vocoder') and self.config.vocoder:
                vocoder_name = self.config.vocoder
                self.vocoder = TTS(model_name=vocoder_name)
                logger.info(f"Loaded vocoder: {vocoder_name}")
            
            self.is_initialized = True
            logger.info("Coqui TTS engine initialized successfully")
            
        except ImportError:
            logger.error("Coqui TTS not available, using mock implementation")
            self.model = None
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise
    
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None,
                            **kwargs) -> TTSOutput:
        """Generate speech using Coqui TTS."""
        if not self.is_initialized:
            raise RuntimeError("Coqui TTS engine not initialized")
        
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
            
            # Add emotion if supported
            if emotion and self.config.emotion_support:
                generation_params['emotion'] = emotion.value
            
            # Generate speech
            if self.vocoder:
                # Use separate vocoder
                wav = self.model.tts(**generation_params)
                wav = self.vocoder.tts(wav)
            else:
                # Use built-in vocoder
                wav = self.model.tts(**generation_params)
            
            # Convert to numpy array
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # Ensure correct format
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8
            
            # Get sample rate
            sample_rate = getattr(self.model, 'synthesizer', {}).get('audio_config', {}).get('sample_rate', 22050)
            
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
                    'engine': 'coqui',
                    'model': self.config.model_path,
                    'voice_id': self.config.voice_id,
                    'emotion': emotion.value if emotion else None,
                    'generation_time': time.time() - start_time
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech with Coqui TTS: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def _extract_phonemes(self, text: str, audio: np.ndarray, 
                               sample_rate: int) -> List[Phoneme]:
        """Extract phonemes from generated audio."""
        try:
            # This is a simplified phoneme extraction
            # In practice, you would use a forced alignment tool like Montreal Forced Aligner
            
            # For now, create mock phonemes based on text length
            words = text.split()
            phonemes = []
            current_time = 0.0
            
            for word in words:
                # Estimate phoneme duration based on word length
                word_duration = len(word) * 0.1  # 100ms per character
                
                # Create mock phonemes for each character
                for char in word:
                    if char.isalpha():
                        phoneme = Phoneme(
                            symbol=char.upper(),
                            start_time=current_time,
                            end_time=current_time + 0.05,  # 50ms per phoneme
                            confidence=0.8
                        )
                        phonemes.append(phoneme)
                        current_time += 0.05
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
        duration = len(text) * 0.1  # 100ms per character
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Create sine wave with frequency based on text
        base_freq = 200 + hash(text) % 200  # 200-400 Hz
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
        
        return TTSOutput(
            audio=audio.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            phonemes=phonemes,
            visemes=visemes,
            emotion_timeline=emotion_timeline,
            metadata={
                'engine': 'coqui_mock',
                'model': 'mock',
                'voice_id': 'mock_voice',
                'emotion': emotion.value if emotion else None,
                'generation_time': 0.1
            }
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get engine capabilities."""
        return {
            'engine_type': 'coqui',
            'emotion_support': self.config.emotion_support,
            'phoneme_output': self.config.phoneme_output,
            'custom_voice_support': True,
            'languages': ['en'],  # Add more as needed
            'voices': ['default'],  # Add more as needed
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
        self.model = None
        self.vocoder = None
        self.speaker_manager = None
        logger.info("Coqui TTS engine cleaned up")


# Register the engine with the factory
from .tts_interface import TTSFactory
TTSFactory.register_engine('coqui', CoquiTTSEngine)


# Example usage and testing
if __name__ == "__main__":
    async def test_coqui_engine():
        """Test Coqui TTS engine functionality."""
        config = TTSConfig(
            engine_type='coqui',
            model_path='tts_models/en/ljspeech/tacotron2-DDC',
            emotion_support=True,
            phoneme_output=True
        )
        
        engine = CoquiTTSEngine(config)
        await engine.initialize()
        
        # Test speech generation
        result = await engine.generate_speech(
            "Hello, this is a test of the Coqui TTS engine.",
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
    asyncio.run(test_coqui_engine())
