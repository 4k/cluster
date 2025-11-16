"""
Abstract TTS interface and base implementations.
Provides a unified interface for different TTS engines.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import asyncio
import time

from core.types import TTSOutput, TTSConfig, Phoneme, Viseme, EmotionMarker, EmotionType

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.is_initialized = False
        self.is_processing = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS engine."""
        pass
    
    @abstractmethod
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None, 
                            **kwargs) -> TTSOutput:
        """Generate speech from text."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get engine capabilities."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup engine resources."""
        pass
    
    def is_ready(self) -> bool:
        """Check if engine is ready for use."""
        return self.is_initialized and not self.is_processing
    
    def get_config(self) -> TTSConfig:
        """Get engine configuration."""
        return self.config


class TTSFactory:
    """Factory for creating TTS engine instances."""
    
    _engines = {}
    
    @classmethod
    def register_engine(cls, engine_type: str, engine_class: type) -> None:
        """Register a TTS engine class."""
        cls._engines[engine_type] = engine_class
        logger.debug(f"Registered TTS engine: {engine_type}")
    
    @classmethod
    def create_engine(cls, engine_type: str, config: TTSConfig) -> TTSEngine:
        """Create a TTS engine instance."""
        if engine_type not in cls._engines:
            raise ValueError(f"Unknown TTS engine type: {engine_type}")
        
        engine_class = cls._engines[engine_type]
        return engine_class(config)
    
    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of available engine types."""
        return list(cls._engines.keys())
    
    @classmethod
    def is_engine_available(cls, engine_type: str) -> bool:
        """Check if an engine type is available."""
        return engine_type in cls._engines


class TTSManager:
    """Manages TTS engine instances and provides unified interface."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.current_engine: Optional[TTSEngine] = None
        self.engines: Dict[str, TTSEngine] = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the TTS manager."""
        try:
            # Import mock engine to ensure it's registered
            from .mock_engine import MockTTSEngine
            
            # Try to create the requested engine
            if TTSFactory.is_engine_available(self.config.engine_type):
                self.current_engine = TTSFactory.create_engine(
                    self.config.engine_type, self.config
                )
                await self.current_engine.initialize()
                self.engines[self.config.engine_type] = self.current_engine
                logger.info(f"TTS manager initialized with engine: {self.config.engine_type}")
            else:
                # Fall back to mock engine
                logger.warning(f"TTS engine '{self.config.engine_type}' not available, falling back to mock")
                self.current_engine = TTSFactory.create_engine('mock', self.config)
                await self.current_engine.initialize()
                self.engines['mock'] = self.current_engine
                logger.info("TTS manager initialized with mock engine")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS manager: {e}")
            raise
    
    async def generate_speech(self, text: str, emotion: Optional[EmotionType] = None,
                            engine_type: Optional[str] = None, **kwargs) -> TTSOutput:
        """Generate speech using specified or current engine."""
        if not self.is_initialized:
            raise RuntimeError("TTS manager not initialized")
        
        # Use specified engine or current engine
        if engine_type and engine_type != self.config.engine_type:
            engine = await self._get_or_create_engine(engine_type)
        else:
            engine = self.current_engine
        
        if not engine or not engine.is_ready():
            raise RuntimeError("TTS engine not ready")
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            # Generate speech
            result = await engine.generate_speech(text, emotion, **kwargs)
            
            # Add timing information
            result.metadata["generation_time"] = time.time() - start_time
            result.metadata["engine_type"] = engine_type or self.config.engine_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def _get_or_create_engine(self, engine_type: str) -> TTSEngine:
        """Get or create an engine instance."""
        if engine_type in self.engines:
            return self.engines[engine_type]
        
        # Create new engine instance
        engine_config = TTSConfig(
            engine_type=engine_type,
            model_path=self.config.model_path,
            voice_id=self.config.voice_id,
            speed=self.config.speed,
            pitch=self.config.pitch,
            volume=self.config.volume,
            emotion_support=self.config.emotion_support,
            phoneme_output=self.config.phoneme_output,
            custom_voice_path=self.config.custom_voice_path,
            additional_params=self.config.additional_params
        )
        
        # Try to create the requested engine, fall back to mock if not available
        if TTSFactory.is_engine_available(engine_type):
            engine = TTSFactory.create_engine(engine_type, engine_config)
        else:
            logger.warning(f"TTS engine '{engine_type}' not available, using mock")
            engine = TTSFactory.create_engine('mock', engine_config)
        
        await engine.initialize()
        
        self.engines[engine_type] = engine
        return engine
    
    def switch_engine(self, engine_type: str) -> None:
        """Switch to a different TTS engine."""
        if engine_type not in self.engines:
            raise ValueError(f"Engine {engine_type} not available")
        
        self.current_engine = self.engines[engine_type]
        self.config.engine_type = engine_type
        
        logger.info(f"Switched to TTS engine: {engine_type}")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine types."""
        return TTSFactory.get_available_engines()
    
    def get_current_engine_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of current engine."""
        if not self.current_engine:
            return {}
        return self.current_engine.get_capabilities()
    
    def get_state(self) -> Dict[str, Any]:
        """Get TTS manager state."""
        return {
            "is_initialized": self.is_initialized,
            "current_engine": self.config.engine_type,
            "is_processing": self.is_processing,
            "available_engines": list(self.engines.keys()),
            "engines_ready": {
                engine_type: engine.is_ready() 
                for engine_type, engine in self.engines.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup TTS manager and all engines."""
        for engine in self.engines.values():
            engine.cleanup()
        
        self.engines.clear()
        self.current_engine = None
        self.is_initialized = False
        
        logger.info("TTS manager cleaned up")


# Utility functions for phoneme and viseme processing
def phoneme_to_viseme(phoneme: str) -> str:
    """Convert phoneme to viseme for lip-sync."""
    # Basic phoneme to viseme mapping
    phoneme_viseme_map = {
        # Vowels
        'AA': 'a_shape', 'AE': 'a_shape', 'AH': 'a_shape',
        'AO': 'o_shape', 'AW': 'o_shape', 'AY': 'a_shape',
        'EH': 'e_shape', 'ER': 'e_shape', 'EY': 'e_shape',
        'IH': 'i_shape', 'IY': 'i_shape',
        'OW': 'o_shape', 'OY': 'o_shape',
        'UH': 'u_shape', 'UW': 'u_shape',
        
        # Consonants
        'B': 'm_shape', 'P': 'm_shape',
        'D': 't_shape', 'T': 't_shape',
        'F': 'f_shape', 'V': 'f_shape',
        'G': 'g_shape', 'K': 'g_shape',
        'L': 'l_shape',
        'M': 'm_shape',
        'N': 'n_shape', 'NG': 'n_shape',
        'R': 'r_shape',
        'S': 's_shape', 'Z': 's_shape',
        'SH': 'sh_shape', 'ZH': 'sh_shape',
        'TH': 'th_shape',
        'W': 'w_shape',
        'Y': 'y_shape',
        
        # Default
        'SIL': 'closed', 'SPN': 'closed'
    }
    
    return phoneme_viseme_map.get(phoneme.upper(), 'closed')


def create_visemes_from_phonemes(phonemes: List[Phoneme]) -> List[Viseme]:
    """Create visemes from phonemes for lip-sync animation."""
    from core.types import Viseme, MouthShape
    
    visemes = []
    for phoneme in phonemes:
        viseme_shape = phoneme_to_viseme(phoneme.symbol)
        
        try:
            mouth_shape = MouthShape(viseme_shape)
        except ValueError:
            mouth_shape = MouthShape.CLOSED
        
        viseme = Viseme(
            shape=mouth_shape,
            start_time=phoneme.start_time,
            end_time=phoneme.end_time,
            intensity=phoneme.confidence
        )
        visemes.append(viseme)
    
    return visemes


def create_emotion_timeline(text: str, emotion: EmotionType, 
                           duration: float) -> List[EmotionMarker]:
    """Create emotion timeline for the entire speech."""
    from core.types import EmotionMarker
    
    return [EmotionMarker(
        emotion=emotion,
        start_time=0.0,
        end_time=duration,
        intensity=1.0
    )]


# Example usage and testing
if __name__ == "__main__":
    async def test_tts_interface():
        """Test TTS interface functionality."""
        # This would be used with actual TTS engine implementations
        print("TTS interface module loaded successfully")
        print(f"Available engines: {TTSFactory.get_available_engines()}")
    
    # Run test
    asyncio.run(test_tts_interface())
