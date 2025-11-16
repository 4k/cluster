"""
Simplified STT Manager - Vosk only with Mock fallback.
Replaces complex STT system with simple Vosk + Mock.
"""

import logging
import numpy as np
import asyncio
import json
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from core.types import AudioFrame, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""
    model_path: str = "models/vosk-model-small-en-us-0.22"
    language: str = "en"
    sample_rate: int = 16000
    partial_results: bool = True
    words: bool = True
    confidence_threshold: float = 0.5
    max_alternatives: int = 3
    grammar: Optional[list] = None
    timeout: float = 10.0


class SimpleSTTManager:
    """Simplified STT manager using only Vosk with Mock fallback."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.vosk_model = None
        self.recognizer = None
        self.is_initialized = False
        self.is_listening = False
        
        # State tracking
        self.current_utterance = ""
        self.partial_text = ""
        self.last_confidence = 0.0
        self.utterance_start_time = 0.0
        
        # Callbacks
        self.on_partial_result: Optional[Callable] = None
        self.on_final_result: Optional[Callable] = None
        self.on_utterance_start: Optional[Callable] = None
        self.on_utterance_end: Optional[Callable] = None
    
    async def initialize(self) -> None:
        """Initialize the STT manager."""
        try:
            # Try to initialize Vosk
            if await self._init_vosk():
                logger.info("STT manager initialized with Vosk")
            else:
                logger.warning("Vosk initialization failed, using mock mode")
                self.is_initialized = True  # Mock mode doesn't need initialization
                logger.info("STT manager initialized with mock mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT manager: {e}")
            # Fallback to mock mode
            self.is_initialized = True
            logger.info("STT manager initialized with mock mode (fallback)")
    
    async def _init_vosk(self) -> bool:
        """Initialize Vosk STT engine."""
        try:
            # Import Vosk
            import vosk
            
            # Load model
            self.vosk_model = vosk.Model(self.config.model_path)
            self.recognizer = vosk.KaldiRecognizer(
                self.vosk_model, 
                self.config.sample_rate
            )
            
            # Set partial results
            self.recognizer.SetWords(self.config.words)
            self.recognizer.SetPartialWords(self.config.partial_results)
            
            # Set grammar if provided
            if self.config.grammar:
                grammar = json.dumps(self.config.grammar)
                self.recognizer.SetGrammar(grammar)
            
            logger.info(f"Vosk STT initialized with model: {self.config.model_path}")
            return True
            
        except ImportError:
            logger.error("Vosk not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Vosk STT: {e}")
            return False
    
    def start_listening(self) -> None:
        """Start listening for speech."""
        if not self.is_initialized:
            logger.warning("STT not initialized")
            return
        
        self.is_listening = True
        self.current_utterance = ""
        self.partial_text = ""
        self.utterance_start_time = time.time()
        
        if self.on_utterance_start:
            asyncio.create_task(self._call_callback(self.on_utterance_start))
        
        logger.debug("STT started listening")
    
    def stop_listening(self) -> None:
        """Stop listening for speech."""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.on_utterance_end:
            asyncio.create_task(self._call_callback(self.on_utterance_end, self.current_utterance))
        
        logger.debug("STT stopped listening")
    
    def process_audio(self, audio_frame: AudioFrame) -> Dict[str, Any]:
        """Process audio frame and return STT results."""
        if not self.is_initialized or not self.is_listening:
            return {"text": "", "confidence": 0.0, "is_final": False}
        
        try:
            if self.recognizer is None:
                # Mock implementation for testing
                return self._mock_recognition(audio_frame)
            
            # Process audio with Vosk
            if self.recognizer.AcceptWaveform(audio_frame.data.tobytes()):
                # Final result
                result = json.loads(self.recognizer.Result())
                return self._process_final_result(result)
            else:
                # Partial result
                result = json.loads(self.recognizer.PartialResult())
                return self._process_partial_result(result)
                
        except Exception as e:
            logger.error(f"Error processing audio in STT: {e}")
            return {"text": "", "confidence": 0.0, "is_final": False}
    
    def _process_final_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process final recognition result."""
        text = result.get("text", "").strip()
        confidence = result.get("confidence", 0.0)
        
        if text and confidence >= self.config.confidence_threshold:
            self.current_utterance = text
            self.last_confidence = confidence
            
            # Create conversation turn
            turn = ConversationTurn(
                speaker_id="unknown",  # Will be updated by speaker ID
                text=text,
                timestamp=time.time(),
                confidence=confidence
            )
            
            # Call final result callback
            if self.on_final_result:
                asyncio.create_task(self._call_callback(self.on_final_result, turn))
            
            return {
                "text": text,
                "confidence": confidence,
                "is_final": True,
                "alternatives": result.get("alternatives", [])
            }
        else:
            return {"text": "", "confidence": 0.0, "is_final": True}
    
    def _process_partial_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process partial recognition result."""
        text = result.get("partial", "").strip()
        
        if text != self.partial_text:
            self.partial_text = text
            
            # Call partial result callback
            if self.on_partial_result:
                asyncio.create_task(self._call_callback(self.on_partial_result, text))
            
            return {
                "text": text,
                "confidence": 0.0,
                "is_final": False
            }
        
        return {"text": text, "confidence": 0.0, "is_final": False}
    
    def _mock_recognition(self, audio_frame: AudioFrame) -> Dict[str, Any]:
        """Mock speech recognition for testing."""
        # Simple mock: generate random text based on audio energy
        energy = np.mean(np.abs(audio_frame.data))
        
        if energy > 0.01:  # Audio detected
            # Generate mock text
            mock_texts = [
                "hello world",
                "how are you",
                "what time is it",
                "tell me a joke",
                "goodbye"
            ]
            
            # Select text based on energy level
            text_index = int(energy * 100) % len(mock_texts)
            text = mock_texts[text_index]
            confidence = min(energy * 10, 1.0)
            
            return {
                "text": text,
                "confidence": confidence,
                "is_final": True
            }
        else:
            return {"text": "", "confidence": 0.0, "is_final": False}
    
    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in STT callback: {e}")
    
    def set_callbacks(self, on_partial_result: Optional[Callable] = None,
                     on_final_result: Optional[Callable] = None,
                     on_utterance_start: Optional[Callable] = None,
                     on_utterance_end: Optional[Callable] = None) -> None:
        """Set callback functions for STT events."""
        self.on_partial_result = on_partial_result
        self.on_final_result = on_final_result
        self.on_utterance_start = on_utterance_start
        self.on_utterance_end = on_utterance_end
    
    def get_state(self) -> Dict[str, Any]:
        """Get current STT state."""
        return {
            "is_initialized": self.is_initialized,
            "is_listening": self.is_listening,
            "current_utterance": self.current_utterance,
            "partial_text": self.partial_text,
            "last_confidence": self.last_confidence,
            "utterance_start_time": self.utterance_start_time,
            "engine_type": "vosk" if self.vosk_model else "mock"
        }
    
    def reset(self) -> None:
        """Reset STT state."""
        self.current_utterance = ""
        self.partial_text = ""
        self.last_confidence = 0.0
        self.utterance_start_time = 0.0
        
        if self.recognizer:
            self.recognizer.Reset()
        
        logger.debug("STT state reset")
    
    def cleanup(self) -> None:
        """Cleanup STT resources."""
        self.is_initialized = False
        self.is_listening = False
        self.vosk_model = None
        self.recognizer = None
        logger.info("STT manager cleaned up")
