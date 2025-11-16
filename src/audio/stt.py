"""
Speech-to-Text (STT) using Vosk.
Provides real-time speech recognition with streaming support.
"""

import logging
import numpy as np
import asyncio
import json
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass
import time

from core.types import AudioFrame, ConversationTurn
from core.event_bus import EventBus, EventType, emit_event

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
    grammar: Optional[List[str]] = None
    timeout: float = 10.0


class VoskSTT:
    """Vosk-based Speech-to-Text implementation."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.model = None
        self.recognizer = None
        self.is_initialized = False
        
        # State tracking
        self.is_listening = False
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
        """Initialize the STT model."""
        try:
            # Import Vosk
            import vosk
            
            # Load model
            self.model = vosk.Model(self.config.model_path)
            self.recognizer = vosk.KaldiRecognizer(
                self.model, 
                self.config.sample_rate
            )
            
            # Set partial results
            self.recognizer.SetWords(self.config.words)
            self.recognizer.SetPartialWords(self.config.partial_results)
            
            # Set grammar if provided
            if self.config.grammar:
                grammar = json.dumps(self.config.grammar)
                self.recognizer.SetGrammar(grammar)
            
            self.is_initialized = True
            logger.info(f"Vosk STT initialized with model: {self.config.model_path}")
            
        except ImportError:
            logger.error("Vosk not available, using mock implementation")
            self.model = None
            self.recognizer = None
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Vosk STT: {e}")
            raise
    
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
        """
        Process audio frame and return STT results.
        
        Args:
            audio_frame: Audio frame to process
            
        Returns:
            Dictionary with STT results
        """
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
            
            # Emit event
            asyncio.create_task(emit_event(EventType.SPEECH_DETECTED, {
                "text": text,
                "confidence": confidence,
                "timestamp": time.time(),
                "duration": time.time() - self.utterance_start_time
            }))
            
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
            "utterance_start_time": self.utterance_start_time
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
        self.model = None
        self.recognizer = None
        logger.info("STT cleaned up")


class MockSTT:
    """Mock STT for testing and development."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.is_initialized = True
        self.is_listening = False
        
        # Callbacks (like VoskSTT)
        self.on_partial_result: Optional[Callable] = None
        self.on_final_result: Optional[Callable] = None
        self.on_utterance_start: Optional[Callable] = None
        self.on_utterance_end: Optional[Callable] = None
        
        # Mock state tracking
        self.utterance_count = 0
        self.last_detection_time = 0.0
        # Trigger every 30 seconds
        self.detection_interval = 30.0
        self.current_utterance = ""
        self._background_task = None
        
        logger.info("Mock STT initialized - will trigger detections")
    
    async def initialize(self) -> None:
        """Mock initialization."""
        logger.info("Mock STT initialized - using hardcoded text responses")
        
        # In mock mode, auto-start listening and create background processing task
        self.start_listening()
        logger.info("Mock STT auto-started listening")
        
        # Create background task to process mock audio frames
        self._background_task = asyncio.create_task(self._mock_audio_processor())
    
    def start_listening(self) -> None:
        """Mock start listening."""
        self.is_listening = True
        self.last_detection_time = time.time()
        logger.debug("Mock STT started listening")
        
        # Trigger utterance start callback
        if self.on_utterance_start:
            asyncio.create_task(self._call_callback(self.on_utterance_start))
    
    def stop_listening(self) -> None:
        """Mock stop listening."""
        self.is_listening = False
        
        # Trigger utterance end callback
        if self.on_utterance_end:
            asyncio.create_task(self._call_callback(self.on_utterance_end, self.current_utterance))
        
        logger.debug("Mock STT stopped listening")
    
    async def _mock_audio_processor(self) -> None:
        """Background task to generate mock audio frames and process them."""
        while self.is_initialized:
            try:
                # Generate a dummy audio frame every 100ms
                await asyncio.sleep(0.1)
                
                if not self.is_listening:
                    continue
                
                # Create a dummy audio frame
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
                audio_frame = AudioFrame(
                    data=dummy_audio,
                    sample_rate=16000,
                    timestamp=time.time(),
                    duration=1.0
                )
                
                # Process the frame (this will trigger detection every 10 seconds)
                self.process_audio(audio_frame)
                
            except asyncio.CancelledError:
                logger.debug("Mock STT background processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in mock audio processor: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    def process_audio(self, audio_frame: AudioFrame) -> Dict[str, Any]:
        """Mock audio processing - triggers detection every 10 seconds."""
        if not self.is_listening:
            return {"text": "", "confidence": 0.0, "is_final": False}
        
        # Check if it's time to trigger a detection (every 10 seconds)
        current_time = time.time()
        time_since_last_detection = current_time - self.last_detection_time
        
        if time_since_last_detection >= self.detection_interval:
            # Time to trigger a detection
            self.utterance_count += 1
            import random
            
            mock_texts = [
                "hello world",
                "how are you",
                "what time is it",
                "tell me a joke",
                "goodbye",
                "what's the weather like",
                "turn on the lights",
                "play some music",
                "set a timer for five minutes",
                "what's your name",
                "help me with something",
                "thank you very much",
                "that's interesting",
                "I don't understand",
                "can you repeat that",
                "yes that's correct",
                "no I don't think so",
                "maybe later",
                "I'm not sure",
                "that sounds good"
            ]
            
            text = mock_texts[self.utterance_count % len(mock_texts)]
            confidence = random.uniform(0.7, 1.0)
            
            # Update state
            self.current_utterance = text
            self.last_detection_time = current_time
            
            # Create conversation turn
            turn = ConversationTurn(
                speaker_id="unknown",
                text=text,
                timestamp=current_time,
                confidence=confidence
            )
            
            # Call final result callback (like VoskSTT does)
            if self.on_final_result:
                asyncio.create_task(self._call_callback(self.on_final_result, turn))
            
            # Emit event (like VoskSTT does)
            asyncio.create_task(emit_event(EventType.SPEECH_DETECTED, {
                "text": text,
                "confidence": confidence,
                "timestamp": current_time,
                "duration": time_since_last_detection
            }))
            
            logger.info(f"Mock STT detected: '{text}' (confidence: {confidence:.3f})")
            
            return {
                "text": text,
                "confidence": confidence,
                "is_final": True
            }
        
        return {"text": "", "confidence": 0.0, "is_final": False}
    
    def set_callbacks(self, on_partial_result: Optional[Callable] = None,
                     on_final_result: Optional[Callable] = None,
                     on_utterance_start: Optional[Callable] = None,
                     on_utterance_end: Optional[Callable] = None) -> None:
        """Set callback functions for STT events (like VoskSTT)."""
        self.on_partial_result = on_partial_result
        self.on_final_result = on_final_result
        self.on_utterance_start = on_utterance_start
        self.on_utterance_end = on_utterance_end
    
    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely (like VoskSTT)."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in Mock STT callback: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get mock state."""
        return {
            "is_initialized": True,
            "is_listening": self.is_listening,
            "current_utterance": self.current_utterance,
            "partial_text": "",
            "last_confidence": 0.0,
            "utterance_start_time": self.last_detection_time,
            "next_detection_in": max(0, self.detection_interval - (time.time() - self.last_detection_time))
        }
    
    def reset(self) -> None:
        """Mock reset."""
        self.current_utterance = ""
        self.last_detection_time = 0.0
        logger.debug("Mock STT state reset")
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        self.is_initialized = False
        self.is_listening = False
        
        # Cancel background task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
        
        logger.info("Mock STT cleaned up")


def create_stt(config: STTConfig, mock: bool = False) -> Union[VoskSTT, MockSTT]:
    """Create STT instance based on configuration."""
    if mock:
        return MockSTT(config)
    
    # Try to create real implementation, fall back to mock if it fails
    try:
        stt = VoskSTT(config)
        # Test if Vosk is available by trying to import it
        import vosk
        return stt
    except ImportError:
        logger.warning("Vosk not available, falling back to mock implementation")
        return MockSTT(config)
    except Exception as e:
        logger.warning(f"Failed to create Vosk STT: {e}, falling back to mock implementation")
        return MockSTT(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_stt():
        """Test STT functionality."""
        config = STTConfig(
            model_path="models/vosk-model-small-en-us-0.22",
            confidence_threshold=0.5
        )
        stt = create_stt(config, mock=True)
        
        await stt.initialize()
        stt.start_listening()
        
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
        result = stt.process_audio(audio_frame)
        
        print(f"Recognized text: '{result['text']}'")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Is final: {result['is_final']}")
        print(f"STT state: {stt.get_state()}")
        
        stt.stop_listening()
        stt.cleanup()
    
    # Run test
    asyncio.run(test_stt())
