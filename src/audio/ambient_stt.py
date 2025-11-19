"""
Ambient Speech-to-Text (STT) for continuous listening.
Provides always-on speech recognition with mode tagging (ambient vs wakeword).
"""

import logging
import numpy as np
import asyncio
import json
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import time

from core.types import AudioFrame, ConversationTurn
from core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


@dataclass
class AmbientSTTConfig:
    """Configuration for Ambient Speech-to-Text."""
    enabled: bool = True
    model_path: str = "models/vosk-model-small-en-us-0.15"  # Lighter model for ambient
    language: str = "en"
    sample_rate: int = 16000
    partial_results: bool = True
    words: bool = True
    confidence_threshold: float = 0.3  # Lower threshold for ambient mode
    max_alternatives: int = 3
    grammar: Optional[List[str]] = None
    timeout: float = 10.0
    wake_word_timeout: float = 5.0  # Seconds to stay in 'wakeword' mode
    frame_skip: int = 1  # Process every Nth frame (1=all, 2=every other)
    min_confidence: float = 0.3  # Minimum confidence to emit result


class AmbientSTT:
    """
    Continuous speech recognition for ambient listening.
    Tags results as 'ambient' or 'wakeword' based on wake word state.
    Always processes audio - no VAD gating.
    """

    def __init__(self, config: AmbientSTTConfig):
        self.config = config
        self.model = None
        self.recognizer = None
        self.is_initialized = False

        # Wake word state tracking
        self.is_wake_word_active = False
        self.last_wake_word_time = 0.0
        self.wake_word_timeout = config.wake_word_timeout

        # State tracking
        self.current_utterance = ""
        self.partial_text = ""
        self.last_confidence = 0.0
        self.utterance_start_time = 0.0
        self.frame_counter = 0  # For frame skipping

        # Callbacks
        self.on_ambient_result: Optional[Callable] = None
        self.on_wakeword_result: Optional[Callable] = None
        self.on_partial_result: Optional[Callable] = None

    async def initialize(self) -> None:
        """Initialize the ambient STT model."""
        try:
            # Import Vosk
            import vosk

            # Set Vosk log level to suppress warnings
            vosk.SetLogLevel(-1)

            # Load model
            logger.info(f"Loading ambient STT model from: {self.config.model_path}")
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
            logger.info(f"Ambient STT initialized with model: {self.config.model_path}")

        except ImportError:
            logger.error("Vosk not available for ambient STT")
            self.model = None
            self.recognizer = None
            self.is_initialized = False
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ambient STT: {e}")
            raise

    def on_wake_word_detected(self, timestamp: float) -> None:
        """
        Called when wake word detector fires.
        Switches mode from 'ambient' to 'wakeword'.
        """
        self.is_wake_word_active = True
        self.last_wake_word_time = timestamp
        logger.debug(f"Ambient STT: Wake word activated at {timestamp:.2f}s")

    def _determine_mode(self) -> str:
        """
        Determine if current speech is ambient or wakeword-triggered.

        Returns:
            'wakeword' if within timeout window, 'ambient' otherwise
        """
        if self.is_wake_word_active:
            # Check if we're still within the timeout window
            elapsed = time.time() - self.last_wake_word_time
            if elapsed < self.wake_word_timeout:
                return 'wakeword'
            else:
                # Timeout expired, reset to ambient
                self.is_wake_word_active = False
                logger.debug("Ambient STT: Wake word timeout, reverting to ambient mode")
                return 'ambient'
        return 'ambient'

    async def process_audio(self, audio_frame: AudioFrame) -> Optional[Dict[str, Any]]:
        """
        Process audio frame continuously (no VAD gating).

        Args:
            audio_frame: Audio frame to process

        Returns:
            Dictionary with STT results and mode tag, or None if no result
        """
        if not self.is_initialized or not self.recognizer:
            return None

        # Implement frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % self.config.frame_skip != 0:
            return None

        try:
            # Process audio with Vosk
            if self.recognizer.AcceptWaveform(audio_frame.data.tobytes()):
                # Final result
                result = json.loads(self.recognizer.Result())
                return await self._process_final_result(result, audio_frame.timestamp)
            else:
                # Partial result
                result = json.loads(self.recognizer.PartialResult())
                await self._process_partial_result(result)
                return None

        except Exception as e:
            logger.error(f"Error processing audio in ambient STT: {e}")
            return None

    async def _process_final_result(self, result: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        """Process final recognition result with mode tagging."""
        text = result.get("text", "").strip()
        confidence = result.get("confidence", 0.0)

        # Filter by minimum confidence
        if not text or confidence < self.config.min_confidence:
            return None

        # Determine mode tag
        mode = self._determine_mode()

        self.current_utterance = text
        self.last_confidence = confidence

        # Create conversation turn
        turn = ConversationTurn(
            speaker_id="unknown",
            text=text,
            timestamp=timestamp,
            confidence=confidence
        )

        # Create result dictionary
        result_data = {
            'text': text,
            'confidence': confidence,
            'mode': mode,  # 'ambient' or 'wakeword'
            'timestamp': timestamp,
            'turn': turn,
            'alternatives': result.get("alternatives", [])
        }

        # Call appropriate callback based on mode
        if mode == 'wakeword' and self.on_wakeword_result:
            await self._call_callback(self.on_wakeword_result, result_data)
        elif mode == 'ambient' and self.on_ambient_result:
            await self._call_callback(self.on_ambient_result, result_data)

        # Emit event with mode tag
        event_type = (
            EventType.WAKEWORD_SPEECH_DETECTED
            if mode == 'wakeword'
            else EventType.AMBIENT_SPEECH_DETECTED
        )

        await emit_event(event_type, {
            "text": text,
            "confidence": confidence,
            "mode": mode,
            "timestamp": timestamp,
            "duration": time.time() - self.utterance_start_time
        })

        logger.info(f"Ambient STT [{mode}]: '{text}' (confidence: {confidence:.3f})")

        return result_data

    async def _process_partial_result(self, result: Dict[str, Any]) -> None:
        """Process partial recognition result."""
        text = result.get("partial", "").strip()

        if text and text != self.partial_text:
            self.partial_text = text

            # Call partial result callback
            if self.on_partial_result:
                mode = self._determine_mode()
                await self._call_callback(self.on_partial_result, {
                    'text': text,
                    'mode': mode,
                    'is_partial': True
                })

    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in ambient STT callback: {e}")

    def set_callbacks(self,
                     on_ambient_result: Optional[Callable] = None,
                     on_wakeword_result: Optional[Callable] = None,
                     on_partial_result: Optional[Callable] = None) -> None:
        """Set callback functions for ambient STT events."""
        self.on_ambient_result = on_ambient_result
        self.on_wakeword_result = on_wakeword_result
        self.on_partial_result = on_partial_result

    def get_state(self) -> Dict[str, Any]:
        """Get current ambient STT state."""
        mode = self._determine_mode()
        return {
            "is_initialized": self.is_initialized,
            "current_mode": mode,
            "is_wake_word_active": self.is_wake_word_active,
            "current_utterance": self.current_utterance,
            "partial_text": self.partial_text,
            "last_confidence": self.last_confidence,
            "utterance_start_time": self.utterance_start_time,
            "wake_word_time_remaining": max(0, self.wake_word_timeout - (time.time() - self.last_wake_word_time)) if self.is_wake_word_active else 0
        }

    def reset(self) -> None:
        """Reset ambient STT state."""
        self.current_utterance = ""
        self.partial_text = ""
        self.last_confidence = 0.0
        self.utterance_start_time = 0.0
        self.is_wake_word_active = False
        self.last_wake_word_time = 0.0

        if self.recognizer:
            self.recognizer.Reset()

        logger.debug("Ambient STT state reset")

    def cleanup(self) -> None:
        """Cleanup ambient STT resources."""
        self.is_initialized = False
        self.model = None
        self.recognizer = None
        logger.info("Ambient STT cleaned up")


class MockAmbientSTT:
    """Mock Ambient STT for testing and development."""

    def __init__(self, config: AmbientSTTConfig):
        self.config = config
        self.is_initialized = True

        # Wake word state
        self.is_wake_word_active = False
        self.last_wake_word_time = 0.0
        self.wake_word_timeout = config.wake_word_timeout

        # Mock state tracking
        self.utterance_count = 0
        self.last_detection_time = 0.0
        self.detection_interval = 20.0  # Trigger every 20 seconds
        self.current_utterance = ""

        # Callbacks
        self.on_ambient_result: Optional[Callable] = None
        self.on_wakeword_result: Optional[Callable] = None
        self.on_partial_result: Optional[Callable] = None

        logger.info("Mock Ambient STT initialized")

    async def initialize(self) -> None:
        """Mock initialization."""
        logger.info("Mock Ambient STT initialized - will generate periodic detections")

    def on_wake_word_detected(self, timestamp: float) -> None:
        """Mock wake word activation."""
        self.is_wake_word_active = True
        self.last_wake_word_time = timestamp
        logger.debug(f"Mock Ambient STT: Wake word activated at {timestamp:.2f}s")

    def _determine_mode(self) -> str:
        """Determine mock mode."""
        if self.is_wake_word_active:
            elapsed = time.time() - self.last_wake_word_time
            if elapsed < self.wake_word_timeout:
                return 'wakeword'
            else:
                self.is_wake_word_active = False
                return 'ambient'
        return 'ambient'

    async def process_audio(self, audio_frame: AudioFrame) -> Optional[Dict[str, Any]]:
        """Mock audio processing - triggers detection periodically."""
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time

        if time_since_last >= self.detection_interval:
            # Time to trigger a detection
            self.utterance_count += 1
            import random

            mock_texts = [
                "ambient listening test",
                "hello from the background",
                "continuous recognition active",
                "ambient mode working",
                "speech detected automatically"
            ]

            text = mock_texts[self.utterance_count % len(mock_texts)]
            confidence = random.uniform(0.5, 0.9)
            mode = self._determine_mode()

            self.current_utterance = text
            self.last_detection_time = current_time

            # Create conversation turn
            turn = ConversationTurn(
                speaker_id="unknown",
                text=text,
                timestamp=current_time,
                confidence=confidence
            )

            result_data = {
                'text': text,
                'confidence': confidence,
                'mode': mode,
                'timestamp': current_time,
                'turn': turn
            }

            # Call callback
            if mode == 'wakeword' and self.on_wakeword_result:
                await self._call_callback(self.on_wakeword_result, result_data)
            elif mode == 'ambient' and self.on_ambient_result:
                await self._call_callback(self.on_ambient_result, result_data)

            # Emit event
            event_type = (
                EventType.WAKEWORD_SPEECH_DETECTED
                if mode == 'wakeword'
                else EventType.AMBIENT_SPEECH_DETECTED
            )

            await emit_event(event_type, {
                "text": text,
                "confidence": confidence,
                "mode": mode,
                "timestamp": current_time,
                "duration": 0.0
            })

            logger.info(f"Mock Ambient STT [{mode}]: '{text}' (confidence: {confidence:.3f})")

            return result_data

        return None

    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in mock ambient STT callback: {e}")

    def set_callbacks(self,
                     on_ambient_result: Optional[Callable] = None,
                     on_wakeword_result: Optional[Callable] = None,
                     on_partial_result: Optional[Callable] = None) -> None:
        """Set callback functions."""
        self.on_ambient_result = on_ambient_result
        self.on_wakeword_result = on_wakeword_result
        self.on_partial_result = on_partial_result

    def get_state(self) -> Dict[str, Any]:
        """Get mock state."""
        mode = self._determine_mode()
        return {
            "is_initialized": True,
            "current_mode": mode,
            "is_wake_word_active": self.is_wake_word_active,
            "current_utterance": self.current_utterance,
            "next_detection_in": max(0, self.detection_interval - (time.time() - self.last_detection_time))
        }

    def reset(self) -> None:
        """Mock reset."""
        self.current_utterance = ""
        self.is_wake_word_active = False
        self.last_wake_word_time = 0.0
        logger.debug("Mock Ambient STT state reset")

    def cleanup(self) -> None:
        """Mock cleanup."""
        self.is_initialized = False
        logger.info("Mock Ambient STT cleaned up")


def create_ambient_stt(config: AmbientSTTConfig, mock: bool = False) -> Any:
    """
    Create ambient STT instance based on configuration.

    Args:
        config: Ambient STT configuration
        mock: If True, create mock implementation

    Returns:
        AmbientSTT or MockAmbientSTT instance
    """
    if mock:
        return MockAmbientSTT(config)

    # Try to create real implementation, fall back to mock if it fails
    try:
        return AmbientSTT(config)
    except ImportError:
        logger.warning("Vosk not available for ambient STT, falling back to mock implementation")
        return MockAmbientSTT(config)
    except Exception as e:
        logger.warning(f"Failed to create ambient STT: {e}, falling back to mock implementation")
        return MockAmbientSTT(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_ambient_stt():
        """Test ambient STT functionality."""
        config = AmbientSTTConfig(
            model_path="models/vosk-model-small-en-us-0.15",
            confidence_threshold=0.3,
            wake_word_timeout=5.0
        )

        # Create ambient STT (will use mock if Vosk not available)
        ambient_stt = create_ambient_stt(config, mock=True)
        await ambient_stt.initialize()

        # Set up callbacks
        async def on_ambient(result):
            print(f"[AMBIENT] {result['text']} (confidence: {result['confidence']:.3f})")

        async def on_wakeword(result):
            print(f"[WAKEWORD] {result['text']} (confidence: {result['confidence']:.3f})")

        ambient_stt.set_callbacks(
            on_ambient_result=on_ambient,
            on_wakeword_result=on_wakeword
        )

        # Simulate wake word detection
        print("Simulating wake word detection...")
        ambient_stt.on_wake_word_detected(time.time())

        # Process some audio frames
        print("Processing audio frames...")
        for i in range(30):
            audio_data = np.random.randn(1600).astype(np.float32) * 0.1
            audio_frame = AudioFrame(
                data=audio_data,
                sample_rate=16000,
                timestamp=time.time(),
                duration=0.1
            )

            result = await ambient_stt.process_audio(audio_frame)
            if result:
                print(f"Result: {result}")

            await asyncio.sleep(0.1)

        print(f"State: {ambient_stt.get_state()}")
        ambient_stt.cleanup()

    # Run test
    asyncio.run(test_ambient_stt())
