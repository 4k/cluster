"""
Audio Manager for coordinating audio input/output and processing.
Manages the audio pipeline including capture, VAD, wake word detection, and STT.
"""

import logging
import asyncio
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import time
import threading
from queue import Queue

from core.types import AudioFrame, AudioConfig
from core.event_bus import EventBus, EventType, emit_event
from audio.vad import create_vad, VADConfig, SileroVAD, MockVAD
from audio.wake_word import create_wake_word_detector, WakeWordConfig, SileroWakeWordDetector, MockWakeWordDetector
from audio.stt import create_stt, STTConfig, VoskSTT, MockSTT

logger = logging.getLogger(__name__)


@dataclass
class AudioManagerConfig:
    """Configuration for Audio Manager."""
    audio: AudioConfig
    vad: VADConfig
    wake_word: WakeWordConfig
    stt: STTConfig
    mock_mode: bool = False
    enable_vad: bool = True
    enable_wake_word: bool = True
    enable_stt: bool = True


class AudioManager:
    """Manages audio input/output and processing pipeline."""
    
    def __init__(self, config: AudioManagerConfig):
        self.config = config
        self.is_running = False
        self.is_listening = False
        
        # Audio components
        self.vad: Optional[SileroVAD] = None
        self.wake_word_detector: Optional[SileroWakeWordDetector] = None
        self.stt: Optional[VoskSTT] = None
        
        # Audio stream
        self.audio_stream: Optional[sd.InputStream] = None
        self.audio_queue: Queue = Queue(maxsize=100)
        
        # State tracking
        self.current_audio_frame: Optional[AudioFrame] = None
        self.audio_buffer: List[np.ndarray] = []
        self.buffer_size = config.audio.buffer_size
        self.sample_rate = config.audio.sample_rate
        
        # Callbacks
        self.on_audio_data: Optional[Callable] = None
        self.on_speech_detected: Optional[Callable] = None
        self.on_wake_word_detected: Optional[Callable] = None
        self.on_text_recognized: Optional[Callable] = None
        
        # Threading
        self._audio_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    async def initialize(self) -> None:
        """Initialize audio manager and all components."""
        try:
            # Initialize VAD
            if self.config.enable_vad:
                self.vad = create_vad(self.config.vad, mock=self.config.mock_mode)
                await self.vad.initialize()
                self._setup_vad_callbacks()
            
            # Initialize wake word detector
            if self.config.enable_wake_word:
                self.wake_word_detector = create_wake_word_detector(
                    self.config.wake_word, mock=self.config.mock_mode
                )
                await self.wake_word_detector.initialize()
                self._setup_wake_word_callbacks()
            
            # Initialize STT
            if self.config.enable_stt:
                self.stt = create_stt(self.config.stt, mock=self.config.mock_mode)
                await self.stt.initialize()
                self._setup_stt_callbacks()
            
            # Initialize audio stream
            await self._initialize_audio_stream()
            
            logger.info("Audio manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio manager: {e}")
            raise
    
    def _is_docker_environment(self) -> bool:
        """Check if running in Docker environment."""
        try:
            with open('/proc/1/cgroup', 'r') as f:
                return 'docker' in f.read().lower()
        except:
            return False
    
    async def _initialize_audio_stream(self) -> None:
        """Initialize audio input stream."""
        try:
            # Check if we're in mock mode or Docker environment
            if self.config.mock_mode or self._is_docker_environment():
                logger.info("Running in mock/Docker mode - skipping audio stream initialization")
                self.audio_stream = None
                return
            
            # Configure audio stream
            stream_config = {
                'samplerate': self.sample_rate,
                'channels': self.config.audio.channels,
                'dtype': np.float32,
                'blocksize': self.buffer_size,
                'callback': self._audio_callback,
                'device': self.config.audio.device
            }
            
            # Create audio stream
            self.audio_stream = sd.InputStream(**stream_config)
            
            logger.info(f"Audio stream initialized: {self.sample_rate}Hz, {self.buffer_size} samples")
            
        except Exception as e:
            logger.warning(f"Failed to initialize audio stream: {e}")
            logger.info("Continuing without audio input (mock mode)")
            self.audio_stream = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Audio input callback function."""
        try:
            # Check for audio underruns
            if status.input_underflow:
                logger.warning("Audio input underflow detected")
                asyncio.create_task(emit_event(EventType.ERROR_OCCURRED, {
                    "error": "audio_input_underflow",
                    "component": "audio_manager"
                }))
            
            # Convert to mono if stereo
            if indata.ndim > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # Create audio frame
            audio_frame = AudioFrame(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=time.inputBufferAdcTime,
                duration=len(audio_data) / self.sample_rate,
                channels=1
            )
            
            # Add to queue for processing
            if not self.audio_queue.full():
                self.audio_queue.put(audio_frame)
            else:
                logger.warning("Audio queue full, dropping frame")
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
    
    def _setup_vad_callbacks(self) -> None:
        """Setup VAD callbacks."""
        if not self.vad:
            return
        
        async def on_speech_start(timestamp: float, confidence: float):
            logger.debug(f"Speech started at {timestamp:.2f}s, confidence: {confidence:.3f}")
            self.is_listening = True
            
            # Start STT if enabled
            if self.stt:
                self.stt.start_listening()
            
            # Call user callback
            if self.on_speech_detected:
                await self._call_callback(self.on_speech_detected, timestamp, confidence)
        
        async def on_speech_end(timestamp: float, duration: float):
            logger.debug(f"Speech ended at {timestamp:.2f}s, duration: {duration:.2f}s")
            self.is_listening = False
            
            # Stop STT if enabled
            if self.stt:
                self.stt.stop_listening()
        
        async def on_confidence_change(confidence: float, duration: float):
            logger.debug(f"Speech confidence: {confidence:.3f}, duration: {duration:.2f}s")
        
        self.vad.set_callbacks(
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_confidence_change=on_confidence_change
        )
    
    def _setup_wake_word_callbacks(self) -> None:
        """Setup wake word detector callbacks."""
        if not self.wake_word_detector:
            return
        
        async def on_wake_word_detected(activation_record: Dict[str, Any]):
            logger.info(f"Wake word detected: {activation_record['wake_word']} "
                       f"(confidence: {activation_record['confidence']:.3f})")
            
            # Call user callback
            if self.on_wake_word_detected:
                await self._call_callback(self.on_wake_word_detected, activation_record)
        
        async def on_activation_start(wake_word: str, confidence: float, timestamp: float):
            logger.debug(f"Wake word activation started: {wake_word}")
        
        async def on_activation_end(activation_record: Dict[str, Any]):
            logger.debug(f"Wake word activation ended: {activation_record['wake_word']}")
        
        self.wake_word_detector.set_callbacks(
            on_wake_word_detected=on_wake_word_detected,
            on_activation_start=on_activation_start,
            on_activation_end=on_activation_end
        )
    
    def _setup_stt_callbacks(self) -> None:
        """Setup STT callbacks."""
        if not self.stt:
            return
        
        async def on_partial_result(text: str):
            logger.debug(f"Partial STT result: '{text}'")
        
        async def on_final_result(turn):
            logger.info(f"Final STT result: '{turn.text}' (confidence: {turn.confidence:.3f})")
            
            # Call user callback
            if self.on_text_recognized:
                await self._call_callback(self.on_text_recognized, turn)
        
        async def on_utterance_start():
            logger.debug("STT utterance started")
        
        async def on_utterance_end(text: str):
            logger.debug(f"STT utterance ended: '{text}'")
        
        self.stt.set_callbacks(
            on_partial_result=on_partial_result,
            on_final_result=on_final_result,
            on_utterance_start=on_utterance_start,
            on_utterance_end=on_utterance_end
        )
    
    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback function safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in audio manager callback: {e}")
    
    async def start(self) -> None:
        """Start audio manager."""
        if self.is_running:
            return
        
        try:
            # Start audio stream
            if self.audio_stream:
                self.audio_stream.start()
            
            # Start processing thread
            self._stop_event.clear()
            self._processing_thread = threading.Thread(target=self._processing_loop)
            self._processing_thread.start()
            
            self.is_running = True
            logger.info("Audio manager started")
            
        except Exception as e:
            logger.error(f"Failed to start audio manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop audio manager."""
        if not self.is_running:
            return
        
        try:
            # Stop processing thread
            self._stop_event.set()
            if self._processing_thread:
                self._processing_thread.join(timeout=1.0)
            
            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
            
            self.is_running = False
            logger.info("Audio manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio manager: {e}")
    
    def _processing_loop(self) -> None:
        """Main audio processing loop."""
        while not self._stop_event.is_set():
            try:
                # Get audio frame from queue
                if not self.audio_queue.empty():
                    audio_frame = self.audio_queue.get_nowait()
                    self.current_audio_frame = audio_frame
                    
                    # Process with VAD
                    if self.vad:
                        is_speech, confidence = asyncio.run(self.vad.process_audio(audio_frame))
                    
                    # Process with wake word detector
                    if self.wake_word_detector:
                        wake_word_result = asyncio.run(self.wake_word_detector.process_audio(audio_frame))
                    
                    # Process with STT (only if listening)
                    if self.stt and self.is_listening:
                        stt_result = asyncio.run(self.stt.process_audio(audio_frame))
                    
                    # Call audio data callback
                    if self.on_audio_data:
                        asyncio.create_task(self._call_callback(self.on_audio_data, audio_frame))
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
    
    def set_callbacks(self, on_audio_data: Optional[Callable] = None,
                     on_speech_detected: Optional[Callable] = None,
                     on_wake_word_detected: Optional[Callable] = None,
                     on_text_recognized: Optional[Callable] = None) -> None:
        """Set callback functions for audio events."""
        self.on_audio_data = on_audio_data
        self.on_speech_detected = on_speech_detected
        self.on_wake_word_detected = on_wake_word_detected
        self.on_text_recognized = on_text_recognized
    
    def get_state(self) -> Dict[str, Any]:
        """Get current audio manager state."""
        state = {
            "is_running": self.is_running,
            "is_listening": self.is_listening,
            "queue_size": self.audio_queue.qsize(),
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size
        }
        
        # Add component states
        if self.vad:
            state["vad"] = self.vad.get_state()
        if self.wake_word_detector:
            state["wake_word"] = self.wake_word_detector.get_state()
        if self.stt:
            state["stt"] = self.stt.get_state()
        
        return state
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio devices."""
        try:
            devices = sd.query_devices()
            return [
                {
                    "index": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"]
                }
                for i, device in enumerate(devices)
                if device["max_input_channels"] > 0
            ]
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []
    
    def set_audio_device(self, device_index: int) -> None:
        """Set audio input device."""
        try:
            # Stop current stream
            if self.audio_stream:
                self.audio_stream.stop()
            
            # Update config
            self.config.audio.device = device_index
            
            # Reinitialize stream
            asyncio.create_task(self._initialize_audio_stream())
            
            logger.info(f"Audio device set to index {device_index}")
            
        except Exception as e:
            logger.error(f"Error setting audio device: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup audio manager resources."""
        await self.stop()
        
        # Cleanup components
        if self.vad:
            self.vad.cleanup()
        if self.wake_word_detector:
            self.wake_word_detector.cleanup()
        if self.stt:
            self.stt.cleanup()
        
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.close()
        
        logger.info("Audio manager cleaned up")


# Example usage and testing
if __name__ == "__main__":
    async def test_audio_manager():
        """Test audio manager functionality."""
        # Create configuration
        audio_config = AudioConfig(
            sample_rate=16000,
            buffer_size=512,
            device=None  # Use default device
        )
        
        vad_config = VADConfig(threshold=0.5)
        wake_word_config = WakeWordConfig(
            wake_words=["hey assistant", "hey pi"],
            threshold=0.5
        )
        stt_config = STTConfig(
            model_path="models/vosk-model-small-en-us-0.22",
            confidence_threshold=0.5
        )
        
        config = AudioManagerConfig(
            audio=audio_config,
            vad=vad_config,
            wake_word=wake_word_config,
            stt=stt_config,
            mock_mode=True  # Use mock components for testing
        )
        
        # Create and initialize audio manager
        audio_manager = AudioManager(config)
        await audio_manager.initialize()
        
        # Set up callbacks
        async def on_audio_data(audio_frame):
            print(f"Audio data: {len(audio_frame.data)} samples")
        
        async def on_speech_detected(timestamp, confidence):
            print(f"Speech detected at {timestamp:.2f}s, confidence: {confidence:.3f}")
        
        async def on_wake_word_detected(activation_record):
            print(f"Wake word detected: {activation_record['wake_word']}")
        
        async def on_text_recognized(turn):
            print(f"Text recognized: '{turn.text}'")
        
        audio_manager.set_callbacks(
            on_audio_data=on_audio_data,
            on_speech_detected=on_speech_detected,
            on_wake_word_detected=on_wake_word_detected,
            on_text_recognized=on_text_recognized
        )
        
        # Start audio manager
        await audio_manager.start()
        
        # Run for a few seconds
        print("Audio manager running... Press Ctrl+C to stop")
        try:
            await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("Stopping audio manager...")
        
        # Stop and cleanup
        await audio_manager.stop()
        await audio_manager.cleanup()
    
    # Run test
    asyncio.run(test_audio_manager())
