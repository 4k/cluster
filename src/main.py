"""
Cluster: The Bridge
"""

import logging
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Add src directory to Python path FIRST before other imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager
from core.event_bus import EventBus, EventType
from core.types import EmotionType, GazeDirection
from audio.audio_manager import AudioManager, AudioManagerConfig
from audio.ambient_stt import AmbientSTTConfig
from voice.simple_tts_manager import SimpleTTSManager
from display.simple_display_manager import DisplayManager, DisplayConfig
from ai.llm_manager import LLMManager
from animation import get_animation_engine

logger = logging.getLogger('cluster.main')

class Cluster:
    """Main Cluster Application."""
    
    def __init__(self):
        self.config = None
        self.is_running = False
        
        # Core components
        self.event_bus: Optional[EventBus] = None
        self.audio_manager: Optional[AudioManager] = None
        self.tts_manager: Optional[SimpleTTSManager] = None
        self.display_manager: Optional[DisplayManager] = None
        self.llm_manager: Optional[LLMManager] = None
        self.conversation_manager = None
        self.animation_engine = None
        
        # State
        self.current_speaker: Optional[str] = None
        self.is_processing = False

        # Model download state - simplified
        self.is_downloading_model = False
    
    async def _setup_configured_model(self) -> None:
        """Check if the configured model is available.
        
        Simplified version - just checks availability, no automatic downloading.
        """
        try:
            # Get configured model ID from config
            model_id = self.config.llm.get('model_id')

            if not model_id:
                logger.warning("No model_id configured. LLM will use fallback/mock provider.")
                return

            # Get model availability info
            availability = self.llm_manager.get_model_availability(model_id)

            if availability["model_info"] is None:
                logger.warning(f"Model '{model_id}' not found in models.json. LLM will use fallback/mock provider.")
                return

            if availability["available"]:
                logger.info(f"✓ Configured model '{availability['model_name']}' is available")
            else:
                logger.info(f"Configured model '{availability['model_name']}' not downloaded")
                logger.info("LLM will use mock/fallback provider. Download model manually if needed.")

        except Exception as e:
            logger.error(f"Error checking configured model: {e}")
            import traceback
            logger.error(traceback.format_exc())


    async def initialize(self) -> None:
        """Initialize the voice assistant."""
        try:
            # Load configuration
            config_manager = ConfigManager()
            self.config = config_manager.load_config()

            logger.info(f"Configuration loaded: {self.config.name} v{self.config.version}")
            
            # Setup logging
            self._setup_logging()
            
            # Initialize event bus
            self.event_bus = await EventBus.get_instance()
            await self.event_bus.start()

            # Initialize components
            await self._initialize_components()

            # Check and setup configured model (may trigger background download)
            await self._setup_configured_model()

            # Setup event handlers (includes model download handlers)
            self._setup_event_handlers()
            
            logger.info("Voice assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice assistant: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create logs directory
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.logs_dir / 'assistant.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"Logging configured: {self.config.log_level}")
    
    async def _initialize_components(self) -> None:
        """Initialize all components."""
        try:
            # Initialize audio manager
            # Create AmbientSTTConfig from config dict
            ambient_stt_config = AmbientSTTConfig(
                enabled=self.config.ambient_stt.get('enabled', True),
                model_path=self.config.ambient_stt.get('model_path', 'models/vosk-model-small-en-us-0.15'),
                language=self.config.ambient_stt.get('language', 'en'),
                sample_rate=self.config.audio.sample_rate,
                confidence_threshold=self.config.ambient_stt.get('confidence_threshold', 0.3),
                wake_word_timeout=self.config.ambient_stt.get('wake_word_timeout', 5.0),
                frame_skip=self.config.ambient_stt.get('frame_skip', 1),
                min_confidence=self.config.ambient_stt.get('min_confidence', 0.3)
            )

            audio_config = AudioManagerConfig(
                audio=self.config.audio,
                vad=self.config.vad,
                wake_word=self.config.wake_word,
                stt=self.config.stt,
                ambient_stt=ambient_stt_config,
                mock_mode=self.config.development.get('mock_audio', False) or self.config.mock_stt,
                mock_ambient_stt=self.config.mock_ambient_stt,
                enable_ambient_stt=ambient_stt_config.enabled
            )
            self.audio_manager = AudioManager(audio_config)
            await self.audio_manager.initialize()
            
            # Initialize TTS manager
            self.tts_manager = SimpleTTSManager(self.config.tts)
            await self.tts_manager.initialize()
            
            # Initialize display manager
            display_config = DisplayConfig(
                mode=self.config.display.get('mode', 'dual_display'),
                resolution=tuple(self.config.display.get('resolution', [800, 480])),
                fps=self.config.display.get('fps', 30),
                development_mode=self.config.display.get('development_mode', True),
                window_positions=self.config.display.get('window_positions'),
                fullscreen=self.config.display.get('fullscreen', False),
                borderless=self.config.display.get('borderless', False),
                always_on_top=self.config.display.get('always_on_top', False),
                led_count=self.config.display.get('led_count', 60),
                led_pin=self.config.display.get('led_pin', 18),
                touch_enabled=self.config.display.get('touch_enabled', False),
                calibration_file=self.config.display.get('calibration_file')
            )
            self.display_manager = DisplayManager(display_config)
            await self.display_manager.initialize()
            
            # Initialize LLM manager
            logger.info(f"Initializing LLM manager with provider_type: {self.config.llm.get('provider_type', 'NOT SET')}")
            self.llm_manager = LLMManager(self.config.llm)
            await self.llm_manager.initialize()
            
            
            
            # Initialize animation engine (after display manager)
            self.animation_engine = await get_animation_engine(self.display_manager)
            
            logger.info("All components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for component communication."""
        # Audio events
        self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word_detected)
        self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
        self.event_bus.subscribe(EventType.SPEECH_ENDED, self._on_speech_ended)

        # Ambient speech events
        self.event_bus.subscribe(EventType.AMBIENT_SPEECH_DETECTED, self._on_ambient_speech_detected)
        self.event_bus.subscribe(EventType.WAKEWORD_SPEECH_DETECTED, self._on_wakeword_speech_detected)

        # TTS events
        self.event_bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
        self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)

        # LLM response events
        self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)

        # Display events
        self.event_bus.subscribe(EventType.EXPRESSION_CHANGE, self._on_expression_change)

        # Model management events - simplified
        self.event_bus.subscribe(EventType.PROVIDER_SWITCHED, self._on_provider_switched)

        # System events
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error_occurred)
    
    async def _on_wake_word_detected(self, event) -> None:
        """Handle wake word detection."""
        logger.info(f"Wake word detected: {event.data.get('wake_word')}")
        
        # Tell animation engine to update display
        if self.animation_engine:
            await self.animation_engine.on_wake_word_detected()
    
    async def _on_speech_detected(self, event) -> None:
        """Handle speech detection."""
        text = event.data.get('text', '')
        confidence = event.data.get('confidence', 0.0)
        
        # Tell animation engine to show processing state
        if self.animation_engine:
            await self.animation_engine.on_speech_detected()
        
        # Process the speech (LLM generation)
        await self._process_speech(text, confidence)
    
    async def _on_speech_ended(self, event) -> None:
        """Handle speech end."""
        # No display update - animation engine handles state transitions
        pass

    async def _on_ambient_speech_detected(self, event) -> None:
        """Handle ambient speech detection (background listening)."""
        text = event.data.get('text', '')
        confidence = event.data.get('confidence', 0.0)
        mode = event.data.get('mode', 'ambient')

        logger.info(f"[AMBIENT] Speech detected: '{text}' (confidence: {confidence:.3f})")

        # You can choose to process ambient speech or just log it
        # For now, we'll just log it without triggering animations
        # Uncomment below to process ambient speech through LLM:
        # await self._process_speech(text, confidence)

    async def _on_wakeword_speech_detected(self, event) -> None:
        """Handle wake word triggered speech detection."""
        text = event.data.get('text', '')
        confidence = event.data.get('confidence', 0.0)
        mode = event.data.get('mode', 'wakeword')

        logger.info(f"[WAKEWORD] Speech detected: '{text}' (confidence: {confidence:.3f})")

        # Tell animation engine to show processing state
        if self.animation_engine:
            await self.animation_engine.on_speech_detected()

        # Process the speech (LLM generation) - wake word triggered speech is active
        await self._process_speech(text, confidence)

    async def _on_tts_started(self, event) -> None:
        """Handle TTS start."""
        # Tell animation engine to start speaking animations
        if self.animation_engine:
            await self.animation_engine.on_tts_started()
    
    async def _on_tts_completed(self, event) -> None:
        """Handle TTS completion."""
        # Tell animation engine to return to idle
        if self.animation_engine:
            await self.animation_engine.on_tts_completed()
    
    async def _on_response_generated(self, event) -> None:
        """Handle LLM response generation."""
        pass  # LLM module already logs response generation
    
    async def _on_expression_change(self, event) -> None:
        """Handle expression change."""
        # This is handled by the display manager itself
        pass
    
    async def _on_error_occurred(self, event) -> None:
        """Handle error events."""
        error = event.data.get('error', 'Unknown error')
        component = event.data.get('component', 'Unknown')
        logger.error(f"Error in {component}: {error}")

        # Tell animation engine to show error state
        if self.animation_engine:
            await self.animation_engine.on_error_occurred()


    async def _on_provider_switched(self, event) -> None:
        """Handle provider switch."""
        provider_type = event.data.get('provider_type', 'Unknown')
        model_name = event.data.get('model_name', 'Unknown')
        from_provider = event.data.get('from_provider', 'Unknown')
        logger.info(f"Provider switched from {from_provider} to {provider_type} (model: {model_name})")
    
    async def _process_speech(self, text: str, confidence: float) -> None:
        """Process detected speech and generate response."""
        try:
            self.is_processing = True
            
            # Create a simple conversation context for the LLM
            from core.types import ConversationContext, ConversationTurn
            import time
            conversation = ConversationContext(
                turns=[],
                current_emotion=EmotionType.NEUTRAL,
                last_activity=time.time()
            )
            
            # Generate response using LLM
            response = await self.llm_manager.generate_response(text, conversation)
            
            if response:
                # Notify animation engine that processing succeeded
                if self.animation_engine:
                    await self.animation_engine.on_speech_processed(True)
                
                # TODO: Generate TTS audio and play it
                
            else:
                logger.warning("LLM did not generate a response")
                # Notify animation engine that processing failed
                if self.animation_engine:
                    await self.animation_engine.on_speech_processed(False)
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Notify animation engine of error
            if self.animation_engine:
                await self.animation_engine.on_error_occurred()
        finally:
            self.is_processing = False
    
    async def _play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio output."""
        # This would be implemented with actual audio output
        # For now, just log the audio info
        logger.info(f"Playing audio: {len(audio)} samples at {sample_rate} Hz")
    
    async def _update_lip_sync(self, visemes) -> None:
        """Update display with lip-sync animation."""
        # TODO: Implement lip-sync through animation engine
        # The animation engine should handle mouth shape updates during speech
        # For now, lip-sync is handled automatically by animation state (speaking)
        pass
    
    async def start(self) -> None:
        """Start the voice assistant."""
        if self.is_running:
            return
        
        try:
            # Start components
            await self.audio_manager.start()
            await self.display_manager.start()
            
            self.is_running = True
            logger.info("Voice assistant started")
            
        except Exception as e:
            logger.error(f"Failed to start voice assistant: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the voice assistant."""
        if not self.is_running:
            return
        
        try:
            # Stop components
            if self.audio_manager:
                await self.audio_manager.stop()
            if self.display_manager:
                await self.display_manager.stop()
            
            self.is_running = False
            logger.info("Voice assistant stopped")
            
        except Exception as e:
            logger.error(f"Error stopping voice assistant: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup voice assistant resources."""
        try:
            # Stop if running
            await self.stop()


            # Cleanup components
            if self.audio_manager:
                await self.audio_manager.cleanup()
            if self.tts_manager:
                await self.tts_manager.cleanup()
            if self.display_manager:
                await self.display_manager.cleanup()
            if self.llm_manager:
                await self.llm_manager.cleanup()

            # Stop event bus
            if self.event_bus:
                await self.event_bus.stop()

            logger.info("Voice assistant cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> dict:
        """Get current status of the voice assistant."""
        return {
            "is_running": self.is_running,
            "is_processing": self.is_processing,
            "current_speaker": self.current_speaker,
            "is_downloading_model": self.is_downloading_model,
            "components": {
                "audio_manager": self.audio_manager.get_state() if self.audio_manager else None,
                "tts_manager": self.tts_manager.get_state() if self.tts_manager else None,
                "display_manager": self.display_manager.get_state() if self.display_manager else None,
                "llm_manager": self.llm_manager.get_state() if self.llm_manager else None
            }
        }


async def main():
    """Main application entry point."""
    assistant = Cluster()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(assistant.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start assistant
        await assistant.initialize()
        await assistant.start()
        
        # Keep running
        logger.info("Voice assistant is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await assistant.cleanup()


if __name__ == "__main__":
    # Let's gooooo
    asyncio.run(main())
