#!/usr/bin/env python3
"""
Speech-to-Text Service with Wake Word Detection.
Uses openwakeword to detect "Computer" wake word, then transcribes speech to text.
"""
import sys
import logging
import numpy as np
import pyaudio
from openwakeword.model import Model
import speech_recognition as sr

logger = logging.getLogger(__name__)


def list_audio_devices():
    """List all available audio input devices."""
    audio = pyaudio.PyAudio()
    print("\n📱 Available Audio Input Devices:")
    print("=" * 70)

    default_device = audio.get_default_input_device_info()
    default_index = default_device['index']

    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Input device
            is_default = " (DEFAULT)" if i == default_index else ""
            print(f"  [{i}] {device_info['name']}{is_default}")
            print(f"      Channels: {device_info['maxInputChannels']}, "
                  f"Sample Rate: {int(device_info['defaultSampleRate'])} Hz")

    print("=" * 70)
    audio.terminate()
    return default_index


class STTService:
    """Speech-to-Text service with wake word detection."""

    def __init__(self, wake_word="computer", chunk_size=1280, sample_rate=16000, threshold=0.5, device_index=None):
        """
        Initialize the STT service.

        Args:
            wake_word: Wake word to detect (default: "computer")
            chunk_size: Audio chunk size in frames (default: 1280 for 80ms at 16kHz)
            sample_rate: Audio sample rate in Hz (default: 16000)
            threshold: Detection threshold 0.0-1.0 (default: 0.5)
            device_index: Audio input device index (default: None = system default)
        """
        self.wake_word = wake_word.lower()
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.device_index = device_index
        self.is_running = False

        # Initialize wake word model
        logger.info("Initializing wake word detection model...")
        try:
            # Load pre-trained models (or specific model if available)
            self.wake_word_model = Model(inference_framework='onnx')
            logger.info(f"Loaded wake word models: {list(self.wake_word_model.models.keys())}")
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}", exc_info=True)
            raise

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        # Adjust for better recognition
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1

        logger.info(f"STT Service initialized with wake word: '{self.wake_word}'")

    def _detect_wake_word(self, audio_data):
        """
        Detect wake word in audio data.

        Args:
            audio_data: Audio data as numpy array (float32, range -1.0 to 1.0)

        Returns:
            tuple: (detected, model_name, score) - detected is True if wake word found
        """
        try:
            # Get prediction from model
            prediction = self.wake_word_model.predict(audio_data)

            # Check all models for wake word detection
            for model_name, score in prediction.items():
                # Check if this model matches our wake word and score is above threshold
                if self.wake_word in model_name.lower() and score > self.threshold:
                    logger.info(f"Wake word '{self.wake_word}' detected in model '{model_name}' (confidence: {score:.3f})")
                    return True, model_name, score

            return False, None, 0.0

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}", exc_info=True)
            return False, None, 0.0

    def _transcribe_speech(self):
        """
        Transcribe speech after wake word is detected.

        Returns:
            str: Transcribed text or None if recognition failed
        """
        try:
            with sr.Microphone(device_index=self.device_index, sample_rate=self.sample_rate) as source:
                logger.info("Listening for speech...")
                print("\n🎤 Listening... (speak now)")

                # Adjust for ambient noise (quick adjustment)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for speech (with timeout and phrase limit)
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )

                logger.info("Processing speech...")
                print("⏳ Processing...")

                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                return text

        except sr.WaitTimeoutError:
            logger.warning("No speech detected (timeout)")
            print("⚠️  No speech detected")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            print("⚠️  Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            print(f"❌ Recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return None

    def start(self):
        """Start the STT service with wake word detection."""
        self.is_running = True

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        try:
            # Get device info
            if self.device_index is not None:
                device_info = audio.get_device_info_by_index(self.device_index)
                logger.info(f"Using audio device [{self.device_index}]: {device_info['name']}")
            else:
                device_info = audio.get_default_input_device_info()
                self.device_index = device_info['index']
                logger.info(f"Using default audio device [{self.device_index}]: {device_info['name']}")

            # Open audio stream
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )

            logger.info("STT Service started. Listening for wake word...")
            print(f"\n{'='*60}")
            print(f"👂 Listening for wake word: '{self.wake_word}'")
            print(f"   Audio Device: [{self.device_index}] {device_info['name']}")
            print(f"   Available models: {list(self.wake_word_model.models.keys())}")
            print(f"   (Say 'Computer' to activate speech recognition)")
            print(f"   (Press Ctrl+C to stop)")
            print(f"{'='*60}\n")

            while self.is_running:
                try:
                    # Read audio data
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)

                    # Convert to numpy array (float32, normalized to -1.0 to 1.0)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Check for wake word
                    detected, model_name, score = self._detect_wake_word(audio_array)

                    if detected:
                        print(f"\n✅ Wake word detected! (model: {model_name}, confidence: {score:.3f})")

                        # Stop wake word detection temporarily
                        stream.stop_stream()

                        # Transcribe speech
                        text = self._transcribe_speech()

                        if text:
                            print(f"\n📝 Transcribed text:")
                            print(f"   \"{text}\"\n")
                            logger.info(f"Transcribed: {text}")

                        # Resume wake word detection
                        print(f"👂 Listening for wake word: '{self.wake_word}'...\n")
                        stream.start_stream()

                except IOError as e:
                    # Handle audio buffer overflow
                    logger.warning(f"Audio buffer overflow: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Stopping STT service (user interrupt)...")
            print("\n\n👋 Stopping STT service...")

        except Exception as e:
            logger.error(f"STT service error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")

        finally:
            # Clean up
            if 'stream' in locals():
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            audio.terminate()
            self.is_running = False
            logger.info("STT service stopped")

    def stop(self):
        """Stop the STT service."""
        self.is_running = False


def main():
    """Main entry point for testing the STT service."""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Speech-to-Text Service with Wake Word Detection')
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List all available audio input devices and exit'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Audio input device index (use --list-devices to see available devices)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Wake word detection threshold 0.0-1.0 (default: 0.5)'
    )

    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0

    # Create and start STT service
    try:
        stt_service = STTService(
            wake_word="computer",
            threshold=args.threshold,
            device_index=args.device
        )
        stt_service.start()
    except Exception as e:
        logger.error(f"Failed to start STT service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
