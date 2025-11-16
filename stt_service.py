#!/usr/bin/env python3
"""
Speech-to-Text Service with Wake Word Detection.
Uses openwakeword to detect "Computer" wake word, then transcribes speech to text.
"""
import os
import sys
import logging
import numpy as np
import pyaudio
from openwakeword.model import Model
import speech_recognition as sr
from collections import deque

logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-Text service with wake word detection."""

    def __init__(self, wake_word="computer", chunk_size=1280, sample_rate=16000):
        """
        Initialize the STT service.

        Args:
            wake_word: Wake word to detect (default: "computer")
            chunk_size: Audio chunk size in frames (default: 1280)
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.wake_word = wake_word.lower()
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.is_running = False

        # Initialize wake word model
        logger.info("Initializing wake word detection model...")
        self.wake_word_model = Model(
            wakeword_models=[self.wake_word],
            inference_framework='onnx'
        )

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1

        logger.info(f"STT Service initialized with wake word: '{self.wake_word}'")

    def _detect_wake_word(self, audio_data):
        """
        Detect wake word in audio data.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            bool: True if wake word detected, False otherwise
        """
        # Get prediction from model
        prediction = self.wake_word_model.predict(audio_data)

        # Check if wake word was detected
        for model_name, score in prediction.items():
            if self.wake_word in model_name.lower() and score > 0.5:
                logger.info(f"Wake word detected! (confidence: {score:.2f})")
                return True

        return False

    def _transcribe_speech(self):
        """
        Transcribe speech after wake word is detected.

        Returns:
            str: Transcribed text or None if recognition failed
        """
        try:
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                logger.info("Listening for speech...")
                print("\n🎤 Listening... (speak now)")

                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for speech (with timeout)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

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
            # Open audio stream
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            logger.info("STT Service started. Listening for wake word...")
            print(f"\n👂 Listening for wake word: '{self.wake_word}'")
            print("   (Say 'Computer' to activate speech recognition)")
            print("   (Press Ctrl+C to stop)\n")

            # Buffer to accumulate audio for wake word detection
            audio_buffer = deque(maxlen=10)

            while self.is_running:
                # Read audio data
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Check for wake word
                if self._detect_wake_word(audio_array):
                    print(f"\n✅ Wake word '{self.wake_word}' detected!")

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

        except KeyboardInterrupt:
            logger.info("Stopping STT service (user interrupt)...")
            print("\n\n👋 Stopping STT service...")

        except Exception as e:
            logger.error(f"STT service error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")

        finally:
            # Clean up
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self.is_running = False
            logger.info("STT service stopped")

    def stop(self):
        """Stop the STT service."""
        self.is_running = False


def main():
    """Main entry point for testing the STT service."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start STT service
    stt_service = STTService(wake_word="computer")

    try:
        stt_service.start()
    except Exception as e:
        logger.error(f"Failed to start STT service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
