#!/usr/bin/env python3
"""
Text-to-Speech Service using Piper-TTS.
Converts text to speech and plays it through loudspeakers.
"""
import sys
import logging
import argparse
import io
import wave
from pathlib import Path
from piper import PiperVoice
import pyaudio

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service using Piper-TTS for high-quality voice synthesis."""

    def __init__(self, model_path=None, speaker_id=0):
        """
        Initialize the TTS service.

        Args:
            model_path: Path to Piper voice model (.onnx file)
            speaker_id: Speaker ID for multi-speaker models (default: 0)
        """
        self.speaker_id = speaker_id
        self.model_path = model_path or self._get_default_model()

        # Initialize Piper voice
        logger.info(f"Loading Piper voice model: {self.model_path}")
        try:
            self.voice = PiperVoice.load(str(self.model_path))
            logger.info("TTS Service initialized successfully")
            logger.info(f"Voice: {self.voice.config.get('name', 'Unknown')}")
            logger.info(f"Language: {self.voice.config.get('language', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load Piper voice model: {e}", exc_info=True)
            raise

        # Initialize PyAudio for playback
        self.audio = pyaudio.PyAudio()

    def _get_default_model(self):
        """Get default model path or download if needed."""
        # Default model directory
        models_dir = Path.home() / ".local" / "share" / "piper" / "voices"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Look for any existing .onnx model
        existing_models = list(models_dir.glob("*.onnx"))
        if existing_models:
            logger.info(f"Using existing model: {existing_models[0]}")
            return existing_models[0]

        # If no model exists, inform user
        logger.warning("No Piper voice model found.")
        logger.info(f"Please download a model to: {models_dir}")
        logger.info("Download models from: https://github.com/rhasspy/piper/releases/tag/v1.2.0")
        logger.info("Example: en_US-lessac-medium.onnx")

        raise FileNotFoundError(
            f"No Piper voice model found in {models_dir}. "
            f"Please download a model from https://github.com/rhasspy/piper/releases"
        )

    def synthesize(self, text):
        """
        Synthesize text to audio data.

        Args:
            text: Text to convert to speech

        Returns:
            tuple: (audio_data, sample_rate) or (None, None) on error
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to synthesize")
            return None, None

        try:
            logger.info(f"Synthesizing: {text}")

            # Synthesize speech
            audio_stream = io.BytesIO()
            wav_file = wave.open(audio_stream, 'wb')

            # Configure WAV file
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.voice.config['sample_rate'])

            # Synthesize and write to WAV
            self.voice.synthesize(text, wav_file, speaker_id=self.speaker_id)
            wav_file.close()

            # Get audio data
            audio_stream.seek(0)
            wav_reader = wave.open(audio_stream, 'rb')
            audio_data = wav_reader.readframes(wav_reader.getnframes())
            sample_rate = wav_reader.getframerate()
            wav_reader.close()

            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Failed to synthesize text: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return None, None

    def play(self, audio_data, sample_rate):
        """
        Play audio data through loudspeakers.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )

            # Play audio
            stream.write(audio_data)

            # Clean up
            stream.stop_stream()
            stream.close()

            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}", exc_info=True)
            print(f"❌ Playback error: {e}")
            return False

    def speak(self, text):
        """
        Convert text to speech and play through loudspeakers.

        Args:
            text: Text to convert to speech

        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to speak")
            return False

        try:
            print(f"\n🔊 Speaking: \"{text}\"")

            # Synthesize audio
            audio_data, sample_rate = self.synthesize(text)

            if audio_data is None:
                return False

            # Play audio
            success = self.play(audio_data, sample_rate)

            if success:
                logger.info("Speech completed successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to speak text: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def get_voice_info(self):
        """Get information about the current voice."""
        info = {
            'name': self.voice.config.get('name', 'Unknown'),
            'language': self.voice.config.get('language', 'Unknown'),
            'quality': self.voice.config.get('quality', 'Unknown'),
            'sample_rate': self.voice.config.get('sample_rate', 'Unknown'),
            'num_speakers': self.voice.config.get('num_speakers', 1)
        }
        return info

    def stop(self):
        """Stop the TTS engine and clean up resources."""
        try:
            self.audio.terminate()
            logger.info("TTS engine stopped")
        except Exception as e:
            logger.error(f"Error stopping TTS engine: {e}")


def main():
    """Main entry point for the TTS service."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Piper Text-to-Speech Service')
    parser.add_argument(
        'text',
        nargs='*',
        help='Text to convert to speech'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to Piper voice model (.onnx file)'
    )
    parser.add_argument(
        '--speaker',
        type=int,
        default=0,
        help='Speaker ID for multi-speaker models (default: 0)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show voice information and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize TTS service
        tts = TTSService(model_path=args.model, speaker_id=args.speaker)

        # Show voice info if requested
        if args.info:
            info = tts.get_voice_info()
            print("\n📢 Voice Information:")
            for key, value in info.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            print()
            return 0

        # Get text from arguments
        if not args.text:
            print("❌ Error: No text provided")
            print("\nUsage examples:")
            print('  python tts_service.py "Hello, how are you?"')
            print('  python tts_service.py --model path/to/model.onnx "Custom voice"')
            print('  python tts_service.py --info')
            return 1

        text = ' '.join(args.text)

        # Speak the text
        success = tts.speak(text)

        # Clean up
        tts.stop()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"TTS service error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
