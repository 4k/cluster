#!/usr/bin/env python3
"""
Text-to-Speech Service using Piper-TTS 1.3.0.
Converts text to speech and plays it through loudspeakers.
"""
import sys
import logging
import argparse
import wave
import io
import tempfile
from pathlib import Path
import numpy as np
from piper.voice import PiperVoice
import pyaudio

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service using Piper-TTS for high-quality voice synthesis."""

    def __init__(self, model_path=None):
        """
        Initialize the TTS service.

        Args:
            model_path: Path to Piper voice model (.onnx file)
        """
        self.model_path = model_path or self._get_default_model()

        # Initialize Piper voice
        logger.info(f"Loading Piper voice model: {self.model_path}")
        try:
            self.voice = PiperVoice.load(str(self.model_path))
            logger.info("TTS Service initialized successfully")
            logger.info(f"Sample rate: {self.voice.config.sample_rate} Hz")
        except Exception as e:
            logger.error(f"Failed to load Piper voice model: {e}", exc_info=True)
            raise

        # Initialize PyAudio for playback
        self.audio = pyaudio.PyAudio()

    def _get_default_model(self):
        """Get default model path or provide download instructions."""
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
        logger.info("Download models from: https://huggingface.co/rhasspy/piper-voices")
        logger.info("Or run: python download_voice.py")

        raise FileNotFoundError(
            f"No Piper voice model found in {models_dir}. "
            f"Please run 'python download_voice.py' to download a voice model."
        )

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
            logger.info(f"Synthesizing: {text}")

            # Try streaming method first (if available)
            if hasattr(self.voice, 'synthesize_stream_raw'):
                return self._speak_streaming(text)
            else:
                # Fallback to WAV file method
                logger.info("Using WAV file method (synthesize_stream_raw not available)")
                return self._speak_via_wav(text)

        except Exception as e:
            logger.error(f"Failed to speak text: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def _speak_streaming(self, text):
        """Stream audio directly (if synthesize_stream_raw is available)."""
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.voice.config.sample_rate,
                output=True
            )

            # Synthesize and stream audio
            for audio_bytes in self.voice.synthesize_stream_raw(text):
                # Convert bytes to numpy array and play
                int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(int_data.tobytes())

            # Clean up
            stream.stop_stream()
            stream.close()

            logger.info("Speech completed successfully (streaming)")
            return True

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    def _speak_via_wav(self, text):
        """Generate WAV file and play it (fallback method)."""
        temp_file = None
        try:
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Synthesize speech to WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                self.voice.synthesize(text, wav_file)

            # Read and play the WAV file
            with wave.open(temp_path, 'rb') as wav_reader:
                # Open PyAudio stream
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wav_reader.getsampwidth()),
                    channels=wav_reader.getnchannels(),
                    rate=wav_reader.getframerate(),
                    output=True
                )

                # Play audio
                chunk_size = 1024
                audio_data = wav_reader.readframes(chunk_size)
                while audio_data:
                    stream.write(audio_data)
                    audio_data = wav_reader.readframes(chunk_size)

                # Clean up stream
                stream.stop_stream()
                stream.close()

            logger.info("Speech completed successfully (WAV file)")
            return True

        except Exception as e:
            logger.error(f"WAV playback failed: {e}")
            raise

        finally:
            # Clean up temporary file
            if temp_file and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except:
                    pass

    def speak_to_file(self, text, output_path):
        """
        Convert text to speech and save to WAV file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the WAV file

        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, nothing to synthesize")
            return False

        try:
            logger.info(f"Synthesizing to file: {output_path}")

            with wave.open(str(output_path), 'wb') as wav_file:
                self.voice.synthesize(text, wav_file)

            logger.info(f"Audio saved to: {output_path}")
            print(f"✅ Audio saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def get_voice_info(self):
        """Get information about the current voice."""
        info = {
            'sample_rate': self.voice.config.sample_rate,
            'num_speakers': self.voice.config.num_speakers if hasattr(self.voice.config, 'num_speakers') else 1,
            'has_streaming': hasattr(self.voice, 'synthesize_stream_raw'),
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
        '--output',
        type=str,
        help='Save audio to file instead of playing'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show voice information and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize TTS service
        tts = TTSService(model_path=args.model)

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
            print('  python tts_service.py --output output.wav "Save to file"')
            print('  python tts_service.py --info')
            return 1

        text = ' '.join(args.text)

        # Save to file or play audio
        if args.output:
            success = tts.speak_to_file(text, args.output)
        else:
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
