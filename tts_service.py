#!/usr/bin/env python3
"""
Text-to-Speech Service.
Converts text to speech and plays it through loudspeakers.
"""
import sys
import logging
import argparse
import pyttsx3

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service for converting text to audio output."""

    def __init__(self, rate=150, volume=1.0, voice_index=0):
        """
        Initialize the TTS service.

        Args:
            rate: Speech rate (words per minute, default: 150)
            volume: Volume level 0.0 to 1.0 (default: 1.0)
            voice_index: Voice index to use (default: 0)
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index

        # Initialize TTS engine
        logger.info("Initializing TTS engine...")
        try:
            self.engine = pyttsx3.init()
            self._configure_engine()
            logger.info("TTS Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}", exc_info=True)
            raise

    def _configure_engine(self):
        """Configure the TTS engine with settings."""
        # Set speech rate
        self.engine.setProperty('rate', self.rate)

        # Set volume
        self.engine.setProperty('volume', self.volume)

        # Set voice
        voices = self.engine.getProperty('voices')
        if voices and len(voices) > self.voice_index:
            self.engine.setProperty('voice', voices[self.voice_index].id)
            logger.info(f"Using voice: {voices[self.voice_index].name}")
        else:
            logger.warning(f"Voice index {self.voice_index} not available, using default")

    def list_voices(self):
        """List all available voices."""
        voices = self.engine.getProperty('voices')
        print("\n📢 Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
            print(f"      ID: {voice.id}")
            print(f"      Languages: {voice.languages}")
            print()

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
            logger.info(f"Speaking: {text}")
            print(f"\n🔊 Speaking: \"{text}\"")

            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()

            logger.info("Speech completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to speak text: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False

    def set_rate(self, rate):
        """
        Set speech rate.

        Args:
            rate: Speech rate in words per minute
        """
        self.rate = rate
        self.engine.setProperty('rate', rate)
        logger.info(f"Speech rate set to {rate} WPM")

    def set_volume(self, volume):
        """
        Set volume level.

        Args:
            volume: Volume level from 0.0 to 1.0
        """
        self.volume = max(0.0, min(1.0, volume))
        self.engine.setProperty('volume', self.volume)
        logger.info(f"Volume set to {self.volume}")

    def set_voice(self, voice_index):
        """
        Set voice by index.

        Args:
            voice_index: Index of the voice to use
        """
        voices = self.engine.getProperty('voices')
        if voices and 0 <= voice_index < len(voices):
            self.voice_index = voice_index
            self.engine.setProperty('voice', voices[voice_index].id)
            logger.info(f"Voice changed to: {voices[voice_index].name}")
        else:
            logger.warning(f"Invalid voice index: {voice_index}")

    def stop(self):
        """Stop the TTS engine."""
        try:
            self.engine.stop()
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
    parser = argparse.ArgumentParser(description='Text-to-Speech Service')
    parser.add_argument(
        'text',
        nargs='*',
        help='Text to convert to speech'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=150,
        help='Speech rate in words per minute (default: 150)'
    )
    parser.add_argument(
        '--volume',
        type=float,
        default=1.0,
        help='Volume level 0.0 to 1.0 (default: 1.0)'
    )
    parser.add_argument(
        '--voice',
        type=int,
        default=0,
        help='Voice index to use (default: 0)'
    )
    parser.add_argument(
        '--list-voices',
        action='store_true',
        help='List all available voices and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize TTS service
        tts = TTSService(rate=args.rate, volume=args.volume, voice_index=args.voice)

        # List voices if requested
        if args.list_voices:
            tts.list_voices()
            return 0

        # Get text from arguments
        if not args.text:
            print("❌ Error: No text provided")
            print("\nUsage examples:")
            print('  python tts_service.py "Hello, how are you?"')
            print('  python tts_service.py --rate 200 "I speak faster now"')
            print('  python tts_service.py --list-voices')
            return 1

        text = ' '.join(args.text)

        # Speak the text
        success = tts.speak(text)

        return 0 if success else 1

    except Exception as e:
        logger.error(f"TTS service error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
