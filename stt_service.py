#!/usr/bin/env python3
"""
Speech-to-Text Service with Wake Word Detection.
Uses openwakeword to detect wake word, then transcribes speech using Vosk.
"""
import sys
import logging
import json
import numpy as np
import pyaudio
from vosk import Model as VoskModel, KaldiRecognizer
from openwakeword.model import Model

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

    def __init__(self, wake_word="computer", chunk_size=1280, sample_rate=16000, threshold=0.5, device_index=None, verbose=False, vosk_model_path=None):
        """
        Initialize the STT service.

        Args:
            wake_word: Wake word to detect (default: "computer")
            chunk_size: Audio chunk size in frames (default: 1280 for 80ms at 16kHz)
            sample_rate: Audio sample rate in Hz (default: 16000)
            threshold: Detection threshold 0.0-1.0 (default: 0.5)
            device_index: Audio input device index (default: None = system default)
            verbose: Show detection scores in real-time (default: False)
            vosk_model_path: Path to Vosk model directory (default: None = auto-detect)
        """
        self.wake_word = wake_word.lower()
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.device_index = device_index
        self.verbose = verbose
        self.vosk_model_path = vosk_model_path
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

        # Initialize Vosk speech recognition
        logger.info(f"Loading Vosk model from '{vosk_model_path}'...")
        try:
            if vosk_model_path:
                self.vosk_model = VoskModel(vosk_model_path)
            else:
                # Try to find model in common locations
                import os
                possible_paths = [
                    "models/vosk-model-small-en-us-0.15",
                    "vosk-model-small-en-us-0.15",
                    os.path.expanduser("~/.cache/vosk/vosk-model-small-en-us-0.15"),
                ]
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found Vosk model at: {path}")
                        self.vosk_model = VoskModel(path)
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(
                        "Vosk model not found. Please download a model from:\n"
                        "https://alphacephei.com/vosk/models\n"
                        "Example: vosk-model-small-en-us-0.15\n"
                        "Extract it and pass the path with --vosk-model argument"
                    )

            logger.info(f"Vosk model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}", exc_info=True)
            raise

        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1

        logger.info(f"STT Service initialized with wake word: '{self.wake_word}'")

    def _detect_wake_word(self, audio_data, show_scores=False):
        """
        Detect wake word in audio data.

        Args:
            audio_data: Audio data as numpy array (float32, range -1.0 to 1.0)
            show_scores: If True, display all model scores (for debugging)

        Returns:
            tuple: (detected, model_name, score, all_scores) - detected is True if wake word found
        """
        try:
            # Get prediction from model
            prediction = self.wake_word_model.predict(audio_data)

            # For debugging: show all scores
            if show_scores:
                return False, None, 0.0, prediction

            # Check all models for wake word detection
            max_score = 0.0
            max_model = None

            for model_name, score in prediction.items():
                # Track highest score for any model
                if score > max_score:
                    max_score = score
                    max_model = model_name

                # Check if this model matches our wake word and score is above threshold
                # Match partial names: "alexa" matches "alexa", "jarvis" matches "hey_jarvis", etc.
                model_simple = model_name.lower().replace("hey_", "").replace("_", " ")
                wake_word_simple = self.wake_word.lower().replace("_", " ")

                if wake_word_simple in model_simple or model_simple in wake_word_simple:
                    if score > self.threshold:
                        logger.info(f"Wake word '{self.wake_word}' detected in model '{model_name}' (confidence: {score:.3f})")
                        return True, model_name, score, prediction

            # Log if we got close but didn't trigger
            if max_score > self.threshold * 0.7:
                logger.debug(f"Close detection: {max_model} = {max_score:.3f} (threshold: {self.threshold})")

            return False, None, 0.0, prediction

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}", exc_info=True)
            return False, None, 0.0, {}

    def _transcribe_speech(self, audio_stream):
        """
        Transcribe speech after wake word is detected using Vosk.

        Args:
            audio_stream: PyAudio stream object to record from

        Returns:
            str: Transcribed text or None if recognition failed
        """
        try:
            logger.info("Listening for speech...")
            print("\n🎤 Listening... (speak now)")

            # Get audio parameters
            actual_rate = getattr(self, 'actual_sample_rate', 16000)
            chunk_size = getattr(self, 'actual_chunk_size', 1280)

            # Create Vosk recognizer
            recognizer = KaldiRecognizer(self.vosk_model, actual_rate)
            recognizer.SetWords(True)

            # Record for 5 seconds or until silence
            max_frames = int(actual_rate / chunk_size * 5)  # 5 seconds max
            silent_chunks = 0
            max_silent_chunks = int(actual_rate / chunk_size * 1.5)  # 1.5 seconds of silence
            final_text = ""

            for i in range(max_frames):
                data = audio_stream.read(chunk_size, exception_on_overflow=False)

                # Feed audio to Vosk
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        final_text += " " + result["text"]

                # Check for silence (voice activity detection)
                audio_array = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                if rms < 500:  # Silence threshold
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks and final_text:
                        break  # Stop if we've had enough silence and got some text
                else:
                    silent_chunks = 0

            # Get final result (partial result if any)
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get("text"):
                final_text += " " + final_result["text"]

            final_text = final_text.strip()

            if final_text:
                logger.info(f"Vosk transcription: {final_text}")
                return final_text
            else:
                logger.warning("Vosk returned empty transcription")
                print("⚠️  No speech detected")
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

            # Determine device capabilities
            device_channels = int(device_info['maxInputChannels'])
            device_sample_rate = int(device_info['defaultSampleRate'])

            logger.info(f"Device capabilities: {device_channels} channels, {device_sample_rate} Hz")

            # openWakeWord requires 16kHz audio
            # Try 16kHz first, fallback to device rate if that fails
            sample_rate_options = [16000]
            if device_sample_rate != 16000:
                sample_rate_options.append(device_sample_rate)

            # Try different configurations to find one that works
            # Some devices report stereo but only work in mono
            channel_options = [1, 2] if device_channels >= 2 else [1]
            # Use 1280 samples for 80ms at 16kHz (openWakeWord recommendation)
            buffer_sizes = [1280, 1024, 2048, 4096]

            stream = None
            last_error = None
            input_channels = None
            input_sample_rate = None
            adjusted_chunk_size = None

            # Try each combination
            for test_rate in sample_rate_options:
                for test_channels in channel_options:
                    for buffer_size in buffer_sizes:
                        try:
                            logger.info(f"Trying: {test_channels}ch, {test_rate}Hz, buffer={buffer_size}")
                            stream = audio.open(
                                format=self.format,
                                channels=test_channels,
                                rate=test_rate,
                                input=True,
                                input_device_index=self.device_index,
                                frames_per_buffer=buffer_size
                            )
                            logger.info(f"✓ Successfully opened stream")
                            input_channels = test_channels
                            input_sample_rate = test_rate
                            adjusted_chunk_size = buffer_size
                            break
                        except OSError as e:
                            last_error = e
                            logger.debug(f"  Failed: {e}")
                            continue

                    if stream is not None:
                        break

                if stream is not None:
                    break

            if stream is None:
                raise OSError(f"Could not open audio stream with any configuration.\n"
                            f"Last error: {last_error}\n"
                            f"Possible causes:\n"
                            f"  - Device is in use by another application (browser, Discord, etc.)\n"
                            f"  - Device requires exclusive access\n"
                            f"  - Driver issues\n"
                            f"Try closing other applications using the microphone.")

            # Store actual parameters for processing
            self.actual_sample_rate = input_sample_rate
            self.actual_chunk_size = adjusted_chunk_size

            # Store channels for processing
            self.input_channels = input_channels

            logger.info("STT Service started. Listening for wake word...")
            print(f"\n{'='*60}")
            print(f"👂 Listening for wake word: '{self.wake_word.upper()}'")
            print(f"   Audio Device: [{self.device_index}] {device_info['name']}")
            print(f"   Sample Rate: {input_sample_rate} Hz (device native)")
            print(f"   Channels: {input_channels} ({'Stereo' if input_channels == 2 else 'Mono'})")
            print(f"   Buffer Size: {adjusted_chunk_size} frames")
            print(f"   Detection Threshold: {self.threshold}")
            print(f"   Available models: {', '.join(list(self.wake_word_model.models.keys()))}")
            print(f"")
            print(f"   Say 'Hey {self.wake_word.title()}' to activate speech recognition")
            print(f"   Watch the audio level bar below to see microphone activity")
            print(f"   (Press Ctrl+C to stop)")
            print(f"{'='*60}\n")

            # For audio level display
            import sys
            import time
            last_update_time = 0

            while self.is_running:
                try:
                    # Read audio data (using adjusted chunk size)
                    audio_data = stream.read(self.actual_chunk_size, exception_on_overflow=False)

                    # Convert to numpy array (int16 - as expected by openWakeWord)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)

                    # Convert stereo to mono if needed (wake word model expects mono)
                    if self.input_channels == 2:
                        # Reshape to (samples, channels) and average across channels
                        # Keep as int16
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

                    # Resample to 16kHz if needed (wake word model expects 16kHz)
                    if self.actual_sample_rate != 16000:
                        # Use scipy for proper resampling
                        from scipy import signal
                        # Calculate number of samples after resampling
                        num_samples = int(len(audio_array) * 16000 / self.actual_sample_rate)
                        audio_array = signal.resample(audio_array, num_samples).astype(np.int16)

                    # Check for wake word (pass int16 data as per official example)
                    detected, model_name, score, all_scores = self._detect_wake_word(audio_array)

                    # Display audio level meter (update every 100ms)
                    current_time = time.time()
                    if current_time - last_update_time >= 0.1:
                        # Calculate RMS level (root mean square) for int16 data
                        # Convert to float for calculation
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        rms = np.sqrt(np.mean(audio_float ** 2))
                        # Scale to percentage (typical speech is 0.01-0.3 RMS)
                        level_percent = min(100, int(rms * 300))

                        if self.verbose and all_scores:
                            # Show detection scores for debugging
                            scores_str = " | ".join([f"{k.replace('hey_', '')}:{v:.2f}" for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]])
                            sys.stdout.write(f'\r🎤 Audio: {level_percent:3d}% | Scores: {scores_str:<50} | Threshold: {self.threshold:.2f}')
                        else:
                            # Create visual bar
                            bar_length = 40
                            filled = int(bar_length * level_percent / 100)
                            bar = '█' * filled + '░' * (bar_length - filled)

                            # Print level meter (overwrite same line)
                            sys.stdout.write(f'\r🎤 Audio Level: [{bar}] {level_percent:3d}%')

                        sys.stdout.flush()
                        last_update_time = current_time

                    if detected:
                        # Clear the audio level line and print wake word detection
                        sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear line
                        sys.stdout.flush()
                        print(f"✅ Wake word detected! (model: {model_name}, confidence: {score:.3f})")

                        # Transcribe speech (stream stays running for Vosk)
                        text = self._transcribe_speech(stream)

                        if text:
                            print(f"\n📝 Transcribed text:")
                            print(f"   \"{text}\"\n")
                            logger.info(f"Transcribed: {text}")

                        # Resume wake word detection
                        print(f"👂 Listening for wake word: '{self.wake_word.upper()}'...")
                        last_update_time = 0  # Reset to show level immediately

                except IOError as e:
                    # Handle audio buffer overflow
                    logger.warning(f"Audio buffer overflow: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Stopping STT service (user interrupt)...")
            # Clear audio level line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            print("\n👋 Stopping STT service...")

        except Exception as e:
            logger.error(f"STT service error: {e}", exc_info=True)
            # Clear audio level line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
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
    parser.add_argument(
        '--wake-word',
        type=str,
        default='jarvis',
        help='Wake word to detect: alexa, jarvis, mycroft, rhasspy (default: jarvis)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detection scores in real-time for debugging'
    )

    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0

    # Create and start STT service
    try:
        stt_service = STTService(
            wake_word=args.wake_word,
            threshold=args.threshold,
            device_index=args.device,
            verbose=args.verbose
        )
        stt_service.start()
    except Exception as e:
        logger.error(f"Failed to start STT service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
