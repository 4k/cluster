# AI Assistant

A voice-activated AI assistant with speech-to-text (STT) and text-to-speech (TTS) capabilities.

## Features

- 🎤 **Wake Word Detection** - Activates on hearing "Computer"
- 🗣️ **Speech-to-Text** - Transcribes your speech to text
- 🔊 **Text-to-Speech** - Converts text to spoken audio
- 🖥️ **Cross-Platform** - Works on Windows, Linux, and macOS

## Quick Start

### Windows (Recommended)

Run the automated installation script:

```powershell
.\install_windows.ps1
```

Or install manually:

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```

### Linux / WSL

Run the automated installation script:

```bash
chmod +x install_linux.sh
./install_linux.sh
```

Or install manually - see [INSTALLATION.md](INSTALLATION.md)

## Usage

### STT Service (Speech-to-Text)

```bash
python stt_service.py
```

1. Wait for the prompt: "Listening for wake word: 'computer'"
2. Say **"Computer"**
3. Speak your text after hearing the activation sound
4. Your transcribed text will appear in the console

### TTS Service (Text-to-Speech)

```bash
# Basic usage
python tts_service.py "Hello, how are you?"

# List available voices
python tts_service.py --list-voices

# Custom settings
python tts_service.py --rate 180 --volume 0.8 "Speaking with custom settings"
```

## Files

- `main.py` - Main application entry point
- `stt_service.py` - Speech-to-text service with wake word detection
- `tts_service.py` - Text-to-speech service
- `.env` - Configuration file
- `requirements.txt` - Python dependencies

## Dependencies

- **openwakeword** - Wake word detection
- **SpeechRecognition** - Speech-to-text
- **pyttsx3** - Text-to-speech (offline)
- **PyAudio** - Audio input/output

## Troubleshooting

See [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting steps.

### Common Issues

**PyAudio won't install:**
- Windows: Use `pipwin install pyaudio`
- Linux: Install `portaudio19-dev` first
- See INSTALLATION.md for details

**No audio input/output:**
- Check microphone/speaker permissions
- Ensure default audio devices are set correctly

**Wake word not detected:**
- Speak clearly and at normal volume
- Ensure microphone is working
- Try adjusting microphone sensitivity in system settings

## Configuration

Edit `.env` to customize:
- STT provider and settings
- TTS voice and speed
- Wake word (default: "computer")
- Audio device settings

## License

MIT
