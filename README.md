# AI Assistant

a prototype for a foundational AI operating system. Tier 0 Agent Platform.

## Features

- 🎤 **Wake Word Detection** - Activates on hearing "Computer" using OpenWakeWord
- 🗣️ **Speech-to-Text** - Transcribes your speech to text using Google Speech Recognition
- 🔊 **Text-to-Speech** - High-quality neural voice synthesis using Piper-TTS
- 🖥️ **Cross-Platform** - Works on Windows, Linux, and macOS
- 🌐 **Offline Capable** - TTS works completely offline after model download

## Requirements

- **Python 3.11 or 3.12** (recommended for best compatibility)
- Windows, Linux, or macOS
- Microphone and speakers

## Quick Start

### Windows (Recommended)

1. **Install Python 3.12**
   - Download from https://python.org/downloads/
   - Make sure to check "Add python.exe to PATH"

2. **Clone and setup:**
   ```powershell
   cd C:\Users\YourName\Projects
   git clone <repository-url>
   cd cluster

   # Create virtual environment with Python 3.12
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1

   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

   # Download Piper voice model
   python download_voice.py
   ```

### Linux / WSL

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pyaudio portaudio19-dev python3-dev

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download Piper voice model
python download_voice.py
```

## Usage

### TTS Service (Text-to-Speech)

```bash
# Basic usage - speak text through loudspeakers
python tts_service.py "Hello, how are you?"

# Save to WAV file instead of playing
python tts_service.py --output hello.wav "Hello, world!"

# Use custom voice model
python tts_service.py --model path/to/model.onnx "Custom voice"

# Show voice information
python tts_service.py --info
```

### STT Service (Speech-to-Text with Wake Word)

```bash
python stt_service.py
```

**How it works:**
1. Wait for "Listening for wake word: 'computer'"
2. Say **"Computer"** clearly
3. After detection, speak your text (you have 5 seconds to start, 10 seconds max)
4. Your transcribed text will appear in the console
5. The service returns to listening for the wake word

### Downloading Voice Models

```bash
# Download default English voice (en_US-lessac-medium)
python download_voice.py

# Download specific voice
python download_voice.py --model en_US-lessac-high

# List popular available voices
python download_voice.py --list
```

Voice models are saved to: `~/.local/share/piper/voices/`

## Project Structure

```
cluster/
├── main.py              # Main application entry point (future integration)
├── stt_service.py       # Speech-to-text with wake word detection
├── tts_service.py       # Text-to-speech using Piper-TTS
├── download_voice.py    # Voice model downloader
├── requirements.txt     # Python dependencies
├── .env                 # Configuration file
├── README.md            # This file
└── INSTALLATION.md      # Detailed installation guide
```

## Dependencies

- **piper-tts** (1.3.0) - Neural text-to-speech engine
- **openwakeword** - Wake word detection
- **SpeechRecognition** - Speech-to-text
- **PyAudio** - Audio input/output
- **numpy** - Array processing
- **scipy** - Scientific computing

## Troubleshooting

### Python Version Issues

**Problem:** Can't install piper-tts or other dependencies
**Solution:** Use Python 3.11 or 3.12 (not 3.13+)

### PyAudio Installation

**Windows:**
```powershell
pip install pyaudio
```

If that fails:
```powershell
pip install https://github.com/intxcc/pyaudio_portaudio/releases/download/v19.7.6/PyAudio-0.2.14-cp312-cp312-win_amd64.whl
```

**Linux:**
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio
```

### No Audio Input/Output

- Check microphone/speaker permissions in system settings
- Ensure default audio devices are properly configured
- Test with: `python -m speech_recognition` (should show microphone input)

### Wake Word Not Detected

- Speak clearly at normal volume
- Ensure microphone is working properly
- The default threshold is 0.5 (edit in code to adjust sensitivity)
- Check available models with the service (shown at startup)

### Voice Model Not Found

```bash
# Download a voice model
python download_voice.py

# Or manually place .onnx and .onnx.json files in:
# Windows: C:\Users\YourName\.local\share\piper\voices\
# Linux: ~/.local/share/piper/voices/
```

## Configuration

Edit `.env` to customize settings:
- Wake word detection parameters
- Audio device settings
- Model paths
- Logging levels

## Development

**Running tests:**
```bash
# Test TTS
python tts_service.py "This is a test"

# Test STT (interactive)
python stt_service.py
```

## License

MIT

## Credits

- **Piper-TTS**: https://github.com/rhasspy/piper
- **OpenWakeWord**: https://github.com/dscripka/openWakeWord
- **SpeechRecognition**: https://github.com/Uberi/speech_recognition
