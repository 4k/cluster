# Installation Guide

## Quick Start

### Windows (Recommended for this app)

1. **Create virtual environment:**
   ```powershell
   python -m venv venv
   ```

2. **Activate virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If you get an execution policy error:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **If PyAudio fails, install it separately:**
   ```powershell
   pip install pipwin
   pipwin install pyaudio
   ```

   Then install the rest:
   ```powershell
   pip install -r requirements.txt
   ```

---

### Linux / WSL

1. **Install system dependencies first:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pyaudio portaudio19-dev python3-dev espeak
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running the Applications

### STT Service (Speech-to-Text with Wake Word)
```bash
python stt_service.py
```
Say "Computer" to activate, then speak your text.

### TTS Service (Text-to-Speech)
```bash
python tts_service.py "Hello, this is a test"
```

---

## Troubleshooting

### PyAudio Installation Issues

**Windows:**
- Use pipwin: `pip install pipwin && pipwin install pyaudio`
- Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**Linux:**
- Install portaudio: `sudo apt-get install portaudio19-dev`
- Install python3-pyaudio: `sudo apt-get install python3-pyaudio`

**macOS:**
- Install portaudio: `brew install portaudio`

### pyttsx3 Issues

**Linux:**
- Install espeak: `sudo apt-get install espeak`
- Or install festival: `sudo apt-get install festival`

**Windows:**
- No additional dependencies needed (uses SAPI5)

### openwakeword Issues

If wake word detection doesn't work:
- Ensure numpy and scipy are properly installed
- Check microphone permissions
- Try updating to latest version
