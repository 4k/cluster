#!/usr/bin/env python3
"""
Generate a sample audio file for mock TTS.
Creates a simple sine wave audio file that can be played by the mock TTS engine.
"""

import numpy as np
import wave
import os
from pathlib import Path

def generate_sample_audio():
    """Generate a sample audio file for mock TTS."""
    # Audio parameters
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some harmonics for more interesting sound
    audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.1
    audio += np.sin(2 * np.pi * frequency * 3 * t) * 0.05
    
    # Apply envelope (fade in/out)
    envelope = np.exp(-t * 1.5)  # Decay envelope
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Convert to 16-bit PCM
    audio_16bit = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    
    output_file = voices_dir / "mock_sample.wav"
    
    with wave.open(str(output_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())
    
    print(f"Generated sample audio file: {output_file}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz, Frequency: {frequency}Hz")
    
    return output_file

if __name__ == "__main__":
    generate_sample_audio()

