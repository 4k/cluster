#!/usr/bin/env python3
"""
Download a default Piper TTS voice model.
"""
import sys
import urllib.request
import json
from pathlib import Path


def download_voice(model_name="en_US-lessac-medium"):
    """
    Download a Piper voice model.

    Args:
        model_name: Name of the model to download (default: en_US-lessac-medium)
    """
    # Models directory
    models_dir = Path.home() / ".local" / "share" / "piper" / "voices"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model files
    onnx_url = f"https://github.com/rhasspy/piper/releases/download/v1.2.0/{model_name}.onnx"
    json_url = f"https://github.com/rhasspy/piper/releases/download/v1.2.0/{model_name}.onnx.json"

    onnx_path = models_dir / f"{model_name}.onnx"
    json_path = models_dir / f"{model_name}.onnx.json"

    # Check if already downloaded
    if onnx_path.exists() and json_path.exists():
        print(f"✅ Voice model already exists: {onnx_path}")
        return True

    print(f"📥 Downloading Piper voice model: {model_name}")
    print(f"   Destination: {models_dir}")

    try:
        # Download ONNX model
        print(f"\n⏳ Downloading {model_name}.onnx...")
        urllib.request.urlretrieve(onnx_url, onnx_path, reporthook=_progress_hook)
        print(f"\n✅ Downloaded: {onnx_path}")

        # Download JSON config
        print(f"\n⏳ Downloading {model_name}.onnx.json...")
        urllib.request.urlretrieve(json_url, json_path)
        print(f"✅ Downloaded: {json_path}")

        print(f"\n🎉 Voice model ready!")
        print(f"   Model: {model_name}")
        print(f"   Location: {models_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Error downloading voice model: {e}")
        print(f"\nYou can manually download from:")
        print(f"  {onnx_url}")
        print(f"  {json_url}")
        print(f"\nPlace them in: {models_dir}")
        return False


def _progress_hook(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  [{bar}] {percent:.1f}%", end='', flush=True)


def list_available_voices():
    """List some popular available voices."""
    voices = [
        ("en_US-lessac-medium", "English (US) - Female, Medium quality"),
        ("en_US-lessac-high", "English (US) - Female, High quality"),
        ("en_GB-alan-medium", "English (GB) - Male, Medium quality"),
        ("en_GB-northern_english_male-medium", "English (GB) - Male, Northern accent"),
        ("de_DE-thorsten-medium", "German - Male, Medium quality"),
        ("fr_FR-siwis-medium", "French - Female, Medium quality"),
        ("es_ES-sharvard-medium", "Spanish - Male, Medium quality"),
    ]

    print("\n📢 Popular Piper Voice Models:")
    print("=" * 70)
    for name, description in voices:
        print(f"  {name}")
        print(f"    {description}")
        print()
    print("Full list: https://github.com/rhasspy/piper/blob/master/VOICES.md")
    print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Download Piper TTS voice models')
    parser.add_argument(
        '--model',
        type=str,
        default='en_US-lessac-medium',
        help='Model name to download (default: en_US-lessac-medium)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available voice models'
    )

    args = parser.parse_args()

    if args.list:
        list_available_voices()
        return 0

    success = download_voice(args.model)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
