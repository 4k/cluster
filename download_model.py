#!/usr/bin/env python3
"""Script to download a specific model."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai.model_manager import get_model_manager

async def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "gemma3-270m-q6k"
    print(f"Downloading model: {model_id}")
    
    manager = get_model_manager()
    success = await manager.download_model(model_id)
    
    if success:
        print(f"✓ Model {model_id} downloaded successfully")
        return 0
    else:
        print(f"✗ Failed to download model {model_id}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))

