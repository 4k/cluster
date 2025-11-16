#!/usr/bin/env python3
"""
Setup script for downloading and configuring the default Gemma-3n model.
This script handles the initial model download and configuration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.model_manager import get_model_manager, download_gemma_3n, is_default_model_available
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_default_model():
    """Set up the default Gemma-3n model."""
    logger.info("Setting up default Gemma-3n model...")
    
    # Initialize model manager
    manager = get_model_manager()
    
    # Check if default model is already available
    if is_default_model_available():
        logger.info("✓ Default model is already available")
        return True
    
    # Get default model info
    default_model = manager.get_default_model()
    if not default_model:
        logger.error("✗ No default model configured")
        return False
    
    logger.info(f"Default model: {default_model.name}")
    logger.info(f"Repository: {default_model.repo_id}")
    logger.info(f"File patterns: {default_model.file_patterns}")
    
    # Download the model
    logger.info("Starting model download...")
    success = await download_gemma_3n()
    
    if success:
        logger.info("✓ Default model downloaded successfully")
        
        # Print model stats
        stats = manager.get_model_stats()
        logger.info(f"Model statistics:")
        logger.info(f"  Total models: {stats['total_models']}")
        logger.info(f"  Downloaded models: {stats['downloaded_models']}")
        logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
        logger.info(f"  Default model: {stats['default_model']}")
        
        return True
    else:
        logger.error("✗ Failed to download default model")
        return False


def print_model_info():
    """Print information about available models."""
    manager = get_model_manager()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    available_models = manager.get_available_models()
    downloaded_models = manager.get_downloaded_models()
    
    print(f"\nTotal models configured: {len(available_models)}")
    print(f"Downloaded models: {len(downloaded_models)}")
    
    print("\nPre-defined models:")
    for model in available_models:
        status = "✓ Downloaded" if manager.is_model_downloaded(model.id) else "✗ Not downloaded"
        default_marker = " (DEFAULT)" if model.is_default else ""
        print(f"  {model.name}{default_marker}")
        print(f"    ID: {model.id}")
        print(f"    Status: {status}")
        print(f"    Repository: {model.repo_id}")
        print(f"    Quantization: {model.quantization}")
        print(f"    Parameters: {model.parameters}")
        print(f"    Context Window: {model.context_window}")
        print(f"    Description: {model.description}")
        print()
    
    # Print model stats
    stats = manager.get_model_stats()
    print("Model Statistics:")
    print(f"  Total size: {stats['total_size_gb']:.2f} GB")
    print(f"  Default model: {stats['default_model']}")


async def main():
    """Main setup function."""
    print("Gemma-3n Model Setup")
    print("="*40)
    
    try:
        # Set up default model
        success = await setup_default_model()
        
        if success:
            print("\n✓ Setup completed successfully!")
        else:
            print("\n✗ Setup failed!")
            return 1
        
        # Print model information
        print_model_info()
        
        print("\n" + "="*60)
        print("SETUP COMPLETE")
        print("="*60)
        print("The Gemma-3n model is now configured as the default local model.")
        print("You can use it with the voice assistant application.")
        print("\nTo use the model:")
        print("1. Run the main application: python src/main.py")
        print("2. The model will be automatically loaded on startup")
        print("3. You can switch models using the model manager API")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
