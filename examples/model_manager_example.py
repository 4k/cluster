#!/usr/bin/env python3
"""
Example script demonstrating the model manager functionality.
Shows how to download, manage, and use local LLM models.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.model_manager import get_model_manager
from ai.providers.factory import LLMProviderFactory
from ai.providers.base import LLMConfig, ProviderType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def download_progress_callback(progress):
    """Callback for download progress updates."""
    print(f"Download progress for {progress.model_id}: {progress.percentage:.1f}% "
          f"({progress.status})")


async def demonstrate_model_manager():
    """Demonstrate model manager functionality."""
    print("Model Manager Demonstration")
    print("="*50)
    
    # Initialize model manager
    manager = get_model_manager()
    
    # Print available models
    print("\n1. Available Models:")
    models = manager.get_available_models()
    for model in models:
        status = "✓ Downloaded" if manager.is_model_downloaded(model.id) else "✗ Not downloaded"
        default_marker = " (DEFAULT)" if model.is_default else ""
        print(f"   {model.name}{default_marker} - {status}")
    
    # Get default model
    default_model = manager.get_default_model()
    if default_model:
        print(f"\n2. Default Model: {default_model.name}")
        
        # Check if downloaded
        if not manager.is_model_downloaded(default_model.id):
            print(f"   Downloading {default_model.name}...")
            success = await manager.download_model(
                default_model.id, 
                progress_callback=download_progress_callback
            )
            if success:
                print(f"   ✓ {default_model.name} downloaded successfully")
            else:
                print(f"   ✗ Failed to download {default_model.name}")
                return False
        else:
            print(f"   ✓ {default_model.name} is already downloaded")
    
    # Print model statistics
    stats = manager.get_model_stats()
    print(f"\n3. Model Statistics:")
    print(f"   Total models: {stats['total_models']}")
    print(f"   Downloaded models: {stats['downloaded_models']}")
    print(f"   Total size: {stats['total_size_gb']:.2f} GB")
    
    return True


async def demonstrate_llm_provider():
    """Demonstrate LLM provider with model manager."""
    print("\n" + "="*50)
    print("LLM Provider Demonstration")
    print("="*50)
    
    # Create LLM configuration using model manager
    config = LLMConfig(
        provider_type=ProviderType.LOCAL,
        model="gemma-3n-q4-xl",
        model_id="gemma-3n-q4-xl",  # This will use the model manager
        temperature=0.7,
        max_tokens=100,
        context_window=2048,
        n_gpu_layers=0,
        n_threads=4
    )
    
    # Create provider
    provider = LLMProviderFactory.create_provider(config)
    if not provider:
        print("✗ Failed to create LLM provider")
        return False
    
    print(f"✓ Created {config.provider_type.value} provider")
    
    # Initialize provider
    print("Initializing provider...")
    success = await provider.initialize()
    if not success:
        print("✗ Failed to initialize provider")
        return False
    
    print("✓ Provider initialized successfully")
    
    # Get model info
    model_info = provider.get_model_info()
    print(f"\nModel Information:")
    print(f"   Name: {model_info.get('name', 'Unknown')}")
    print(f"   Path: {model_info.get('path', 'Unknown')}")
    print(f"   Size: {model_info.get('size', 0) / (1024*1024):.1f} MB")
    print(f"   Context Window: {model_info.get('context_window', 'Unknown')}")
    print(f"   Quantization: {model_info.get('quantization', 'Unknown')}")
    print(f"   Parameters: {model_info.get('parameters', 'Unknown')}")
    
    # Test generation
    print(f"\nTesting model generation...")
    try:
        response = await provider.generate_response(
            "Hello! Can you tell me a short joke?",
            system_prompt="You are a helpful assistant."
        )
        
        print(f"✓ Generated response:")
        print(f"   Model: {response.model}")
        print(f"   Provider: {response.provider}")
        print(f"   Response time: {response.response_time:.2f}s")
        print(f"   Text: {response.text}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False
    
    # Cleanup
    await provider.cleanup()
    print("✓ Provider cleaned up")
    
    return True


async def main():
    """Main demonstration function."""
    try:
        # Demonstrate model manager
        success = await demonstrate_model_manager()
        if not success:
            return 1
        
        # Demonstrate LLM provider
        success = await demonstrate_llm_provider()
        if not success:
            return 1
        
        print("\n" + "="*50)
        print("DEMONSTRATION COMPLETE")
        print("="*50)
        print("✓ Model manager is working correctly")
        print("✓ LLM provider integration is working")
        print("✓ Gemma-3n model is ready to use")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
