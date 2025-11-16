#!/usr/bin/env python3
"""
Model Manager CLI for Voice Assistant
Provides easy model switching and management capabilities.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.llm import LLMManager, LLMConfig


class ModelManagerCLI:
    """Command-line interface for model management."""
    
    def __init__(self):
        self.llm_manager = None
    
    async def initialize(self):
        """Initialize the LLM manager."""
        # Default config
        config = {
            "model_path": "models/phi3-mini.gguf",
            "context_window": 2048,
            "temperature": 0.7,
            "max_tokens": 512,
            "n_gpu_layers": 0,
            "n_threads": 4
        }
        
        self.llm_manager = LLMManager(config)
        await self.llm_manager.initialize()
    
    async def list_models(self, models_dir: str = "models"):
        """List all available models."""
        models = self.llm_manager.discover_models(models_dir)
        
        if not models:
            print("No models found in the models directory.")
            return
        
        print(f"\nAvailable Models in '{models_dir}':")
        print("=" * 80)
        print(f"{'Name':<30} {'Size (MB)':<12} {'Status':<15} {'Extension':<10}")
        print("-" * 80)
        
        for model in models:
            size_mb = model['size'] / (1024 * 1024)
            status = "Current" if model['is_current'] else ("Cached" if model['is_loaded'] else "Available")
            print(f"{model['name']:<30} {size_mb:<12.1f} {status:<15} {model['extension']:<10}")
        
        print(f"\nTotal: {len(models)} models")
    
    async def switch_model(self, model_path: str, **kwargs):
        """Switch to a different model."""
        success = await self.llm_manager.switch_model(model_path, **kwargs)
        
        if success:
            print(f"✅ Successfully switched to model: {model_path}")
            model_info = self.llm_manager.get_model_info()
            print(f"   Model: {model_info['name']}")
            print(f"   Size: {model_info['size'] / (1024 * 1024):.1f} MB")
        else:
            print(f"❌ Failed to switch to model: {model_path}")
            return False
        
        return True
    
    async def show_current_model(self):
        """Show information about the current model."""
        model_info = self.llm_manager.get_model_info()
        
        if model_info['status'] == 'no_model_loaded':
            print("No model currently loaded.")
            return
        
        print("\nCurrent Model:")
        print("=" * 50)
        print(f"Name: {model_info['name']}")
        print(f"Path: {model_info['path']}")
        print(f"Size: {model_info['size'] / (1024 * 1024):.1f} MB")
        print(f"Cached: {'Yes' if model_info['is_cached'] else 'No'}")
        print(f"Context Window: {model_info['config']['context_window']}")
        print(f"Temperature: {model_info['config']['temperature']}")
        print(f"Max Tokens: {model_info['config']['max_tokens']}")
        print(f"GPU Layers: {model_info['config']['n_gpu_layers']}")
        print(f"Threads: {model_info['config']['n_threads']}")
    
    async def unload_model(self, model_path: str):
        """Unload a specific model."""
        success = self.llm_manager.unload_model(model_path)
        
        if success:
            print(f"✅ Successfully unloaded model: {model_path}")
        else:
            print(f"❌ Failed to unload model: {model_path}")
    
    async def unload_all_models(self):
        """Unload all cached models."""
        count = self.llm_manager.unload_all_models()
        print(f"✅ Unloaded {count} models from cache")
    
    async def test_model(self, model_path: str, test_prompt: str = "Hello, how are you?"):
        """Test a model with a simple prompt."""
        print(f"Testing model: {model_path}")
        print(f"Prompt: {test_prompt}")
        print("-" * 50)
        
        # Switch to the model
        success = await self.llm_manager.switch_model(model_path)
        if not success:
            print("❌ Failed to load model for testing")
            return
        
        # Generate response
        from core.types import ConversationContext, ConversationTurn
        
        context = ConversationContext(turns=[])
        response = await self.llm_manager.generate_response(test_prompt, context)
        
        print(f"Response: {response}")
        print("-" * 50)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.llm_manager:
            await self.llm_manager.cleanup()


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Voice Assistant Model Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--dir', default='models', help='Models directory path')
    
    # Switch model command
    switch_parser = subparsers.add_parser('switch', help='Switch to a different model')
    switch_parser.add_argument('model_path', help='Path to the model file')
    switch_parser.add_argument('--gpu-layers', type=int, help='Number of GPU layers')
    switch_parser.add_argument('--threads', type=int, help='Number of CPU threads')
    switch_parser.add_argument('--temperature', type=float, help='Temperature setting')
    switch_parser.add_argument('--max-tokens', type=int, help='Maximum tokens')
    
    # Show current model command
    subparsers.add_parser('current', help='Show current model information')
    
    # Unload model command
    unload_parser = subparsers.add_parser('unload', help='Unload a specific model')
    unload_parser.add_argument('model_path', help='Path to the model file to unload')
    
    # Unload all models command
    subparsers.add_parser('unload-all', help='Unload all cached models')
    
    # Test model command
    test_parser = subparsers.add_parser('test', help='Test a model with a prompt')
    test_parser.add_argument('model_path', help='Path to the model file')
    test_parser.add_argument('--prompt', default='Hello, how are you?', help='Test prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize model manager
    cli = ModelManagerCLI()
    await cli.initialize()
    
    try:
        if args.command == 'list':
            await cli.list_models(args.dir)
        
        elif args.command == 'switch':
            kwargs = {}
            if args.gpu_layers is not None:
                kwargs['n_gpu_layers'] = args.gpu_layers
            if args.threads is not None:
                kwargs['n_threads'] = args.threads
            if args.temperature is not None:
                kwargs['temperature'] = args.temperature
            if args.max_tokens is not None:
                kwargs['max_tokens'] = args.max_tokens
            
            await cli.switch_model(args.model_path, **kwargs)
        
        elif args.command == 'current':
            await cli.show_current_model()
        
        elif args.command == 'unload':
            await cli.unload_model(args.model_path)
        
        elif args.command == 'unload-all':
            await cli.unload_all_models()
        
        elif args.command == 'test':
            await cli.test_model(args.model_path, args.prompt)
    
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
