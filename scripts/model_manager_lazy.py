#!/usr/bin/env python3
"""
Model Manager for Lazy Loading Voice Assistant
Provides CLI interface for managing models with lazy loading.
Works with event bus architecture (no HTTP API).
"""

import asyncio
import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.llm import LLMManager
from core.types import ConversationContext
from core.config import AssistantConfig


class LazyModelManager:
    """Manager for lazy loading models."""
    
    def __init__(self):
        self.llm_manager = None
    
    async def initialize(self, config_path: str = None):
        """Initialize the LLM manager with lazy loading."""
        if config_path:
            # Load config from file
            config = AssistantConfig.load_from_file(config_path)
        else:
            # Use default config with environment overrides
            config = AssistantConfig()
        
        self.llm_manager = LLMManager(config.llm)
        await self.llm_manager.initialize()
    
    async def list_models(self, models_dir: str = "models") -> List[Dict[str, Any]]:
        """List available models."""
        return self.llm_manager.discover_models(models_dir)
    
    async def switch_model(self, model_path: str, **kwargs) -> bool:
        """Switch to a different model."""
        return await self.llm_manager.switch_model(model_path, **kwargs)
    
    async def get_current_model(self) -> Dict[str, Any]:
        """Get current model information."""
        return self.llm_manager.get_model_info()
    
    async def test_model(self, model_path: str, prompt: str = "Hello, how are you?", **kwargs) -> str:
        """Test a model with a prompt."""
        success = await self.llm_manager.switch_model(model_path, **kwargs)
        if not success:
            return "Failed to load model"
        
        context = ConversationContext(turns=[])
        response = await self.llm_manager.generate_response(prompt, context)
        return response or "No response generated"
    
    async def unload_current(self) -> bool:
        """Unload current model."""
        model_info = self.llm_manager.get_model_info()
        if model_info.get('status') == 'no_model_loaded':
            return True
        
        current_model = model_info.get('path')
        if current_model:
            return await self.llm_manager.unload_model(current_model)
        return True
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        return self.llm_manager.get_memory_usage()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.llm_manager:
            await self.llm_manager.cleanup()


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Lazy Model Manager for Voice Assistant")
    parser.add_argument("command", choices=["list", "switch", "current", "test", "unload", "memory"], 
                       help="Command to execute")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--gpu-layers", type=int, help="Number of GPU layers")
    parser.add_argument("--threads", type=int, help="Number of CPU threads")
    parser.add_argument("--temperature", type=float, help="Temperature setting")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    parser.add_argument("--context-window", type=int, help="Context window size")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize manager with config
    manager = LazyModelManager()
    await manager.initialize(args.config)
    
    try:
        if args.command == "list":
            models = await manager.list_models(args.models_dir)
            print(f"Found {len(models)} models:")
            for model in models:
                status = "✓" if model["is_current"] else ("●" if model["is_loaded"] else "○")
                size_mb = model["size"] / (1024 * 1024)
                print(f"  {status} {model['name']} ({size_mb:.1f}MB) - {model['path']}")
        
        elif args.command == "switch":
            if not args.model_path:
                print("Error: --model-path is required for switch command")
                sys.exit(1)
            
            kwargs = {}
            if args.gpu_layers is not None:
                kwargs["n_gpu_layers"] = args.gpu_layers
            if args.threads is not None:
                kwargs["n_threads"] = args.threads
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature
            if args.max_tokens is not None:
                kwargs["max_tokens"] = args.max_tokens
            if args.context_window is not None:
                kwargs["context_window"] = args.context_window
            
            success = await manager.switch_model(args.model_path, **kwargs)
            if success:
                print(f"✓ Successfully switched to {args.model_path}")
            else:
                print(f"✗ Failed to switch to {args.model_path}")
                sys.exit(1)
        
        elif args.command == "current":
            model_info = await manager.get_current_model()
            if model_info.get("status") == "no_model_loaded":
                print("No model currently loaded")
            else:
                print(f"Current model: {model_info['name']}")
                print(f"Path: {model_info['path']}")
                print(f"Size: {model_info['size'] / (1024 * 1024):.1f}MB")
                print(f"Loaded: {model_info['is_loaded']}")
                print(f"Cached: {model_info['is_cached']}")
                if 'config' in model_info:
                    print("Configuration:")
                    for key, value in model_info['config'].items():
                        print(f"  {key}: {value}")
        
        elif args.command == "test":
            if not args.model_path:
                print("Error: --model-path is required for test command")
                sys.exit(1)
            
            kwargs = {}
            if args.gpu_layers is not None:
                kwargs["n_gpu_layers"] = args.gpu_layers
            if args.threads is not None:
                kwargs["n_threads"] = args.threads
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature
            if args.max_tokens is not None:
                kwargs["max_tokens"] = args.max_tokens
            if args.context_window is not None:
                kwargs["context_window"] = args.context_window
            
            print(f"Testing model: {args.model_path}")
            print(f"Prompt: {args.prompt}")
            print("Response:")
            response = await manager.test_model(args.model_path, args.prompt, **kwargs)
            print(response)
        
        elif args.command == "unload":
            success = await manager.unload_current()
            if success:
                print("✓ Successfully unloaded current model")
            else:
                print("✗ Failed to unload current model")
                sys.exit(1)
        
        elif args.command == "memory":
            usage = manager.get_memory_usage()
            print("Memory Usage:")
            for key, value in usage.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
