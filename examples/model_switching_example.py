#!/usr/bin/env python3
"""
Example script demonstrating model switching capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.llm import LLMManager
from core.types import ConversationContext, ConversationTurn


async def main():
    """Demonstrate model switching functionality."""
    print("🤖 Voice Assistant Model Switching Example")
    print("=" * 50)
    
    # Initialize LLM manager with default config
    config = {
        "model_path": "models/phi3-mini.gguf",  # Default model
        "context_window": 2048,
        "temperature": 0.7,
        "max_tokens": 512,
        "n_gpu_layers": 0,
        "n_threads": 4
    }
    
    llm_manager = LLMManager(config)
    await llm_manager.initialize()
    
    print(f"✅ LLM Manager initialized")
    print(f"   llama.cpp available: {llm_manager.get_state()['llama_cpp_available']}")
    print()
    
    # Discover available models
    print("🔍 Discovering available models...")
    models = llm_manager.discover_models("models")
    
    if not models:
        print("   No models found in models/ directory")
        print("   Please download some GGUF models to test switching")
        return
    
    print(f"   Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        size_mb = model['size'] / (1024 * 1024)
        status = "Current" if model['is_current'] else ("Cached" if model['is_loaded'] else "Available")
        print(f"   {i}. {model['name']} ({size_mb:.1f} MB) - {status}")
    print()
    
    # Test conversation context
    context = ConversationContext(turns=[])
    
    # Test with each available model
    for i, model in enumerate(models[:3]):  # Test first 3 models
        print(f"🔄 Testing Model {i+1}: {model['name']}")
        print("-" * 30)
        
        # Switch to the model
        success = await llm_manager.switch_model(model['path'])
        
        if not success:
            print(f"   ❌ Failed to load model: {model['name']}")
            continue
        
        # Test with a simple prompt
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me a short joke."
        ]
        
        for prompt in test_prompts:
            print(f"   Prompt: {prompt}")
            
            try:
                response = await llm_manager.generate_response(prompt, context)
                print(f"   Response: {response}")
            except Exception as e:
                print(f"   Error: {e}")
            
            print()
        
        # Add conversation to context
        context.turns.append(ConversationTurn(
            speaker="user",
            text=test_prompts[0],
            timestamp=asyncio.get_event_loop().time(),
            confidence=1.0
        ))
        
        if response:
            context.turns.append(ConversationTurn(
                speaker="assistant",
                text=response,
                timestamp=asyncio.get_event_loop().time(),
                confidence=1.0
            ))
        
        print()
    
    # Show final state
    print("📊 Final LLM Manager State:")
    state = llm_manager.get_state()
    print(f"   Current model: {state['current_model_path']}")
    print(f"   Cached models: {len(state['cached_models'])}")
    print(f"   Available models: {len(state['available_models'])}")
    print()
    
    # Test model switching performance
    print("⚡ Testing model switching performance...")
    
    if len(models) >= 2:
        model1, model2 = models[0], models[1]
        
        # Time switching between two models
        start_time = asyncio.get_event_loop().time()
        
        for i in range(3):
            await llm_manager.switch_model(model1['path'])
            await llm_manager.switch_model(model2['path'])
        
        end_time = asyncio.get_event_loop().time()
        avg_switch_time = (end_time - start_time) / 6  # 6 switches total
        
        print(f"   Average switch time: {avg_switch_time:.2f} seconds")
        print(f"   Model 1: {model1['name']}")
        print(f"   Model 2: {model2['name']}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await llm_manager.cleanup()
    print("   ✅ Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
