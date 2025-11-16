#!/usr/bin/env python3
"""
Example demonstrating the unified LLM system.
Shows how to use local and remote providers seamlessly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.llm import LLMManager
from core.types import ConversationContext, ConversationTurn


async def test_local_provider():
    """Test local provider with llama.cpp."""
    print("🔧 Testing Local Provider (llama.cpp)")
    print("=" * 50)
    
    config = {
        "provider_type": "local",
        "model": "phi3-mini.gguf",
        "model_path": "models/phi3-mini.gguf",
        "temperature": 0.7,
        "max_tokens": 256,
        "context_window": 2048,
        "n_gpu_layers": 0,
        "n_threads": 4
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        print("✅ Local provider initialized")
        
        # Test conversation
        context = ConversationContext(turns=[])
        response = await llm_manager.generate_response("Hello, how are you?", context)
        print(f"Response: {response}")
        
        # Show provider info
        info = llm_manager.get_current_provider_info()
        print(f"Provider: {info['provider_type']}")
        print(f"Model: {info['model']}")
        
    except Exception as e:
        print(f"❌ Local provider failed: {e}")
    finally:
        await llm_manager.cleanup()


async def test_openai_provider():
    """Test OpenAI provider."""
    print("\n🤖 Testing OpenAI Provider")
    print("=" * 50)
    
    config = {
        "provider_type": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "your-openai-api-key-here",  # Replace with actual key
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        print("✅ OpenAI provider initialized")
        
        # Test conversation
        context = ConversationContext(turns=[])
        response = await llm_manager.generate_response("What is the capital of France?", context)
        print(f"Response: {response}")
        
        # Show available models
        models = llm_manager.get_available_models("openai")
        print(f"Available OpenAI models: {len(models)}")
        for model in models[:3]:  # Show first 3
            print(f"  - {model['name']} ({model['id']})")
        
    except Exception as e:
        print(f"❌ OpenAI provider failed: {e}")
    finally:
        await llm_manager.cleanup()


async def test_anthropic_provider():
    """Test Anthropic provider."""
    print("\n🧠 Testing Anthropic Provider")
    print("=" * 50)
    
    config = {
        "provider_type": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "api_key": "your-anthropic-api-key-here",  # Replace with actual key
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        print("✅ Anthropic provider initialized")
        
        # Test conversation
        context = ConversationContext(turns=[])
        response = await llm_manager.generate_response("Tell me a short joke.", context)
        print(f"Response: {response}")
        
        # Show capabilities
        info = llm_manager.get_current_provider_info()
        capabilities = info.get('capabilities', {})
        print(f"Supports vision: {capabilities.get('vision', False)}")
        print(f"Supports streaming: {capabilities.get('streaming', False)}")
        
    except Exception as e:
        print(f"❌ Anthropic provider failed: {e}")
    finally:
        await llm_manager.cleanup()


async def test_provider_switching():
    """Test switching between providers."""
    print("\n🔄 Testing Provider Switching")
    print("=" * 50)
    
    # Start with local provider
    config = {
        "provider_type": "local",
        "model": "phi3-mini.gguf",
        "model_path": "models/phi3-mini.gguf",
        "temperature": 0.7,
        "max_tokens": 128
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        print("✅ Started with local provider")
        
        # Test local response
        context = ConversationContext(turns=[])
        response = await llm_manager.generate_response("Hello!", context)
        print(f"Local response: {response}")
        
        # Show available providers
        providers = llm_manager.get_available_providers()
        print(f"\nAvailable providers: {len(providers)}")
        for provider in providers:
            print(f"  - {provider['name']} ({provider['type']})")
        
        # Try to switch to OpenAI (will fail without API key)
        print("\nTrying to switch to OpenAI...")
        success = await llm_manager.switch_provider("openai", api_key="test-key")
        if success:
            print("✅ Switched to OpenAI")
            response = await llm_manager.generate_response("Hello from OpenAI!", context)
            print(f"OpenAI response: {response}")
        else:
            print("❌ Failed to switch to OpenAI (expected without valid API key)")
        
    except Exception as e:
        print(f"❌ Provider switching failed: {e}")
    finally:
        await llm_manager.cleanup()


async def test_conversation_context():
    """Test conversation context handling."""
    print("\n💬 Testing Conversation Context")
    print("=" * 50)
    
    config = {
        "provider_type": "local",
        "model": "phi3-mini.gguf",
        "model_path": "models/phi3-mini.gguf",
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        
        # Build conversation context
        context = ConversationContext(turns=[
            ConversationTurn(
                speaker="user",
                text="My name is Alice",
                timestamp=asyncio.get_event_loop().time(),
                confidence=1.0
            ),
            ConversationTurn(
                speaker="assistant",
                text="Nice to meet you, Alice!",
                timestamp=asyncio.get_event_loop().time(),
                confidence=1.0
            )
        ])
        
        # Test with context
        response = await llm_manager.generate_response("What's my name?", context)
        print(f"Response with context: {response}")
        
        # Test with memory context
        memory_context = "User mentioned they like pizza and live in New York."
        response = await llm_manager.generate_response("What do you know about me?", context, memory_context)
        print(f"Response with memory: {response}")
        
    except Exception as e:
        print(f"❌ Conversation context test failed: {e}")
    finally:
        await llm_manager.cleanup()


async def test_streaming():
    """Test streaming responses."""
    print("\n🌊 Testing Streaming Responses")
    print("=" * 50)
    
    config = {
        "provider_type": "local",
        "model": "phi3-mini.gguf",
        "model_path": "models/phi3-mini.gguf",
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True
    }
    
    llm_manager = LLMManager(config)
    
    try:
        await llm_manager.initialize()
        
        context = ConversationContext(turns=[])
        print("Streaming response:")
        
        # Note: The current implementation doesn't expose streaming directly
        # This would need to be added to the LLMManager interface
        response = await llm_manager.generate_response("Tell me a story about a robot.", context)
        print(f"Full response: {response}")
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
    finally:
        await llm_manager.cleanup()


async def main():
    """Run all tests."""
    print("🚀 Unified LLM System Test Suite")
    print("=" * 60)
    
    # Test local provider
    await test_local_provider()
    
    # Test OpenAI provider (will fail without API key)
    await test_openai_provider()
    
    # Test Anthropic provider (will fail without API key)
    await test_anthropic_provider()
    
    # Test provider switching
    await test_provider_switching()
    
    # Test conversation context
    await test_conversation_context()
    
    # Test streaming
    await test_streaming()
    
    print("\n✅ All tests completed!")
    print("\nTo test remote providers:")
    print("1. Get API keys from OpenAI and Anthropic")
    print("2. Update the API keys in this script")
    print("3. Run the tests again")


if __name__ == "__main__":
    asyncio.run(main())
