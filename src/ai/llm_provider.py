"""
Unified LLM Provider - Single provider supporting mock, local, and OpenAI-compatible APIs.
Replaces the complex factory pattern with a single class that handles all modes.
"""

import logging
import time
import random
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Types of LLM providers."""
    MOCK = "mock"
    LOCAL = "local"
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class LLMConfig:
    """Configuration for the unified LLM provider."""
    provider_type: str = "local"  # "mock", "local", or "openai"
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    timeout: int = 30
    retries: int = 3
    
    # API settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    
    # Local model settings
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    context_window: int = 2048
    n_gpu_layers: int = 0
    n_threads: int = 4
    use_mmap: bool = True
    use_mlock: bool = False
    
    # GPU offloading parameters for Raspberry Pi
    ngl: int = 33
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None


class UnifiedLLMProvider:
    """Unified LLM provider supporting mock, local, and OpenAI-compatible modes."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_initialized = False
        self.is_available = False
        self.model_info: Optional[Dict[str, Any]] = None
        
        # Mode-specific components
        self.local_model = None
        self.openai_client = None
        
        # Mock responses - simplified to random quotes
        self.mock_quotes = [
            "Hello! I'm a mock AI assistant.",
            "That's an interesting question.",
            "I understand what you're asking.",
            "Thank you for your input.",
            "I'm here to help test the system.",
            "This is a mock response for testing.",
            "The system is working as expected.",
            "I can simulate conversation in mock mode.",
            "What would you like to know?",
            "I'm ready to help with your questions."
        ]
    
    async def initialize(self) -> bool:
        """Initialize the provider based on its mode."""
        try:
            if self.config.provider_type == "mock":
                return await self._init_mock()
            elif self.config.provider_type == "local":
                return await self._init_local()
            elif self.config.provider_type == "openai":
                return await self._init_openai()
            else:
                logger.error(f"Unknown provider type: {self.config.provider_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.provider_type} provider: {e}")
            return False
    
    async def _init_mock(self) -> bool:
        """Initialize mock mode."""
        self.is_initialized = True
        self.is_available = True
        
        self.model_info = {
            'name': 'Mock LLM',
            'model': 'mock-llm-v1.0',
            'provider': 'mock',
            'context_window': 4096,
            'parameters': '0B (mock)',
            'description': 'Mock LLM provider for testing and development'
        }
        
        logger.info("Mock LLM provider initialized successfully")
        return True
    
    async def _init_local(self) -> bool:
        """Initialize local mode with llama.cpp."""
        try:
            # Import llama-cpp-python
            from llama_cpp import Llama
            
            if not self.config.model_path:
                logger.error("Model path required for local provider")
                return False
            
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize llama.cpp model with GPU offloading
            self.local_model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_window,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=False,
                # GPU offloading parameters for Raspberry Pi
                ngl=getattr(self.config, 'ngl', 33),
                main_gpu=getattr(self.config, 'main_gpu', 0),
                tensor_split=getattr(self.config, 'tensor_split', None)
            )
            
            self.is_initialized = True
            self.is_available = True
            
            self.model_info = {
                'name': f'Local Model ({model_path.name})',
                'model': self.config.model,
                'provider': 'local',
                'context_window': self.config.context_window,
                'parameters': 'Unknown',
                'description': f'Local model running with llama.cpp'
            }
            
            logger.info(f"Local LLM provider initialized with model: {model_path}")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not available for local provider")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            return False
    
    async def _init_openai(self) -> bool:
        """Initialize OpenAI-compatible API mode."""
        try:
            # Import OpenAI client
            from openai import AsyncOpenAI
            
            if not self.config.api_key:
                logger.error("API key required for OpenAI provider")
                return False
            
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization
            )
            
            self.is_initialized = True
            self.is_available = True
            
            self.model_info = {
                'name': f'OpenAI API ({self.config.model})',
                'model': self.config.model,
                'provider': 'openai',
                'context_window': 4096,  # Default, varies by model
                'parameters': 'Unknown',
                'description': f'OpenAI API model: {self.config.model}'
            }
            
            logger.info(f"OpenAI provider initialized with model: {self.config.model}")
            return True
            
        except ImportError:
            logger.error("OpenAI client not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response based on the current mode."""
        if not self.is_initialized:
            raise RuntimeError("LLM provider not initialized")
        
        start_time = time.time()
        
        try:
            if self.config.provider_type == "mock":
                return await self._generate_mock_response(prompt, start_time)
            elif self.config.provider_type == "local":
                return await self._generate_local_response(prompt, system_prompt, conversation_history, start_time)
            elif self.config.provider_type == "openai":
                return await self._generate_openai_response(prompt, system_prompt, conversation_history, start_time)
            else:
                raise RuntimeError(f"Unknown provider type: {self.config.provider_type}")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def _generate_mock_response(self, prompt: str, start_time: float) -> LLMResponse:
        """Generate a simple mock response - instant random quote."""
        # No delay - instant response
        response_text = random.choice(self.mock_quotes)
        response_time = time.time() - start_time
        
        return LLMResponse(
            text=response_text,
            model=self.config.model,
            provider="mock",
            usage={
                'prompt_tokens': len(prompt.split()) if prompt else 0,
                'completion_tokens': len(response_text.split()),
                'total_tokens': len(prompt.split()) + len(response_text.split()) if prompt else len(response_text.split())
            },
            metadata={'mock_mode': True},
            finish_reason='stop',
            response_time=response_time
        )
    
    async def _generate_local_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        start_time: float
    ) -> LLMResponse:
        """Generate response using local llama.cpp model."""
        if not self.local_model:
            raise RuntimeError("Local model not initialized")
        
        # Build full prompt
        full_prompt = self._build_prompt(prompt, system_prompt, conversation_history)
        
        # Generate response
        response = self.local_model(
            full_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=self.config.stop
        )
        
        response_text = response['choices'][0]['text']
        response_time = time.time() - start_time
        
        # Calculate token counts (llama.cpp doesn't always provide usage info)
        prompt_tokens = len(full_prompt.split()) if full_prompt else 0
        completion_tokens = len(response_text.split()) if response_text else 0
        
        return LLMResponse(
            text=response_text,
            model=self.config.model,
            provider="local",
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            },
            metadata={'local_model': True},
            finish_reason='stop',
            response_time=response_time
        )
    
    async def _generate_openai_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        start_time: float
    ) -> LLMResponse:
        """Generate response using OpenAI-compatible API."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = await self.openai_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            stop=self.config.stop
        )
        
        response_text = response.choices[0].message.content
        response_time = time.time() - start_time
        
        return LLMResponse(
            text=response_text,
            model=self.config.model,
            provider="openai",
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            metadata={'api_model': True},
            finish_reason=response.choices[0].finish_reason,
            response_time=response_time
        )
    
    def _build_prompt(self, prompt: str, system_prompt: Optional[str], conversation_history: Optional[List[Dict[str, str]]]) -> str:
        """Build full prompt for local models."""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if conversation_history:
            for msg in conversation_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                parts.append(f"{role.title()}: {content}")
        
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response."""
        if not self.is_initialized:
            raise RuntimeError("LLM provider not initialized")
        
        # For now, generate full response and stream it word by word
        # This can be enhanced later for true streaming
        full_response = await self.generate_response(prompt, system_prompt, conversation_history, **kwargs)
        
        words = full_response.text.split()
        for i, word in enumerate(words):
            partial_text = " ".join(words[:i+1])
            yield LLMResponse(
                text=partial_text,
                model=full_response.model,
                provider=full_response.provider,
                usage=full_response.usage,
                metadata=full_response.metadata,
                finish_reason='stop' if i == len(words) - 1 else None,
                response_time=full_response.response_time
            )
            await asyncio.sleep(0.05)  # Small delay between words
    
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        if self.local_model:
            # llama.cpp models don't need explicit cleanup
            self.local_model = None
        
        if self.openai_client:
            # OpenAI client doesn't need explicit cleanup
            self.openai_client = None
        
        self.is_initialized = False
        self.is_available = False
        logger.info(f"{self.config.provider_type} provider cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return self.model_info or {}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for this provider."""
        if self.config.provider_type == "mock":
            return [{'id': 'mock-llm-v1.0', 'name': 'Mock LLM', 'description': 'Mock model for testing'}]
        elif self.config.provider_type == "local":
            # Return local model info if available
            if self.model_info:
                return [self.model_info]
            return []
        elif self.config.provider_type == "openai":
            # Return current model info
            if self.model_info:
                return [self.model_info]
            return []
        return []
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'function_calling': self.config.provider_type == "openai",
            'vision': False,
            'json_mode': False,
            'max_context_length': self.config.context_window,
            'supports_system_prompt': True,
            'supports_conversation_history': True
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            'provider_type': self.config.provider_type,
            'model': self.config.model,
            'is_initialized': self.is_initialized,
            'is_available': self.is_available,
            'model_info': self.model_info,
            'capabilities': self.get_capabilities()
        }
