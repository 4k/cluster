"""
Simplified LLM Manager with integrated model management.
Combines LLM provider management with essential model functionality.
"""

import logging
import asyncio
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from core.types import ConversationContext
from core.event_bus import EventBus, EventType, emit_event
from .llm_provider import UnifiedLLMProvider, LLMConfig, LLMResponse
from .memory import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a local model."""
    id: str
    name: str
    repo_id: str
    local_path: str
    file_patterns: List[str]
    size_bytes: int = 0
    downloaded_at: Optional[datetime] = None
    is_default: bool = False
    description: str = ""
    quantization: str = ""
    context_window: int = 2048
    parameters: str = ""
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    n_threads: Optional[int] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    ngl: Optional[int] = None
    main_gpu: Optional[int] = None
    tensor_split: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.downloaded_at:
            data['downloaded_at'] = self.downloaded_at.isoformat()
        
        # Remove None values for optional fields
        optional_fields = [
            'temperature', 'top_k', 'top_p', 'min_p', 'max_tokens',
            'frequency_penalty', 'presence_penalty', 'stop', 'stream',
            'timeout', 'retries', 'n_gpu_layers', 'n_threads',
            'use_mmap', 'use_mlock', 'ngl', 'main_gpu', 'tensor_split'
        ]
        for field in optional_fields:
            if data.get(field) is None:
                data.pop(field, None)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary."""
        if 'downloaded_at' in data and data['downloaded_at']:
            if isinstance(data['downloaded_at'], str) and data['downloaded_at']:
                try:
                    data['downloaded_at'] = datetime.fromisoformat(data['downloaded_at'])
                except ValueError:
                    data['downloaded_at'] = None
            elif not data['downloaded_at']:
                data['downloaded_at'] = None
        
        # Handle optional fields
        optional_fields = [
            'temperature', 'top_k', 'top_p', 'min_p', 'max_tokens',
            'frequency_penalty', 'presence_penalty', 'stop', 'stream',
            'timeout', 'retries', 'n_gpu_layers', 'n_threads',
            'use_mmap', 'use_mlock', 'ngl', 'main_gpu', 'tensor_split'
        ]
        for field in optional_fields:
            if field not in data:
                data[field] = None
                
        return cls(**data)


class LLMManager:
    """Simplified LLM Manager with integrated model management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM manager with configuration."""
        self.config = config
        self.provider: Optional[UnifiedLLMProvider] = None
        self.is_initialized = False
        self.is_processing = False
        
        # Model management
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.config_file = Path(config.get('models_config_file', 'models.json'))
        self.models: Dict[str, ModelInfo] = {}
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model configurations
        self._load_models_config()
        
        # Initialize memory manager if enabled
        self.memory_manager: Optional[MemoryManager] = None
        if config.get('enable_memory', True):
            memory_config = config.get('memory_config', {
                'max_tokens': 1000,
                'memory_file': 'data/memory.yaml',
                'max_memory_entries': 100
            })
            self.memory_manager = MemoryManager(memory_config)
        
        # Initialize chat logging
        self.chat_log_enabled = config.get('enable_chat_log', True)
        self.chat_log_file: Optional[Path] = None
        if self.chat_log_enabled:
            self._initialize_chat_log()
    
    def _load_models_config(self) -> None:
        """Load model configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for model_data in data.get('models', []):
                        model = ModelInfo.from_dict(model_data)
                        self.models[model.id] = model
                logger.info(f"Loaded {len(self.models)} model configurations")
            except Exception as e:
                logger.error(f"Failed to load models config: {e}")
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by its ID."""
        return self.models.get(model_id)
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get the local path for a model."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        model_path = self.models_dir / model.local_path
        
        if not model_path.exists():
            return None
        
        # Find the main model file (usually .gguf)
        gguf_files = list(model_path.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0]
        
        return model_path
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        model_path = self.get_model_path(model_id)
        return model_path is not None and model_path.exists()
    
    def get_model_availability(self, model_id: str) -> Dict[str, Any]:
        """Get information about model availability."""
        if model_id not in self.models:
            return {
                "available": False,
                "downloaded": False,
                "model_info": None,
                "error": f"Model '{model_id}' not found in configuration"
            }
        
        model_info = self.models[model_id]
        is_downloaded = self.is_model_downloaded(model_id)
        
        return {
            "available": is_downloaded,
            "downloaded": is_downloaded,
            "model_info": model_info,
            "model_id": model_id,
            "model_name": model_info.name,
            "repo_id": model_info.repo_id
        }
    
    async def initialize(self) -> None:
        """Initialize the LLM manager and create provider."""
        try:
            # Determine provider type and resolve model path
            provider_type = self.config.get('provider_type', 'local')
            model_id = self.config.get('model_id', self.config.get('model', 'mock-llm-v1.0'))
            model_path = self.config.get('model_path')
            
            # Get model info from models.json
            model_info = self.get_model_by_id(model_id)
            
            # For local provider, resolve model path from model_id if not provided
            if provider_type == 'local' and not model_path:
                model_path_obj = self.get_model_path(model_id)
                if model_path_obj:
                    model_path = str(model_path_obj)
                    logger.info(f"Resolved model path for {model_id}: {model_path}")
                else:
                    logger.warning(f"Model {model_id} not found locally, falling back to mock")
                    provider_type = 'mock'
                    model_id = 'mock-llm-v1.0'
                    model_info = None
            
            # Merge model settings from models.json with config overrides
            if model_info:
                # Use model.json settings as defaults, config overrides as final values
                temperature = self.config.get('temperature', model_info.temperature or 0.7)
                max_tokens = self.config.get('max_tokens', model_info.max_tokens or 512)
                top_p = self.config.get('top_p', model_info.top_p or 0.9)
                frequency_penalty = self.config.get('frequency_penalty', model_info.frequency_penalty or 0.0)
                presence_penalty = self.config.get('presence_penalty', model_info.presence_penalty or 0.0)
                stop = self.config.get('stop', model_info.stop)
                stream = self.config.get('stream', model_info.stream or False)
                timeout = self.config.get('timeout', model_info.timeout or 30)
                retries = self.config.get('retries', model_info.retries or 3)
                context_window = self.config.get('context_window', model_info.context_window or 2048)
                n_gpu_layers = self.config.get('n_gpu_layers', model_info.n_gpu_layers or 0)
                n_threads = self.config.get('n_threads', model_info.n_threads or 4)
                use_mmap = self.config.get('use_mmap', model_info.use_mmap if hasattr(model_info, 'use_mmap') else True)
                use_mlock = self.config.get('use_mlock', model_info.use_mlock if hasattr(model_info, 'use_mlock') else False)
                ngl = self.config.get('ngl', getattr(model_info, 'ngl', 33))
                main_gpu = self.config.get('main_gpu', getattr(model_info, 'main_gpu', 0))
                tensor_split = self.config.get('tensor_split', getattr(model_info, 'tensor_split', None))
            else:
                # Use config defaults for mock mode
                temperature = self.config.get('temperature', 0.7)
                max_tokens = self.config.get('max_tokens', 512)
                top_p = self.config.get('top_p', 0.9)
                frequency_penalty = self.config.get('frequency_penalty', 0.0)
                presence_penalty = self.config.get('presence_penalty', 0.0)
                stop = self.config.get('stop')
                stream = self.config.get('stream', False)
                timeout = self.config.get('timeout', 30)
                retries = self.config.get('retries', 3)
                context_window = self.config.get('context_window', 2048)
                n_gpu_layers = self.config.get('n_gpu_layers', 0)
                n_threads = self.config.get('n_threads', 4)
                use_mmap = self.config.get('use_mmap', True)
                use_mlock = self.config.get('use_mlock', False)
                ngl = self.config.get('ngl', 33)
                main_gpu = self.config.get('main_gpu', 0)
                tensor_split = self.config.get('tensor_split', None)
            
            # Create LLM config
            llm_config = LLMConfig(
                provider_type=provider_type,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream,
                timeout=timeout,
                retries=retries,
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url'),
                organization=self.config.get('organization'),
                model_path=model_path,
                model_id=model_id,
                context_window=context_window,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                ngl=ngl,
                main_gpu=main_gpu,
                tensor_split=tensor_split
            )
            
            # Create unified provider
            self.provider = UnifiedLLMProvider(llm_config)
            
            # Initialize provider
            success = await self.provider.initialize()
            if not success:
                # Fallback to mock if initialization fails
                logger.warning("Primary provider failed, falling back to mock")
                llm_config.provider_type = "mock"
                llm_config.model = "mock-llm-v1.0"
                llm_config.model_path = None
                self.provider = UnifiedLLMProvider(llm_config)
                await self.provider.initialize()
            
            self.is_initialized = True
            logger.info(f"LLM manager initialized with {llm_config.provider_type} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager: {e}")
            raise
    
    async def generate_response(
        self, 
        text: str, 
        context: ConversationContext,
        memory_context: str = "",
        **kwargs
    ) -> Optional[str]:
        """Generate a response using the unified provider."""
        if not self.is_initialized or not self.provider:
            raise RuntimeError("LLM manager not initialized")
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            # Emit generation started event
            await emit_event(EventType.RESPONSE_GENERATING, {
                "text": text,
                "context_length": len(context.turns) if hasattr(context, 'turns') else 0,
                "has_memory_context": bool(memory_context),
                "provider": self.provider.config.provider_type
            })
            
            # Get memory context if available
            if self.memory_manager and not memory_context:
                memory_context = self.memory_manager.get_memory_context()
            
            # Build system prompt
            system_prompt = self._build_system_prompt(memory_context)
            
            # Build conversation history
            conversation_history = self._build_conversation_history(context)
            
            # Log generation start
            prompt_preview = text[:100] if len(text) > 100 else text
            prompt_suffix = "..." if len(text) > 100 else ""
            logger.info(f"Starting to generate LLM response using {self.provider.config.provider_type} provider")
            logger.info(f"Prompt: {prompt_preview}{prompt_suffix}")
            
            # Generate response
            response = await self.provider.generate_response(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                **kwargs
            )
            
            # Log successful generation
            generation_time = time.time() - start_time
            
            # Truncate long responses for logging
            response_preview = response.text[:200] if len(response.text) > 200 else response.text
            response_suffix = "..." if len(response.text) > 200 else ""
            
            # Build performance info string
            perf_info = f"{generation_time:.2f}s"
            if response.usage and 'completion_tokens' in response.usage:
                completion_tokens = response.usage['completion_tokens']
                tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                perf_info += f", {completion_tokens} tokens ({tokens_per_sec:.2f} tokens/s)"
            
            logger.info(f"LLM response generated successfully in {perf_info} using {response.provider}/{response.model}")
            logger.info(f"Response: {response_preview}{response_suffix}")
            
            # Log chat exchange
            chat_metadata = {
                'provider': response.provider,
                'model': response.model,
                'generation_time': generation_time,
                'start_time': start_time
            }
            if response.usage and 'completion_tokens' in response.usage:
                completion_tokens = response.usage['completion_tokens']
                tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                chat_metadata['tokens'] = completion_tokens
                chat_metadata['tokens_per_second'] = tokens_per_sec
            
            self._log_chat_exchange(text, response.text, chat_metadata)
            
            # Emit generation completed event
            await emit_event(EventType.RESPONSE_GENERATED, {
                "response": response.text,
                "generation_time": time.time() - start_time,
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage
            })
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
        finally:
            self.is_processing = False
    
    async def generate_stream(
        self, 
        text: str, 
        context: ConversationContext,
        memory_context: str = "",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        if not self.is_initialized or not self.provider:
            raise RuntimeError("LLM manager not initialized")
        
        try:
            # Get memory context if available
            if self.memory_manager and not memory_context:
                memory_context = self.memory_manager.get_memory_context()
            
            # Build system prompt and conversation history
            system_prompt = self._build_system_prompt(memory_context)
            conversation_history = self._build_conversation_history(context)
            
            # Generate streaming response
            async for response in self.provider.generate_stream(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                **kwargs
            ):
                yield response.text
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
    
    def _build_system_prompt(self, memory_context: str = "") -> str:
        """Build system prompt for the LLM."""
        system_parts = [
            "You are a helpful voice assistant. Respond naturally and concisely to user queries.",
            "Be friendly, helpful, and engaging in your responses.",
            "Keep responses brief and conversational since this is a voice interface."
        ]
        
        if memory_context:
            system_parts.append(f"Context from previous conversations: {memory_context}")
        
        return "\n".join(system_parts)
    
    def _build_conversation_history(self, context: ConversationContext) -> List[Dict[str, str]]:
        """Build conversation history for the LLM."""
        history = []
        
        if hasattr(context, 'turns'):
            for turn in context.turns[-10:]:  # Last 10 turns for context
                if turn.speaker == "user":
                    history.append({"role": "user", "content": turn.text})
                elif turn.speaker == "assistant":
                    history.append({"role": "assistant", "content": turn.text})
        
        return history
    
    def _initialize_chat_log(self) -> None:
        """Initialize chat log file."""
        try:
            # Create logs/chats directory if it doesn't exist
            chat_log_dir = Path("logs/chats")
            chat_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.chat_log_file = chat_log_dir / f"chat_{timestamp}.log"
            
            # Write header
            with open(self.chat_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Chat Log - Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
            
            logger.info(f"Chat logging initialized: {self.chat_log_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat log: {e}")
            self.chat_log_enabled = False
    
    def _log_chat_exchange(self, user_input: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a chat exchange to the chat log file."""
        if not self.chat_log_enabled or not self.chat_log_file:
            return
        
        try:
            # Calculate timestamps
            if metadata and 'start_time' in metadata:
                user_timestamp = datetime.fromtimestamp(metadata['start_time']).strftime("%Y-%m-%d %H:%M:%S")
            else:
                user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            assistant_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.chat_log_file, 'a', encoding='utf-8') as f:
                # Write user input
                f.write(f"[{user_timestamp}] User:\n")
                f.write(f"{user_input}\n\n")
                
                # Write assistant response
                f.write(f"[{assistant_timestamp}] Assistant:\n")
                f.write(f"{assistant_response}\n")
                
                # Write metadata if available
                if metadata:
                    f.write(f"\n[Metadata: ")
                    meta_parts = []
                    if 'provider' in metadata:
                        meta_parts.append(f"provider={metadata['provider']}")
                    if 'model' in metadata:
                        meta_parts.append(f"model={metadata['model']}")
                    if 'generation_time' in metadata:
                        meta_parts.append(f"time={metadata['generation_time']:.2f}s")
                    if 'tokens' in metadata:
                        meta_parts.append(f"tokens={metadata['tokens']}")
                    if 'tokens_per_second' in metadata:
                        meta_parts.append(f"speed={metadata['tokens_per_second']:.2f} tok/s")
                    f.write(", ".join(meta_parts))
                    f.write("]\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
            
        except Exception as e:
            logger.error(f"Failed to write to chat log: {e}")
    
    async def switch_provider(self, provider_type: str, **kwargs) -> bool:
        """Switch to a different provider type."""
        try:
            if not self.provider:
                return False
            
            # Create new config with updated provider type
            new_config = LLMConfig(
                provider_type=provider_type,
                model=kwargs.get('model', self.provider.config.model),
                temperature=self.provider.config.temperature,
                max_tokens=self.provider.config.max_tokens,
                top_p=self.provider.config.top_p,
                frequency_penalty=self.provider.config.frequency_penalty,
                presence_penalty=self.provider.config.presence_penalty,
                stop=self.provider.config.stop,
                stream=self.provider.config.stream,
                timeout=self.provider.config.timeout,
                retries=self.provider.config.retries,
                api_key=kwargs.get('api_key', self.provider.config.api_key),
                base_url=kwargs.get('base_url', self.provider.config.base_url),
                organization=kwargs.get('organization', self.provider.config.organization),
                model_path=kwargs.get('model_path', self.provider.config.model_path),
                model_id=kwargs.get('model_id', self.provider.config.model_id),
                context_window=self.provider.config.context_window,
                n_gpu_layers=self.provider.config.n_gpu_layers,
                n_threads=self.provider.config.n_threads,
                use_mmap=self.provider.config.use_mmap,
                use_mlock=self.provider.config.use_mlock
            )
            
            # Cleanup old provider
            await self.provider.cleanup()
            
            # Create new provider
            self.provider = UnifiedLLMProvider(new_config)
            await self.provider.initialize()
            
            logger.info(f"Successfully switched to {provider_type} provider")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to {provider_type} provider: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the LLM manager."""
        state = {
            "is_initialized": self.is_initialized,
            "is_processing": self.is_processing,
            "provider_type": self.provider.config.provider_type if self.provider else None,
            "memory_enabled": self.memory_manager is not None,
            "config": self.config
        }
        
        if self.memory_manager:
            state['memory_stats'] = self.memory_manager.get_memory_stats()
        
        if self.provider:
            state['provider_status'] = self.provider.get_status()
        
        return state
    
    async def cleanup(self) -> None:
        """Cleanup LLM manager resources."""
        if self.provider:
            await self.provider.cleanup()
        
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        self.is_initialized = False
        logger.info("LLM manager cleaned up")
