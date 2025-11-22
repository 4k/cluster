# llm_service.py - Language Model Service Analysis

## Overview

`llm_service.py` provides the LLM integration layer, connecting to Ollama instances for natural language processing. It supports multiple API types, auto-detection, and event-driven response generation.

## File Location
`/home/user/cluster/src/services/llm_service.py`

## Class: LLMService

### Initialization

```python
LLMService(
    base_url: str = None,           # Ollama instance URL
    api_key: Optional[str] = None,  # Optional auth
    api_type: str = None,           # 'auto', 'ollama-chat', 'ollama-generate', 'openai'
    default_model: str = None,      # Model name
    system_prompt: Optional[str] = None,
    config: LLMServiceConfig = None # Override with config object
)
```

### Key Methods

| Method | Purpose | Async |
|--------|---------|-------|
| `initialize()` | Connect to event bus | Yes |
| `generate_async()` | Generate response with events | Yes |
| `send_request()` | Low-level API call | No |
| `chat()` | Simple chat interface | No |
| `check_connection()` | Test Ollama connectivity | No |
| `diagnose()` | Run comprehensive diagnostics | No |

### Event Subscriptions

- `SPEECH_DETECTED` → `_on_speech_detected()` - Process user speech
- `AMBIENT_SPEECH_DETECTED` → `_on_ambient_speech_detected()` - Process ambient speech
- `SYSTEM_STOPPED` → `_on_system_stopped()` - Cleanup

### Event Emissions

- `RESPONSE_GENERATING` - When starting response generation
- `RESPONSE_GENERATED` - When response is ready (includes timing metrics)
- `ERROR_OCCURRED` - On generation failure

### API Type Detection

Auto-detection probes endpoints in order:
1. `ollama-chat` → `/api/chat`
2. `ollama-generate` → `/api/generate`
3. `openai` → `/v1/chat/completions`

### Response Handling

```python
# Ollama generate format
{"response": "text"}

# Ollama chat format
{"message": {"content": "text"}}

# OpenAI format
{"choices": [{"message": {"content": "text"}}]}
```

## Architecture Flow

```
SPEECH_DETECTED event
       ↓
_on_speech_detected()
       ↓
generate_async(text)
       ↓
Emit RESPONSE_GENERATING
       ↓
run_in_executor(send_request)
       ↓
Emit RESPONSE_GENERATED
       ↓
TTS service receives response
```

## Error Handling

- Connection errors with clear diagnostics
- Timeout handling (configurable)
- HTTP error status reporting
- JSON parsing error recovery

## Improvements Suggested

### 1. Response Streaming
Implement true async streaming:
```python
async def stream_response(self, prompt: str) -> AsyncIterator[str]:
    """Stream response tokens as they're generated."""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            async for line in response.content:
                yield json.loads(line).get('response', '')
```

### 2. Context Window Management
Track and manage conversation history:
```python
def _prepare_context(self, prompt: str) -> str:
    """Prepare prompt with conversation history within token limit."""
    history = self._get_recent_history()
    available_tokens = self.context_window - self._count_tokens(prompt)
    return self._build_context(history, available_tokens, prompt)
```

### 3. Model Fallback
Implement fallback to alternative models:
```python
FALLBACK_MODELS = ["llama3.2:3b", "llama2:7b", "mistral:7b"]

async def generate_with_fallback(self, prompt: str) -> str:
    for model in [self.default_model] + FALLBACK_MODELS:
        try:
            return await self.generate_async(prompt, model=model)
        except Exception:
            continue
    raise RuntimeError("All models failed")
```

### 4. Response Caching
Cache frequent responses:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _cached_response(self, prompt_hash: str) -> Optional[str]:
    """Check cache for similar prompts."""
    return self._cache.get(prompt_hash)
```

### 5. Rate Limiting
Implement request rate limiting:
```python
from asyncio import Semaphore

class LLMService:
    def __init__(self):
        self._semaphore = Semaphore(3)  # Max 3 concurrent requests

    async def generate_async(self, prompt: str):
        async with self._semaphore:
            return await self._do_generate(prompt)
```

### 6. Token Counting
Add token estimation:
```python
def estimate_tokens(self, text: str) -> int:
    """Estimate token count (rough approximation)."""
    return len(text) // 4  # ~4 chars per token average
```
