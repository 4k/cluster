# LLM Service

Standalone LLM service that connects to Ollama instance at `http://192.168.1.144:11434`

## Installation

```bash
pip install -r llm_requirements.txt
```

## Usage

### Run the example

```bash
python llm_service.py
```

### Use in your code

```python
from llm_service import LLMService

# Initialize service
llm = LLMService("http://192.168.1.144:11434")

# Check connection
if llm.check_connection():
    # Send a simple request
    response = llm.send_request("What is Python?", model="llama3.2")
    print(response)

    # Use chat interface
    response = llm.chat(
        message="Explain quantum computing",
        system="You are a physics teacher",
        model="llama3.2"
    )

    # Stream response
    response = llm.send_request(
        "Tell me a story",
        model="llama3.2",
        stream=True
    )
```

## API

### LLMService(base_url)
Initialize the service with Ollama URL

### send_request(prompt, model, stream, **kwargs)
Send a request to Ollama
- `prompt`: Text prompt
- `model`: Model name (default: "llama3.2")
- `stream`: Stream response (default: False)
- `**kwargs`: Additional Ollama parameters (temperature, max_tokens, etc.)

### chat(message, model, system, **kwargs)
Send a chat message
- `message`: User message
- `model`: Model name
- `system`: Optional system prompt
- `**kwargs`: Additional parameters

### check_connection()
Verify Ollama is accessible and list available models

## Features

✓ Simple request/response interface
✓ Streaming support
✓ Chat with system prompts
✓ Custom parameters (temperature, max_tokens, etc.)
✓ Connection checking
✓ Error handling
✓ Response printing

## Notes

- This service is standalone and NOT connected to other services (STT, TTS, etc.)
- Responses are printed to console
- Default timeout is 120 seconds
- Default model is "llama3.2"
