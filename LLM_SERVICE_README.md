# LLM Service

Standalone LLM service that connects to Ollama instance at `http://192.168.1.144:11434`

## Features

✓ Auto-detection of API endpoints
✓ Support for multiple API types (Ollama native, OpenAI-compatible)
✓ Authentication support
✓ Streaming responses
✓ Comprehensive diagnostics
✓ Enhanced error reporting

## Installation

```bash
pip install -r llm_requirements.txt
```

## Usage

### Run diagnostics

```bash
python llm_service.py
```

This will:
1. Auto-detect the working API endpoint
2. Run comprehensive diagnostics
3. Test connection and list models
4. Run example requests

### Use in your code

```python
from llm_service import LLMService

# Initialize service with your model
llm = LLMService(
    base_url="http://192.168.1.144:11434",
    default_model="qwen2.5-coder:32b"  # YOUR MODEL HERE - only define ONCE!
)

# Or specify all parameters
llm = LLMService(
    base_url="http://192.168.1.144:11434",
    api_type="auto",  # auto|ollama-generate|ollama-chat|openai
    api_key=None,  # optional auth token
    default_model="qwen2.5-coder:32b"  # YOUR MODEL HERE
)

# Run diagnostics
llm.diagnose()

# Check connection
if llm.check_connection():
    # Send a simple request (uses default model)
    response = llm.send_request("What is Python?")
    print(response)

    # Use chat interface (uses default model)
    response = llm.chat(
        message="Explain quantum computing",
        system="You are a physics teacher"
    )

    # Stream response (uses default model)
    response = llm.send_request(
        "Tell me a story",
        stream=True
    )

    # Override model for specific request
    response = llm.send_request(
        "Quick question",
        model="llama3.2"  # Use different model just for this request
    )
```

## API

### LLMService(base_url, api_key=None, api_type="auto", default_model="qwen2.5-coder:32b")
Initialize the service with Ollama URL
- `base_url`: Ollama instance URL
- `api_key`: Optional authentication token
- `api_type`: API endpoint type (auto-detects by default)
- `default_model`: Default model name - **DEFINE ONCE HERE!**

### diagnose()
Run comprehensive endpoint diagnostics to identify connectivity issues

### send_request(prompt, model=None, stream=False, **kwargs)
Send a request to Ollama
- `prompt`: Text prompt
- `model`: Model name (default: None = uses instance default_model)
- `stream`: Stream response (default: False)
- `**kwargs`: Additional Ollama parameters (temperature, max_tokens, etc.)

### chat(message, model=None, system=None, **kwargs)
Send a chat message
- `message`: User message
- `model`: Model name (default: None = uses instance default_model)
- `system`: Optional system prompt
- `**kwargs`: Additional parameters

### check_connection()
Verify Ollama is accessible and list available models

## Troubleshooting

### Quick Port Test

Run the automated port test script first:

```bash
./test_ollama_port.sh
```

This will diagnose connectivity issues automatically.

### Unraid Docker Bridge Mode Port Configuration

**If your Ollama container is in bridge mode**, you need to configure port mapping:

1. See detailed instructions in: **[UNRAID_PORT_CONFIG.md](UNRAID_PORT_CONFIG.md)**
2. Quick fix:
   - Stop Ollama container
   - Edit container → Add port mapping
   - Container Port: `11434` → Host Port: `11434`
   - Apply and restart

**Alternative:** Switch to host network mode (easier, no port mapping needed)

### Getting 403 Access Denied

If you see "403 Access denied" errors:

1. **Check if there's a proxy** (like Envoy) in front of Ollama
2. **Find the correct URL** - Check what URL Open WebUI uses (in its settings/config)
3. **Check Docker port mapping** - Ollama usually runs on port 11434, but Docker might expose it differently
4. **Try different ports** - Common alternatives: 8080, 3000, 11435
5. **Authentication** - Some setups require an API key

### Finding your Ollama URL

```bash
# Check Docker container ports
docker ps | grep ollama

# Check Unraid container settings
# Look at "Port Mappings" section

# Check Open WebUI configuration
# Settings > Admin > Connections > Ollama API URL
```

### Testing with different URLs

```python
# Try different URL/port combinations
llm = LLMService("http://192.168.1.144:11434")  # Default
llm = LLMService("http://192.168.1.144:8080")   # Alternative
llm = LLMService("http://localhost:11434")       # Local

# With authentication
llm = LLMService(
    base_url="http://192.168.1.144:11434",
    api_key="your-api-key"
)
```

## Notes

- This service is standalone and NOT connected to other services (STT, TTS, etc.)
- Responses are printed to console
- Default timeout is 120 seconds
- **Default model is "qwen2.5-coder:32b"** - Change this in `__init__()` to use your model!
- Model name only needs to be defined ONCE in the constructor
- Automatically detects the best API endpoint
- You can override the model per-request if needed
