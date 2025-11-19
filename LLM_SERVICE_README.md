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

# Initialize service (auto-detects best endpoint)
llm = LLMService("http://192.168.1.144:11434")

# Or specify API type explicitly
llm = LLMService(
    base_url="http://192.168.1.144:11434",
    api_type="auto",  # auto|ollama-generate|ollama-chat|openai
    api_key=None  # optional auth token
)

# Run diagnostics
llm.diagnose()

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

### LLMService(base_url, api_key=None, api_type="auto")
Initialize the service with Ollama URL
- `base_url`: Ollama instance URL
- `api_key`: Optional authentication token
- `api_type`: API endpoint type (auto-detects by default)

### diagnose()
Run comprehensive endpoint diagnostics to identify connectivity issues

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
- Default model is "llama3.2"
- Automatically detects the best API endpoint
