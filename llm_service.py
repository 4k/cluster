"""
Standalone LLM Service
Connects to Ollama instance and processes requests
Event-driven version that integrates with the event bus.
"""

import requests
import json
import asyncio
import logging
import time
from typing import Optional, Dict, Any

# Event bus imports
from src.core.event_bus import EventBus, EventType, emit_event

# Default configuration constants
DEFAULT_BASE_URL = "http://192.168.1.144:11434"
DEFAULT_MODEL = "qwen3-coder:30b"

logger = logging.getLogger(__name__)


class LLMService:
    """Simple LLM service that connects to Ollama. Event-driven with async support."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        api_type: str = "auto",
        default_model: str = DEFAULT_MODEL,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM service

        Args:
            base_url: Base URL of Ollama instance
            api_key: Optional API key for authentication
            api_type: API type - 'auto', 'ollama', 'openai', or 'chat'
            default_model: Default model name to use for requests
            system_prompt: Optional system prompt to use for all requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_type = api_type
        self.default_model = default_model
        self.system_prompt = system_prompt or "You are a helpful voice assistant. Provide concise, natural responses suitable for voice output."
        self.detected_endpoint = None
        self.event_bus = None

        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        print(f"LLM Service initialized with Ollama at: {self.base_url}")
        print(f"Default model: {self.default_model}")
        if api_key:
            print(f"Using authentication: Yes")

        # Auto-detect best endpoint if requested
        if api_type == "auto":
            self._detect_endpoint()

    async def initialize(self):
        """Initialize event bus connection and subscribe to events."""
        logger.info("Initializing LLM service with event bus...")
        self.event_bus = await EventBus.get_instance()

        # Subscribe to speech detected events
        self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
        self.event_bus.subscribe(EventType.AMBIENT_SPEECH_DETECTED, self._on_ambient_speech_detected)
        self.event_bus.subscribe(EventType.SYSTEM_STOPPED, self._on_system_stopped)

        logger.info("LLM service initialized with event bus")

    async def _on_system_stopped(self, event):
        """Handle system stopped event."""
        logger.info("Received SYSTEM_STOPPED event")

    async def _on_speech_detected(self, event):
        """Handle speech detected event."""
        text = event.data.get("text", "")
        correlation_id = event.correlation_id

        logger.info(f"Processing speech: {text} (correlation_id: {correlation_id})")

        # Generate response asynchronously
        response = await self.generate_async(text, correlation_id=correlation_id)

        if response:
            logger.info(f"Generated response: {response[:100]}...")
        else:
            logger.error("Failed to generate response")

    async def _on_ambient_speech_detected(self, event):
        """Handle ambient speech detected event."""
        text = event.data.get("text", "")
        correlation_id = event.correlation_id

        logger.info(f"Processing ambient speech: {text} (correlation_id: {correlation_id})")

        # Could handle differently than wake word speech
        response = await self.generate_async(text, correlation_id=correlation_id)

        if response:
            logger.info(f"Generated ambient response: {response[:100]}...")

    async def generate_async(self, prompt: str, model: Optional[str] = None, correlation_id: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generate response asynchronously with event bus integration.

        Args:
            prompt: The text prompt
            model: Model name to use (default: uses instance default_model)
            correlation_id: Correlation ID for request tracing
            **kwargs: Additional parameters

        Returns:
            Generated text response or None if error
        """
        start_time = time.time()

        # Emit response generating event
        if self.event_bus:
            await emit_event(
                EventType.RESPONSE_GENERATING,
                {
                    "prompt": prompt,
                    "model": model or self.default_model,
                    "timestamp": start_time
                },
                correlation_id=correlation_id,
                source="llm"
            )

        try:
            # Run blocking request in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.send_request(prompt, model=model, **kwargs)
            )

            if response:
                elapsed_time = time.time() - start_time

                # Emit response generated event
                if self.event_bus:
                    await emit_event(
                        EventType.RESPONSE_GENERATED,
                        {
                            "response": response,
                            "prompt": prompt,
                            "model": model or self.default_model,
                            "elapsed_time": elapsed_time,
                            "timestamp": time.time(),
                            "word_count": len(response.split())
                        },
                        correlation_id=correlation_id,
                        source="llm"
                    )

                return response
            else:
                # Emit error event
                if self.event_bus:
                    await emit_event(
                        EventType.ERROR_OCCURRED,
                        {
                            "error": "No response from LLM",
                            "service": "llm",
                            "operation": "generation"
                        },
                        correlation_id=correlation_id,
                        source="llm"
                    )
                return None

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)

            # Emit error event
            if self.event_bus:
                await emit_event(
                    EventType.ERROR_OCCURRED,
                    {
                        "error": str(e),
                        "service": "llm",
                        "operation": "generation"
                    },
                    correlation_id=correlation_id,
                    source="llm"
                )
            return None

    def _detect_endpoint(self):
        """Auto-detect which API endpoint works"""
        print("Auto-detecting API endpoint...")

        endpoints = [
            ("ollama-chat", "/api/chat"),
            ("ollama-generate", "/api/generate"),
            ("openai", "/v1/chat/completions"),
        ]

        for name, path in endpoints:
            try:
                url = f"{self.base_url}{path}"
                # Try a minimal request
                test_payload = {
                    "model": self.default_model,
                    "prompt": "test" if "generate" in path else None,
                    "messages": [{"role": "user", "content": "test"}] if "chat" in path else None,
                    "stream": False
                }
                # Remove None values
                test_payload = {k: v for k, v in test_payload.items() if v is not None}

                response = requests.post(
                    url,
                    json=test_payload,
                    headers=self.headers,
                    timeout=5
                )

                if response.status_code in [200, 201]:
                    print(f"✓ Detected working endpoint: {name} ({path})")
                    self.detected_endpoint = name
                    self.api_type = name
                    return
                else:
                    print(f"✗ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ {name}: {str(e)[:50]}")

        print("⚠ Could not auto-detect endpoint, will try all methods")
        self.api_type = "ollama-generate"  # fallback

    def send_request(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Send request to Ollama and get response

        Args:
            prompt: The text prompt to send
            model: Model name to use (default: uses instance default_model)
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response or None if error
        """
        # Use default model if not specified
        if model is None:
            model = self.default_model

        # Determine endpoint and payload based on API type
        if self.api_type in ["openai", "openai-compat"]:
            api_url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs
            }
        elif self.api_type in ["ollama-chat", "chat"]:
            api_url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                **kwargs
            }
        else:  # ollama-generate or default
            api_url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }

        try:
            print(f"\n{'='*60}")
            print(f"Sending request to {api_url}")
            print(f"API Type: {self.api_type}")
            print(f"Model: {model}")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"{'='*60}\n")

            response = requests.post(
                api_url,
                json=payload,
                headers=self.headers,
                timeout=120
            )

            # Enhanced error reporting
            if response.status_code != 200:
                print(f"ERROR: HTTP {response.status_code}")
                print(f"Response: {response.text[:500]}")
                response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return self._handle_normal_response(response)

        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to Ollama at {self.base_url}")
            print("Please ensure Ollama is running and accessible")
            return None
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed - {e}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error - {e}")
            return None

    def _handle_normal_response(self, response: requests.Response) -> Optional[str]:
        """Handle non-streaming response"""
        try:
            data = response.json()

            # Extract response based on API type
            if self.api_type in ["openai", "openai-compat"]:
                # OpenAI format
                answer = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            elif self.api_type in ["ollama-chat", "chat"]:
                # Ollama chat format
                answer = data.get('message', {}).get('content', '')
            else:
                # Ollama generate format
                answer = data.get('response', '')

            print("Response received:")
            print(f"{'-'*60}")
            print(answer)
            print(f"{'-'*60}\n")

            return answer
        except json.JSONDecodeError:
            print("ERROR: Invalid JSON response")
            return None

    def _handle_stream_response(self, response: requests.Response) -> Optional[str]:
        """Handle streaming response"""
        full_response = ""
        print("Streaming response:")
        print(f"{'-'*60}")

        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get('response', '')
                    full_response += chunk
                    print(chunk, end='', flush=True)

                    if data.get('done', False):
                        break

            print(f"\n{'-'*60}\n")
            return full_response
        except json.JSONDecodeError:
            print("\nERROR: Invalid JSON in stream")
            return None

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Send a chat message

        Args:
            message: User message
            model: Model name (default: uses instance default_model)
            system: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        # Build prompt with system message if provided
        if system:
            prompt = f"System: {system}\n\nUser: {message}\n\nAssistant:"
        else:
            prompt = message

        return self.send_request(prompt, model=model, **kwargs)

    def check_connection(self) -> bool:
        """
        Check if Ollama is accessible

        Returns:
            True if connection successful, False otherwise
        """
        print("\n" + "="*60)
        print("CONNECTION CHECK")
        print("="*60)

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self.headers,
                timeout=5
            )

            print(f"Status: HTTP {response.status_code}")

            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✓ Connected to Ollama at {self.base_url}")
                print(f"Available models: {len(models)}")
                for model in models:
                    print(f"  - {model.get('name', 'unknown')}")
                print("="*60 + "\n")
                return True
            elif response.status_code == 403:
                print(f"✗ Access denied (403)")
                print(f"Response: {response.text}")
                print("\nPossible issues:")
                print("  1. Proxy/Envoy blocking access")
                print("  2. Authentication required")
                print("  3. Network/firewall restrictions")
                print("  4. Try a different port or URL")
                print("="*60 + "\n")
                return False
            else:
                print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
                print("="*60 + "\n")
                return False

        except requests.exceptions.ConnectionError as e:
            print(f"✗ Connection Error: Cannot reach {self.base_url}")
            print(f"Details: {e}")
            print("\nPossible issues:")
            print("  1. Ollama is not running")
            print("  2. Wrong IP address or port")
            print("  3. Network connectivity issues")
            print("="*60 + "\n")
            return False
        except Exception as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            print("="*60 + "\n")
            return False

    def diagnose(self):
        """Run comprehensive diagnostics"""
        print("\n" + "="*60)
        print("OLLAMA DIAGNOSTICS")
        print("="*60)
        print(f"Base URL: {self.base_url}")
        print(f"Default Model: {self.default_model}")
        print(f"API Type: {self.api_type}")
        print(f"Auth: {'Yes' if self.api_key else 'No'}")
        print("="*60 + "\n")

        # Test various endpoints
        endpoints = [
            ("Root", "/", "GET"),
            ("Tags/Models", "/api/tags", "GET"),
            ("Version", "/api/version", "GET"),
            ("Generate", "/api/generate", "POST"),
            ("Chat", "/api/chat", "POST"),
            ("OpenAI Chat", "/v1/chat/completions", "POST"),
        ]

        for name, path, method in endpoints:
            url = f"{self.base_url}{path}"
            print(f"\nTesting: {name}")
            print(f"  URL: {url}")
            print(f"  Method: {method}")

            try:
                if method == "GET":
                    resp = requests.get(url, headers=self.headers, timeout=5)
                else:
                    test_data = {
                        "model": self.default_model,
                        "prompt": "test" if path == "/api/generate" else None,
                        "messages": [{"role": "user", "content": "test"}] if "chat" in path else None,
                        "stream": False
                    }
                    test_data = {k: v for k, v in test_data.items() if v is not None}
                    resp = requests.post(url, json=test_data, headers=self.headers, timeout=5)

                print(f"  Status: {resp.status_code}")
                if resp.status_code == 200:
                    print(f"  ✓ SUCCESS")
                elif resp.status_code == 403:
                    print(f"  ✗ FORBIDDEN - Access denied")
                elif resp.status_code == 404:
                    print(f"  ✗ NOT FOUND - Endpoint doesn't exist")
                else:
                    print(f"  ✗ {resp.text[:100]}")

            except Exception as e:
                print(f"  ✗ ERROR: {str(e)[:80]}")

        print("\n" + "="*60 + "\n")


async def main_async():
    """Async main for LLM service with event bus."""
    # Initialize event bus
    bus = await EventBus.get_instance()
    await bus.start()

    # Emit system started event
    await emit_event(EventType.SYSTEM_STARTED, {"service": "llm"}, source="llm")

    # Initialize service with your model
    llm = LLMService(
        base_url=DEFAULT_BASE_URL,
        default_model=DEFAULT_MODEL
    )

    # Run diagnostics first
    llm.diagnose()

    # Check connection
    if not llm.check_connection():
        print("\nCannot connect to Ollama.")
        print("Please check:")
        print("  1. Is Ollama running in Docker?")
        print("  2. Is the IP/port correct?")
        print("  3. Check your docker-compose ports mapping")
        print("  4. What URL does Open WebUI use? Try that URL here.")
        return

    print("\n" + "="*60)
    print("LLM Service Ready (Event-Driven)")
    print("="*60 + "\n")

    # Initialize event bus integration
    await llm.initialize()

    print("LLM service is now listening for SPEECH_DETECTED events...")
    print("Press Ctrl+C to stop")

    try:
        # Keep service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping LLM service...")
    finally:
        # Emit system stopped event and stop bus
        await emit_event(EventType.SYSTEM_STOPPED, {"service": "llm"}, source="llm")
        await bus.stop()


def main():
    """Example usage of LLM Service"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run async main
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
