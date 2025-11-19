"""
Standalone LLM Service
Connects to Ollama instance and processes requests
"""

import requests
import json
from typing import Optional, Dict, Any


class LLMService:
    """Simple LLM service that connects to Ollama"""

    def __init__(
        self,
        base_url: str = "http://192.168.1.144:11434",
        api_key: Optional[str] = None,
        api_type: str = "auto",
        default_model: str = "qwen2.5-coder:32b"
    ):
        """
        Initialize LLM service

        Args:
            base_url: Base URL of Ollama instance
            api_key: Optional API key for authentication
            api_type: API type - 'auto', 'ollama', 'openai', or 'chat'
            default_model: Default model name to use for requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_type = api_type
        self.default_model = default_model
        self.detected_endpoint = None

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


def main():
    """Example usage of LLM Service"""

    # Initialize service with your model
    llm = LLMService(
        base_url="http://192.168.1.144:11434",
        default_model="qwen2.5-coder:32b"  # Change this to your model
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
    print("LLM Service Ready")
    print("="*60 + "\n")

    # Example 1: Simple request (uses default model)
    print("Example 1: Simple request")
    llm.send_request("What is the capital of France?")

    # Example 2: Chat with system prompt (uses default model)
    print("\nExample 2: Chat with system prompt")
    llm.chat(
        message="Tell me a joke",
        system="You are a helpful and funny assistant"
    )

    # Example 3: Streaming response (uses default model)
    print("\nExample 3: Streaming response")
    llm.send_request(
        "Count from 1 to 5",
        stream=True
    )

    # Example 4: With custom parameters (uses default model)
    print("\nExample 4: With custom parameters")
    llm.send_request(
        "Write a haiku about coding",
        temperature=0.7,
        max_tokens=100
    )


if __name__ == "__main__":
    main()
