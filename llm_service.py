"""
Standalone LLM Service
Connects to Ollama instance and processes requests
"""

import requests
import json
from typing import Optional, Dict, Any


class LLMService:
    """Simple LLM service that connects to Ollama"""

    def __init__(self, base_url: str = "http://192.168.1.144:11434"):
        """
        Initialize LLM service

        Args:
            base_url: Base URL of Ollama instance
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        print(f"LLM Service initialized with Ollama at: {self.base_url}")

    def send_request(
        self,
        prompt: str,
        model: str = "llama3.2",
        stream: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Send request to Ollama and get response

        Args:
            prompt: The text prompt to send
            model: Model name to use (default: llama3.2)
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response or None if error
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }

        try:
            print(f"\n{'='*60}")
            print(f"Sending request to {self.api_url}")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}\n")

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
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
        model: str = "llama3.2",
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Send a chat message

        Args:
            message: User message
            model: Model name
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
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get('models', [])
            print(f"✓ Connected to Ollama at {self.base_url}")
            print(f"Available models: {len(models)}")
            for model in models:
                print(f"  - {model.get('name', 'unknown')}")
            return True
        except Exception as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            return False


def main():
    """Example usage of LLM Service"""

    # Initialize service
    llm = LLMService("http://192.168.1.144:11434")

    # Check connection
    if not llm.check_connection():
        print("\nPlease start Ollama and try again")
        return

    print("\n" + "="*60)
    print("LLM Service Ready")
    print("="*60 + "\n")

    # Example 1: Simple request
    print("Example 1: Simple request")
    llm.send_request("What is the capital of France?", model="llama3.2")

    # Example 2: Chat with system prompt
    print("\nExample 2: Chat with system prompt")
    llm.chat(
        message="Tell me a joke",
        system="You are a helpful and funny assistant",
        model="llama3.2"
    )

    # Example 3: Streaming response
    print("\nExample 3: Streaming response")
    llm.send_request(
        "Count from 1 to 5",
        model="llama3.2",
        stream=True
    )

    # Example 4: With parameters
    print("\nExample 4: With custom parameters")
    llm.send_request(
        "Write a haiku about coding",
        model="llama3.2",
        temperature=0.7,
        max_tokens=100
    )


if __name__ == "__main__":
    main()
