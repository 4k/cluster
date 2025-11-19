"""
Diagnostic script to check Ollama API endpoints
"""

import requests
import json

OLLAMA_URL = "http://192.168.1.144:11434"

def check_endpoint(path, method="GET", data=None):
    """Check if an endpoint is available"""
    url = f"{OLLAMA_URL}{path}"
    print(f"\nChecking: {method} {url}")
    print("-" * 60)

    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        try:
            json_data = response.json()
            print(f"Response: {json.dumps(json_data, indent=2)}")
        except:
            print(f"Response Text: {response.text[:500]}")

        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Check various endpoints
print("="*60)
print("OLLAMA API ENDPOINT DIAGNOSTICS")
print("="*60)

# Root endpoint
check_endpoint("/")

# API version
check_endpoint("/api/version")

# List models
check_endpoint("/api/tags")

# Generate endpoint (POST)
check_endpoint("/api/generate", method="POST", data={
    "model": "llama3.2",
    "prompt": "Hello",
    "stream": False
})

# Chat endpoint (POST) - v1 API
check_endpoint("/api/chat", method="POST", data={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": False
})

# OpenAI-compatible endpoint
check_endpoint("/v1/chat/completions", method="POST", data={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello"}]
})

# Check if models exist
check_endpoint("/api/show", method="POST", data={
    "name": "llama3.2"
})

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
