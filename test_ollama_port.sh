#!/bin/bash

# Test Ollama Port Accessibility
# Run this after configuring Unraid Docker port mapping

OLLAMA_IP="192.168.1.144"
OLLAMA_PORT="11434"
OLLAMA_URL="http://${OLLAMA_IP}:${OLLAMA_PORT}"

echo "============================================================"
echo "OLLAMA PORT TEST"
echo "============================================================"
echo "Target: ${OLLAMA_URL}"
echo "============================================================"
echo ""

# Test 1: Check if port is reachable
echo "Test 1: Port Connectivity"
echo "-----------------------------------------------------------"
if command -v nc &> /dev/null; then
    nc -zv ${OLLAMA_IP} ${OLLAMA_PORT} 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Port ${OLLAMA_PORT} is OPEN"
    else
        echo "✗ Port ${OLLAMA_PORT} is CLOSED or UNREACHABLE"
        echo ""
        echo "Possible issues:"
        echo "  1. Ollama container not running"
        echo "  2. Port not mapped in Docker settings"
        echo "  3. Firewall blocking access"
        exit 1
    fi
else
    echo "⚠ netcat not available, skipping port check"
fi
echo ""

# Test 2: HTTP Root
echo "Test 2: HTTP Root Endpoint"
echo "-----------------------------------------------------------"
curl -v -m 5 ${OLLAMA_URL}/ 2>&1 | grep -E "HTTP|Connected|Access denied|404|200"
echo ""

# Test 3: API Tags (list models)
echo "Test 3: API Tags Endpoint (List Models)"
echo "-----------------------------------------------------------"
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -m 10 ${OLLAMA_URL}/api/tags 2>&1)
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

echo "HTTP Status: ${http_code}"
echo "Response:"
echo "$body" | head -20
echo ""

if [ "$http_code" = "200" ]; then
    echo "✓ SUCCESS - Ollama is accessible!"
    echo ""
    echo "Models found:"
    echo "$body" | grep -o '"name":"[^"]*"' | cut -d'"' -f4
elif [ "$http_code" = "403" ]; then
    echo "✗ FORBIDDEN (403) - Access denied by proxy/auth"
    echo ""
    echo "This means:"
    echo "  - Port is open and reachable"
    echo "  - But a proxy (like Envoy) is blocking access"
    echo "  - You need to find the correct URL that bypasses the proxy"
    echo "  - Check what URL Open WebUI uses"
elif [ "$http_code" = "404" ]; then
    echo "✗ NOT FOUND (404) - Endpoint doesn't exist"
    echo ""
    echo "Try different endpoints:"
    echo "  - /v1/chat/completions"
    echo "  - /api/chat"
    echo "  - /api/generate"
else
    echo "✗ FAILED - Cannot reach Ollama"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check if Ollama Docker container is running"
    echo "  2. Verify port mapping in Unraid Docker settings"
    echo "  3. Try using host network mode instead of bridge"
fi

echo ""
echo "============================================================"
echo "TEST COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. If test passed: Run 'python llm_service.py'"
echo "  2. If 403 error: Check Open WebUI settings for correct URL"
echo "  3. If failed: Review UNRAID_PORT_CONFIG.md for setup help"
echo ""
