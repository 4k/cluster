# Configuring Ollama Port Access in Unraid Docker (Bridge Mode)

If your Ollama container is in **bridge mode** and you need local network access, follow these steps:

## Step-by-Step Configuration

### 1. Stop the Ollama Container
- Go to Docker tab in Unraid
- Click on the Ollama container
- Click "Stop"

### 2. Edit Container Settings
- Click the container icon
- Select "Edit"

### 3. Add/Fix Port Mapping

Look for the **Port Mappings** section. You need to add:

```
Container Port: 11434
Host Port: 11434
Connection Type: TCP
```

**Important Settings:**
- **Container Port (Internal)**: `11434` (this is what Ollama uses internally)
- **Host Port (External)**: `11434` (this is what you'll access from your network)
- **Protocol**: TCP
- **IP**: Leave as `0.0.0.0` (allows all local network access)

### 4. Visual Guide

In the Unraid Docker edit screen:

```
Add another Path, Port, Variable, Label or Device

Config Type:     [Port]
Name:           Ollama API
Container Port: 11434
Host Port:      11434
Default Value:  11434
Connection Type: TCP
Display:        Always
Required:       Yes
```

### 5. Apply and Start
- Scroll to bottom
- Click "Apply"
- Container will restart automatically

### 6. Verify Port is Open

From your Unraid terminal (or SSH):

```bash
# Check if Ollama container is listening
docker ps | grep ollama

# Check port mapping (should show 0.0.0.0:11434->11434/tcp)
docker port <container_name>

# Test from Unraid host
curl http://localhost:11434/api/tags

# Test from your local network
curl http://192.168.1.144:11434/api/tags
```

## Alternative: Host Network Mode (Easier)

If bridge mode port mapping doesn't work, you can switch to **host mode**:

1. Edit container
2. Find **Network Type**
3. Change from `bridge` to `host`
4. Apply

**Note:** In host mode, the container uses the host's network directly, so port mapping isn't needed.

## Common Issues

### Issue 1: Port Already in Use
If port 11434 is already used:
- Use a different host port (e.g., `11435`)
- Access via `http://192.168.1.144:11435`

### Issue 2: Firewall Blocking
Unraid's built-in firewall might block:
- Go to Settings > Network Settings
- Check firewall rules
- Ensure port 11434 is allowed

### Issue 3: Still Getting 403
If you still get 403 errors after fixing ports:
- There might be an auth proxy in front of Ollama
- Check if Open WebUI is bundled with Ollama
- Open WebUI sometimes runs on port 3000 and proxies to Ollama
- Try accessing Open WebUI's proxy instead

## Finding Open WebUI's Ollama Connection

Open WebUI often runs as a separate container that connects to Ollama. Check:

1. **Open WebUI Container Settings**
   - Look for environment variables
   - Find `OLLAMA_API_BASE_URL` or similar
   - This shows the URL Open WebUI uses

2. **From Open WebUI Interface**
   - Open Open WebUI in browser
   - Go to Settings (gear icon)
   - Look for "Connections" or "Admin Settings"
   - Find "Ollama API URL"

3. **Common Setups**
   ```
   Scenario 1: Separate containers
   - Ollama: http://192.168.1.144:11434
   - Open WebUI: http://192.168.1.144:3000

   Scenario 2: Docker network
   - Ollama: http://ollama:11434 (internal Docker network)
   - Open WebUI connects internally
   - External access needs proxy

   Scenario 3: All-in-one
   - Everything on port 3000
   - Ollama proxied through Open WebUI
   ```

## Quick Test After Configuration

Run this to test if the port is accessible:

```bash
# From Unraid console
curl -v http://192.168.1.144:11434/api/tags

# If this works, your Python script should work too
python llm_service.py
```

## What to Try Next

1. **First**: Fix the port mapping in bridge mode (steps above)
2. **Test**: `curl http://192.168.1.144:11434/api/tags`
3. **If fails**: Switch to host network mode
4. **If still fails**: Use Open WebUI's proxy URL instead
5. **Last resort**: Check if there's an API key required

## Expected Output When Working

When the port is properly configured:

```bash
$ curl http://192.168.1.144:11434/api/tags
{"models":[{"name":"llama3.2:latest",...}]}
```

When it's NOT working:

```bash
$ curl http://192.168.1.144:11434/api/tags
curl: (7) Failed to connect to 192.168.1.144 port 11434: Connection refused
# OR
Access denied
```
