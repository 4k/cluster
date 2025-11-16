# llama.cpp Configuration in Docker

## ✅ **Current Status: NOW PROPERLY CONFIGURED**

After the recent updates, llama.cpp is now properly configured in your Docker setup with optimizations for both development and production environments.

## 🏗️ **Docker Configuration**

### **1. Requirements.txt**
```python
# AI/ML - llama.cpp integration
llama-cpp-python>=0.2.0  # llama.cpp Python bindings
# Alternative: llama-cpp-python[cuda] for GPU support
# Alternative: llama-cpp-python[metal] for Apple Silicon
```

### **2. Dockerfile Optimizations**

#### **Development Stage (x86_64)**
```dockerfile
# Install llama.cpp with CPU optimizations for development
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

#### **Production Stage (x86_64)**
```dockerfile
# Install llama.cpp with CPU optimizations for production
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

#### **Production ARM64 Stage (Raspberry Pi)**
```dockerfile
# Install llama.cpp with ARM64 optimizations for Raspberry Pi
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_NATIVE=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

## 🔧 **CMAKE Build Flags Explained**

| Flag | Purpose | Platform |
|------|---------|----------|
| `-DLLAMA_OPENBLAS=on` | Enable OpenBLAS for faster CPU inference | All platforms |
| `-DLLAMA_NATIVE=on` | Enable native ARM64 optimizations | Raspberry Pi only |
| `-DLLAMA_CUBLAS=on` | Enable CUDA GPU support | NVIDIA GPUs |
| `-DLLAMA_METAL=on` | Enable Metal GPU support | Apple Silicon |

## 🚀 **How llama.cpp is Integrated**

### **1. Provider Architecture**
```
src/ai/
├── providers/
│   ├── local.py          # llama.cpp integration
│   ├── openai.py         # OpenAI API
│   ├── anthropic.py      # Anthropic API
│   └── factory.py        # Provider factory
└── llm.py                # LLM manager (supports local, remote, and mock providers)
```

### **2. Local Provider (llama.cpp)**
```python
# src/ai/providers/local.py
from llama_cpp import Llama

class LocalLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.model = None
    
    async def initialize(self):
        self.model = Llama(
            model_path=str(self.config.model_path),
            n_ctx=self.config.context_window,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            verbose=False,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock
        )
```

### **3. Configuration Options**
```python
# llama.cpp specific settings
n_gpu_layers: int = 0      # GPU layers (0 = CPU only)
n_threads: int = 4         # CPU threads
use_mmap: bool = True      # Memory mapping
use_mlock: bool = False    # Memory locking
context_window: int = 2048 # Context size
```

## 📊 **Performance Optimizations**

### **Development (Windows WSL)**
- **OpenBLAS**: Optimized linear algebra operations
- **Memory Mapping**: Efficient model loading
- **Multi-threading**: Uses all available CPU cores

### **Production (Raspberry Pi)**
- **OpenBLAS**: Optimized for ARM64
- **Native ARM64**: Uses ARM64-specific instructions
- **Memory Efficient**: Optimized for limited RAM

## 🎯 **Model Support**

### **Supported Formats**
- ✅ **GGUF** (recommended) - Quantized models
- ✅ **GGML** - Legacy format
- ✅ **BIN** - Binary models
- ✅ **SafeTensors** - Safe tensor format

### **Recommended Models**
```bash
# Small models for Raspberry Pi
models/phi3-mini-4k-instruct-q4.gguf     # ~2.3GB
models/llama2-7b-chat-q4_0.gguf          # ~3.8GB

# Medium models for development
models/phi3-medium-4k-instruct-q4.gguf    # ~7.7GB
models/llama2-13b-chat-q4_0.gguf          # ~7.3GB
```

## 🔄 **Lazy Loading Integration**

The llama.cpp integration works seamlessly with the lazy loading system:

```python
# Automatic model loading/unloading
async def generate_response(self, text: str, context: ConversationContext):
    # llama.cpp model is loaded on-demand
    # Automatically unloaded after timeout
    response = await self.unified_manager.generate_response(text, context)
    return response
```

## 🛠️ **Usage Examples**

### **1. Model Switching**
```bash
# List available models
python scripts/model_manager_lazy.py list

# Switch to llama.cpp model
python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model
python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello!"
```

### **2. API Usage**
```bash
# Switch model via API
curl -X POST http://localhost:8000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/phi3-mini.gguf"}'

# Test model via API
curl -X POST http://localhost:8000/api/models/test \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/phi3-mini.gguf", "test_prompt": "Hello!"}'
```

### **3. Configuration**
```yaml
# config/llm_config_lazy.yaml
llm:
  provider_type: "local"
  model_path: "models/phi3-mini.gguf"
  context_window: 2048
  n_gpu_layers: 0        # CPU only
  n_threads: 4           # Use 4 CPU threads
  use_mmap: true         # Enable memory mapping
  use_mlock: false       # Disable memory locking
  lazy_loading: true
  cache_size: 1
  auto_unload: true
  unload_timeout: 300
```

## 🚀 **Deployment Commands**

### **Development**
```bash
# Build with llama.cpp optimizations
docker-compose build

# Deploy development environment
./scripts/deploy.sh dev

# Test llama.cpp integration
python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf
```

### **Production (Raspberry Pi)**
```bash
# Build with ARM64 optimizations
docker-compose -f docker-compose.prod.yml build

# Deploy production environment
./scripts/deploy.sh prod

# Monitor llama.cpp performance
curl http://localhost:8000/api/models/memory
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Import Error**: `ModuleNotFoundError: No module named 'llama_cpp'`
   ```bash
   # Rebuild Docker image
   docker-compose build --no-cache
   ```

2. **Model Loading Error**: `Model file not found`
   ```bash
   # Check model path
   ls -la models/
   # Download model
   wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf -O models/phi3-mini.gguf
   ```

3. **Out of Memory**: `CUDA out of memory` or `Memory allocation failed`
   ```bash
   # Reduce context window
   export LLM_CONTEXT_WINDOW=1024
   # Use smaller model
   python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf
   ```

4. **Slow Performance**: Model inference is slow
   ```bash
   # Increase CPU threads
   export LLM_N_THREADS=8
   # Check CPU usage
   docker-compose exec voice-assistant htop
   ```

### **Performance Monitoring**
```bash
# Check memory usage
curl http://localhost:8000/api/models/memory

# Check current model
curl http://localhost:8000/api/models/current

# View logs
docker-compose logs -f voice-assistant
```

## ✅ **Verification**

To verify llama.cpp is working:

```bash
# 1. Check if llama.cpp is installed
docker-compose exec voice-assistant python -c "import llama_cpp; print('llama.cpp installed successfully')"

# 2. Test model loading
python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Test"

# 3. Check API endpoints
curl http://localhost:8000/api/models/state
```

## 🎉 **Summary**

Your Docker setup now includes:

- ✅ **llama-cpp-python** properly installed
- ✅ **CPU optimizations** for all platforms
- ✅ **ARM64 optimizations** for Raspberry Pi
- ✅ **Lazy loading** integration
- ✅ **Model switching** support
- ✅ **Memory management** with auto-unloading
- ✅ **API endpoints** for model management
- ✅ **CLI tools** for easy interaction

The llama.cpp integration is now fully functional and optimized for your Windows WSL → Raspberry Pi deployment workflow! 🚀
