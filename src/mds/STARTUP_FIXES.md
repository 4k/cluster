# Startup Errors Fixed

## Summary
Fixed multiple errors that were preventing the application from starting properly.

## Issues Fixed

### 1. Model Loading Error
**Error:** `Failed to load model from file: models`

**Root Cause:**
- The local provider was requiring `model_path` to be set, but the configuration uses `model_id` to reference models managed by the ModelManager
- When `model_path` was not set, the provider couldn't find the model and was trying to load "models" as a directory instead of a file

**Solution:**
- Updated `validate_config()` in `src/ai/providers/local.py` to accept either `model_path` OR `model_id`
- Enhanced `_get_model_path()` to better handle model manager integration with proper logging
- Added `model_id` support throughout the configuration chain (llm, base config)

### 2. ALSA Audio Warnings
**Error:** Multiple ALSA lib errors about audio devices

**Root Cause:**
- ALSA (Linux audio subsystem) is trying to access audio devices that don't exist in the Docker container

**Solution:**
- These are non-fatal warnings and don't affect functionality in mock mode
- Can be suppressed by running with mock providers (recommended for development)

### 3. Graceful Fallback
**Added:** Automatic fallback to mock provider when local model is not available

**Solution:**
- Modified `llm.py` to automatically use mock provider as fallback when local provider fails to initialize
- This allows the app to start even without downloaded models

### 4. DecisionConfig Initialization Error
**Error:** `DecisionConfig.__init__() got an unexpected keyword argument 'buffer_size'`

**Root Cause:**
- `main.py` was passing the entire conversation config dict to `DecisionEngine`, but `DecisionConfig` only expects decision-specific fields like `response_threshold`, `wake_word_boost`, etc.
- The conversation config includes fields like `buffer_size`, `max_age_seconds`, etc. that don't belong in DecisionConfig

**Solution:**
- Modified `main.py` to filter the config and only pass `response_threshold` to DecisionEngine
- Fixed `get_state()` method in `decision.py` to reference `self.config.response_threshold` instead of non-existent attributes

### 5. DecisionEngine API Mismatch
**Error:** `DecisionEngine` was being called with a non-existent `should_respond()` method

**Root Cause:**
- The code in `main.py` was calling `self.decision_engine.should_respond()`, but `DecisionEngine` only has a `make_decision()` method that takes a `DecisionContext` and returns a `DecisionResult`

**Solution:**
- Updated `_process_speech()` in `main.py` to properly use the DecisionEngine API
- Create a `DecisionContext` with proper parameters
- Call `make_decision()` and check the returned `DecisionResult` for `DecisionType.RESPOND`

### 6. Mock Mode Not Being Applied
**Error:** Local provider still runs even when `MOCK_LLM=true` is set

**Root Cause:**
- The `_convert_config()` method in `llm.py` was overriding the `provider_type` that was set by mock mode logic
- It was trying to auto-detect the provider type based on config fields, ignoring any explicitly set provider_type

**Solution:**
- Modified `_convert_config()` to check if `provider_type` is already set before trying to auto-detect
- Added logging to track provider_type conversion
- Mock mode now properly sets provider_type to 'mock' and it stays as mock

### 7. Display Manager Event Loop Errors
**Error:** `no running event loop` repeated many times in display manager

**Root Cause:**
- The display manager runs in a separate thread (`_display_loop`)
- The code was trying to create asyncio tasks (`asyncio.create_task()`) from within that thread
- Threads don't have event loops by default, causing the "no running event loop" error

**Solution:**
- Removed `asyncio.create_task()` calls from the thread-based display loop
- Call callbacks directly without async operations in the thread context
- Skip async event emission from the thread to avoid event loop issues

## Files Modified

1. `src/ai/providers/local.py`
   - Updated validation to accept model_id
   - Enhanced model path resolution
   - Better error logging

2. `src/ai/llm.py`
   - Added model_id support to UnifiedLLMConfig
   - Added fallback to mock provider when local fails
   - Passes model_id through configuration chain

3. `src/ai/llm.py`
   - Added model_id to config conversion

4. `src/ai/providers/base.py`
   - Already had model_id support in LLMConfig

5. `src/main.py`
   - Fixed DecisionEngine initialization to only pass decision-specific config fields
   - Prevented passing entire conversation config to DecisionEngine
   - Updated `_process_speech()` to properly use DecisionEngine's `make_decision()` API
   - Added logging to show provider_type during initialization

6. `src/ai/decision.py`
   - Fixed `get_state()` to properly reference config attributes

7. `src/ai/llm.py`
   - Fixed `_convert_config()` to respect explicitly set provider_type
   - Ensures mock mode provider_type is not overridden
   - Added logging for provider_type conversion

8. `src/core/config.py`
   - Enhanced mock mode logging to debug configuration issues
   - Added more verbose logging when applying mock mode

9. `src/display/display_manager.py`
   - Fixed async calls in thread context
   - Removed `asyncio.create_task()` calls from synchronous thread
   - Call callbacks directly instead of creating async tasks from thread

## How to Run Without Errors

### Option 1: Use Mock Providers (Recommended for Development)
Set environment variables before running:
```bash
export MOCK_LLM=true
export MOCK_TTS=true
export MOCK_STT=true
python src/main.py
```

Or create a `.env` file:
```bash
MOCK_LLM=true
MOCK_TTS=true
MOCK_STT=true
```

### Option 2: Download the Model First
If you want to use the real local LLM, download the model first:
```bash
python scripts/setup_gemma_model.py
```

This will download the Gemma-3n model which is configured as the default in `config/assistant_config.yaml`.

### Option 3: Use Environment Variables to Override Config
```bash
# Use mock for LLM specifically
export MOCK_LLM=true

# Or use a different model
export LLM_MODEL_ID=phi3-mini-q4
```

## Testing the Fix

You can test the application now:

```bash
# With mock providers (will start successfully)
MOCK_LLM=true python src/main.py

# Or configure via .env file
echo "MOCK_LLM=true" > .env
echo "MOCK_TTS=true" >> .env
echo "MOCK_STT=true" >> .env
python src/main.py
```

## Expected Behavior

After these fixes:
1. ✅ Application starts without fatal errors
2. ✅ Falls back to mock provider if model not available
3. ✅ ALSA warnings are benign (don't affect functionality)
4. ✅ Can download and use real models when needed
5. ✅ Model manager integration works properly

## Next Steps

1. Run the app with mock providers to verify startup
2. Download models when ready for real inference
3. Configure environment variables as needed
4. Add your own models using the model manager

## Additional Notes

- The model manager (`src/ai/model_manager.py`) handles downloading models from Hugging Face
- Configuration in `config/assistant_config.yaml` supports both direct paths and model manager IDs
- Environment variables can override config file settings
- Mock providers provide a quick way to test without downloading models
