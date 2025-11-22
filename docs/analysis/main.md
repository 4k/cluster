# main.py - Application Entry Point Analysis

## Overview

`main.py` serves as the primary entry point for the Cluster Voice Assistant application. It implements an event-driven architecture that orchestrates core services and pluggable features.

## File Location
`/home/user/cluster/main.py`

## Dependencies
- `asyncio` - Async/await support
- `argparse` - Command-line argument parsing
- `signal` - Signal handling for graceful shutdown
- `dotenv` - Environment variable loading

## Classes

### VoiceAssistant

**Purpose**: Main orchestrator class that manages the lifecycle of all services and features.

**Attributes**:
- `enabled_features: List[str]` - List of feature names to enable
- `event_bus: EventBus` - Central event bus instance
- `feature_loader: FeatureLoader` - Manages pluggable features
- `stt_service: STTService` - Speech-to-text service
- `llm_service: LLMService` - Language model service
- `tts_service: TTSService` - Text-to-speech service
- `is_running: bool` - Application running state

**Methods**:

| Method | Purpose | Async |
|--------|---------|-------|
| `initialize()` | Initialize event bus and all services | Yes |
| `_init_services()` | Initialize STT, LLM, TTS services | Yes |
| `_init_features()` | Load and initialize enabled features | Yes |
| `start()` | Start STT listening and features | Yes |
| `stop()` | Graceful shutdown of all components | Yes |
| `run()` | Main event loop until interrupted | Yes |

## Functions

### setup_logging(level: str)
Configures application logging with both console and file handlers.

### list_features()
Displays available pluggable features to the user.

### async_main(features: List[str])
Async entry point that creates and runs the VoiceAssistant with signal handlers.

### main()
CLI entry point with argument parsing.

## Architecture Flow

```
main()
  -> async_main(features)
     -> VoiceAssistant.run()
        -> initialize()
           -> EventBus.get_instance()
           -> _init_services()
              -> STTService.initialize()
              -> LLMService.initialize()
              -> TTSService.initialize()
           -> _init_features()
              -> FeatureLoader.load_features()
              -> FeatureLoader.initialize_all()
        -> start()
           -> STTService.start() [in thread]
           -> FeatureLoader.start_all()
        -> while is_running: sleep(1)
        -> stop()
```

## Command-Line Interface

```bash
python main.py                          # Run with no features
python main.py --features display       # Enable display feature
python main.py --list-features          # Show available features
python main.py --log-level DEBUG        # Set logging level
```

## Improvements Suggested

### 1. Service Health Monitoring
Add periodic health checks for services:
```python
async def _health_check_loop(self) -> None:
    while self.is_running:
        await asyncio.sleep(30)
        for service in [self.stt_service, self.llm_service, self.tts_service]:
            if not await service.health_check():
                logger.warning(f"Service {service.__class__.__name__} unhealthy")
```

### 2. Configuration Validation at Startup
Add early validation before service initialization:
```python
async def _validate_configuration(self) -> bool:
    """Validate all required configuration before starting."""
    from src.core.config import load_config
    config = load_config()
    return config is not None
```

### 3. Graceful Degradation
Allow the system to continue with partial services:
```python
async def _init_services(self) -> None:
    for service_class, name in [(STTService, 'stt'), (LLMService, 'llm'), (TTSService, 'tts')]:
        try:
            service = service_class()
            await service.initialize()
            setattr(self, f'{name}_service', service)
        except Exception as e:
            logger.error(f"Failed to init {name}: {e}")
            setattr(self, f'{name}_service', None)
```

### 4. Restart Capability
Add ability to restart individual services without full restart.

### 5. Metrics Collection
Add startup timing metrics for performance monitoring.

## Dependencies Graph

```
main.py
├── src.core.event_bus (EventBus)
├── src.services (STTService, LLMService, TTSService)
├── src.features (FeatureLoader, get_available_features)
└── dotenv (load_dotenv)
```

## Thread Safety Considerations

- Event bus is accessed from multiple async contexts
- STT service runs in a separate thread via `asyncio.to_thread()`
- Signal handlers create async tasks for shutdown

## Error Handling

- Catches `KeyboardInterrupt` for graceful shutdown
- Catches generic `Exception` for fatal errors with traceback
- Uses `asyncio.CancelledError` for task cancellation during shutdown
