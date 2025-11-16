# Animation & Display Architecture

## Overview

This document describes the separation of concerns between the `AnimationEngine` and `DisplayManager` components.

**See also:** [DISPLAY_MODES_GUIDE.md](DISPLAY_MODES_GUIDE.md) for detailed information about display configuration modes.

## Architecture Principles

### AnimationEngine
**Responsibilities:**
- Decide **what** to display based on system state (idle, listening, processing, speaking, error)
- Load static images from `data/displays/eyes/` and `data/displays/mouth/`
- Render appropriate images for current emotion and mouth shape
- Pass rendered frames to DisplayManager for display

**Does NOT:**
- Handle physical display hardware
- Manage pygame windows or DSI displays
- Control display refresh rates

**Key Methods:**
- `initialize()` - Loads all static images from data/displays/
- `on_wake_word_detected()` - Transitions to listening state
- `on_speech_detected()` - Transitions to processing state  
- `on_tts_started()` - Transitions to speaking state
- `on_tts_completed()` - Returns to idle state
- `_update_eyes(emotion, gaze)` - Renders eye image and passes to display manager
- `_update_mouth(shape)` - Renders mouth image and passes to display manager

**Log Messages:**
```
animation.animation_engine - DEBUG - Animation state changed to: speaking
animation.animation_engine - DEBUG - Updated eyes: happy, center
animation.animation_engine - DEBUG - Updated mouth: smile
```

### DisplayManager
**Responsibilities:**
- Display **already-rendered frames** (from AnimationEngine)
- Support multiple display modes: dual_display, single_display, led_stripe, disabled
- Manage pygame windows during development (`development_mode: true`)
- Send frames to DSI displays on Raspberry Pi (`development_mode: false`)
- Output to LED stripes (console logging for now)
- Handle touch events on displays
- Maintain display refresh rate (FPS)

**Does NOT:**
- Decide what emotion or mouth shape to show
- Load or render images
- Change state based on system events
- Know about "eyes" or "mouth" - uses generic display indices

**Key Methods:**
- `initialize()` - Sets up displays based on mode
- `start()` - Starts the display loop (if needed)
- `display_frame(display_index, surface)` - Generic method to display a frame on any display
- `stop()` - Stops the display loop
- `cleanup()` - Cleans up resources

**Display Modes:**
- `dual_display` - Two separate displays (e.g., display 0 and 1)
- `single_display` - One display (display 0 only)
- `led_stripe` - LED strip output (extracts color, logs to console)
- `disabled` - No output (frames are discarded)

**Log Messages:**
```
display.display_manager - INFO - Initializing display manager in dual_display mode
display.display_manager - DEBUG - Received frame for display 0
display.display_manager - DEBUG - Received frame for display 1
display.display_manager - INFO - LED Stripe [0]: RGB=(255, 200, 150)  # LED mode only
```

## Data Flow

```
System Event (e.g., TTS started)
    ↓
AnimationEngine.on_tts_started()
    ↓
AnimationEngine._set_animation_state(SPEAKING)
    ↓
AnimationEngine._render_speaking()
    ↓
AnimationEngine._update_eyes(HAPPY, CENTER)
    ↓
Load static image from data/displays/eyes/happy.png
    ↓
AnimationEngine logs: "Updated eyes: happy, center on display 0"
    ↓
DisplayManager.display_frame(0, surface)  # Generic API - display index 0
    ↓
DisplayManager logs: "Received frame for display 0"
    ↓
DisplayManager displays frame based on mode:
  - dual_display: pygame window or DSI /dev/fb0
  - single_display: pygame window or DSI /dev/fb0
  - led_stripe: Extract color, log to console
  - disabled: Discard frame (no output)
```

## Static Images

The AnimationEngine loads static images from:
- `data/displays/eyes/` - Eye emotions (neutral, happy, sad, angry, surprised, etc.)
- `data/displays/mouth/` - Mouth shapes (closed, open, smile, frown, a_shape, e_shape, etc.)

All images are:
- PNG format
- Loaded at startup
- Scaled to display resolution (800x480)
- Cached in memory for fast rendering

## Configuration

### DisplayConfig (in config/assistant_config.yaml)
```yaml
display:
  mode: "dual_display"               # Display mode: dual_display, single_display, led_stripe, disabled
  display_devices: ["/dev/fb0", "/dev/fb1"]  # DSI display paths for Raspberry Pi
  resolution: [800, 480]             # Display resolution
  fps: 30                            # Display refresh rate
  development_mode: true             # true = pygame window, false = DSI displays
  
  # LED stripe settings (for led_stripe mode)
  led_count: 60
  led_pin: 18
  
  # Touch/interaction
  touch_enabled: false               # Enable touch event handling
  calibration_file: null             # Optional touch calibration file
```

**Display Modes:**
- `dual_display`: Two separate displays (default for facial animations)
- `single_display`: One display
- `led_stripe`: LED strip output (console logging only for now)
- `disabled`: No display output

### AnimationConfig (passed to AnimationEngine)
```python
AnimationConfig(
    displays_dir="data/displays",
    eyes_dir="data/displays/eyes",
    mouth_dir="data/displays/mouth",
    resolution=(800, 480),
    
    # Display mapping (which physical display shows which content)
    eyes_display_index=0,      # Display 0 shows eyes
    mouth_display_index=1,     # Display 1 shows mouth
    
    # Emotion settings
    default_emotion=EmotionType.NEUTRAL,
    processing_emotion=EmotionType.FOCUSED,
    speaking_emotion=EmotionType.HAPPY,
    error_emotion=EmotionType.CONFUSED
)
```

**Display Index Configuration:**
- For `dual_display` mode: Use indices 0 and 1 (eyes=0, mouth=1)
- For `single_display` mode: Use index 0 for both (eyes=0, mouth=0)
- For `led_stripe` mode: Indices still used for color mapping
- For `disabled` mode: Indices don't matter (frames discarded)

## Development vs Production

### Development Mode (`development_mode: true`)
- Uses pygame windows for display
- Shows visual feedback on development machine
- Single window limitation (pygame can only create one display window)
- Useful for testing without Raspberry Pi hardware

### Production Mode (`development_mode: false`)
- Sends frames directly to DSI displays via `/dev/fb0` and `/dev/fb1`
- Runs on Raspberry Pi with dual DSI displays
- No pygame window overhead

## Future Enhancements

### AnimationEngine
- [ ] Implement gaze direction transformations
- [ ] Add eye blinking animation
- [ ] Implement smooth transitions between emotions
- [ ] Add lip-sync based on phonemes/visemes
- [ ] Support animated GIF or sprite sheets

### DisplayManager
- [ ] Implement actual DSI display support for Raspberry Pi
- [ ] Support dual display windows in development mode
- [ ] Add touch calibration support
- [ ] Implement display mirroring/recording for debugging

## Migration Notes

### Breaking Changes from Previous Architecture

1. **DisplayManager API changes:**
   - ❌ `display_eyes_frame(surface)` → ✅ `display_frame(0, surface)`
   - ❌ `display_mouth_frame(surface)` → ✅ `display_frame(1, surface)`
   - ❌ `update_eyes(emotion, gaze)` → Use AnimationEngine instead
   - ❌ `update_mouth(shape, emotion)` → Use AnimationEngine instead
   - ❌ `set_speaking(bool)` → Use AnimationEngine state methods
   - ❌ `set_listening(bool)` → Use AnimationEngine state methods
   - ❌ `set_processing(bool)` → Use AnimationEngine state methods

2. **DisplayConfig changes:**
   - ✅ `mode` added (required): "dual_display", "single_display", "led_stripe", or "disabled"
   - ✅ `display_devices` (list): Replaces `eyes_display` and `mouth_display`
   - ❌ `static_mode` removed
   - ✅ `led_count` and `led_pin` added for LED stripe mode

3. **AnimationConfig changes:**
   - ✅ `eyes_display_index` added: Which display shows eyes (default: 0)
   - ✅ `mouth_display_index` added: Which display shows mouth (default: 1)

4. **Component initialization order:**
   ```python
   # Correct order:
   display_manager = DisplayManager(display_config)
   await display_manager.initialize()
   
   animation_engine = await get_animation_engine(display_manager, animation_config)
   # AnimationEngine initializes and loads images automatically
   ```

5. **Old code migration:**
   ```python
   # OLD:
   display_manager.display_eyes_frame(eyes_surface)
   display_manager.display_mouth_frame(mouth_surface)
   
   # NEW:
   display_manager.display_frame(0, eyes_surface)    # Display 0
   display_manager.display_frame(1, mouth_surface)   # Display 1
   ```

## Troubleshooting

### "No eye image found for emotion: X"
- Check that `data/displays/eyes/X.png` exists
- Ensure image file name matches emotion value (e.g., "happy.png" for EmotionType.HAPPY)

### "No mouth image found for shape: X"
- Check that `data/displays/mouth/X.png` exists  
- Ensure image file name matches shape value (e.g., "smile.png" for MouthShape.SMILE)

### "Display manager logs emotion updates"
- This is incorrect behavior - only AnimationEngine should log emotion updates
- Check that DisplayManager is using the new `display_eyes_frame()` API
- Verify no code is calling old DisplayManager state methods

### "Pygame window not showing"
- Ensure `development_mode: true` in config
- Check pygame is installed: `pip install pygame`
- Verify display initialization completed without errors

