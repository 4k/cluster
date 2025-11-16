# Display Modes Configuration Guide

## Overview

The Display Manager now supports multiple display configurations to accommodate different hardware setups:

1. **Dual Display** - Two separate displays (e.g., split facial animations)
2. **Single Display** - One combined display  
3. **LED Stripe** - LED stripe output (console logging for now)
4. **Disabled** - No display output

## Display Modes

### 1. Dual Display Mode (`dual_display`)

**Use Case:** Two separate DSI displays for split animations (e.g., one for eyes, one for mouth)

**Configuration:**
```yaml
display:
  mode: "dual_display"
  display_devices: ["/dev/fb0", "/dev/fb1"]
  resolution: [800, 480]
  fps: 30
  development_mode: true
  touch_enabled: false
```

**Behavior:**
- **Development Mode:** Shows one pygame window (display 0 only, due to pygame limitation)
- **Production Mode:** Sends frames to `/dev/fb0` and `/dev/fb1` on Raspberry Pi
- **Display Mapping:**
  - Display 0: Eyes animations
  - Display 1: Mouth animations

**Log Output:**
```
display.display_manager - INFO - Initializing display manager in dual_display mode
animation.animation_engine - DEBUG - Updated eyes: happy, center on display 0
display.display_manager - DEBUG - Received frame for display 0
animation.animation_engine - DEBUG - Updated mouth: smile on display 1
display.display_manager - DEBUG - Received frame for display 1
```

---

### 2. Single Display Mode (`single_display`)

**Use Case:** One display showing combined animations or single screen output

**Configuration:**
```yaml
display:
  mode: "single_display"
  display_devices: ["/dev/fb0"]
  resolution: [800, 480]
  fps: 30
  development_mode: true
```

**Behavior:**
- **Development Mode:** Shows one pygame window
- **Production Mode:** Sends frames to `/dev/fb0` on Raspberry Pi
- **Display Mapping:**
  - Display 0: Combined animations (can show either eyes or mouth, or both combined)

**Log Output:**
```
display.display_manager - INFO - Initializing display manager in single_display mode
animation.animation_engine - DEBUG - Updated eyes: happy, center on display 0
display.display_manager - DEBUG - Received frame for display 0
```

**Note:** For single display with combined animations, you may need to update AnimationEngine to render a combined frame with both eyes and mouth.

---

### 3. LED Stripe Mode (`led_stripe`)

**Use Case:** LED stripe output for ambient lighting or simple status indicators

**Configuration:**
```yaml
display:
  mode: "led_stripe"
  led_count: 60
  led_pin: 18
  fps: 30
  development_mode: true
```

**Behavior:**
- **Current Implementation:** Extracts average color from animation frames and logs to console
- **Future Implementation:** Will control actual LED stripe hardware
- **No pygame initialization** - doesn't create display windows or surfaces

**Log Output:**
```
display.display_manager - INFO - Initializing display manager in led_stripe mode
display.display_manager - INFO - LED stripe mode initialized (console output only)
animation.animation_engine - DEBUG - Updated eyes: happy, center on display 0
display.display_manager - INFO - LED Stripe [0]: RGB=(255, 200, 150)
animation.animation_engine - DEBUG - Updated mouth: smile on display 1
display.display_manager - INFO - LED Stripe [1]: RGB=(255, 220, 180)
```

**Future Hardware Support:**
- Will integrate with LED stripe libraries (e.g., rpi_ws281x, NeoPixel)
- Average color will be displayed across all LEDs
- Could support gradient effects or more complex patterns

---

### 4. Disabled Mode (`disabled`)

**Use Case:** Completely disable display output (for headless operation or testing)

**Configuration:**
```yaml
display:
  mode: "disabled"
```

**Behavior:**
- **No display output** at all
- **No pygame initialization**
- **Minimal overhead** - display manager does nothing
- AnimationEngine still processes state changes but frames are discarded

**Log Output:**
```
display.display_manager - INFO - Initializing display manager in disabled mode
display.display_manager - INFO - Display manager initialized (disabled mode - no output)
animation.animation_engine - DEBUG - Updated eyes: happy, center on display 0
(no display manager output - frame discarded)
```

---

## Generic Display API

### DisplayManager Methods

The DisplayManager now uses a generic, index-based API instead of named displays:

**Old API (deprecated):**
```python
display_manager.display_eyes_frame(surface)    # ❌ No longer exists
display_manager.display_mouth_frame(surface)   # ❌ No longer exists
```

**New API:**
```python
display_manager.display_frame(display_index: int, frame: pygame.Surface)
```

**Examples:**
```python
# Display eyes on display 0
display_manager.display_frame(0, eyes_surface)

# Display mouth on display 1
display_manager.display_frame(1, mouth_surface)

# Single display mode - everything on display 0
display_manager.display_frame(0, combined_surface)
```

### AnimationEngine Display Mapping

The AnimationEngine configuration now includes display index mappings:

```python
AnimationConfig(
    eyes_display_index=0,   # Send eyes to display 0
    mouth_display_index=1   # Send mouth to display 1
)
```

**For single display mode:**
```python
AnimationConfig(
    eyes_display_index=0,   # Send eyes to display 0
    mouth_display_index=0   # Send mouth to display 0 (same display)
)
```

---

## Configuration Examples

### Example 1: Development Testing (Dual Display)
```yaml
display:
  mode: "dual_display"
  display_devices: ["/dev/fb0", "/dev/fb1"]
  resolution: [800, 480]
  fps: 30
  development_mode: true    # Uses pygame window
  touch_enabled: false
```

### Example 2: Raspberry Pi Production (Dual Display)
```yaml
display:
  mode: "dual_display"
  display_devices: ["/dev/fb0", "/dev/fb1"]
  resolution: [800, 480]
  fps: 30
  development_mode: false   # Uses DSI displays
  touch_enabled: true
  calibration_file: "/etc/touchscreen.conf"
```

### Example 3: Single Display with LED Effects
```yaml
display:
  mode: "single_display"
  display_devices: ["/dev/fb0"]
  resolution: [1024, 600]
  fps: 30
  development_mode: false
```

### Example 4: LED Stripe Only (Ambient Lighting)
```yaml
display:
  mode: "led_stripe"
  led_count: 60
  led_pin: 18
  fps: 15  # Lower FPS for LED updates
```

### Example 5: Headless/Testing (No Display)
```yaml
display:
  mode: "disabled"
```

---

## Migration from Old Configuration

### Old Configuration
```yaml
display:
  eyes_display: "/dev/fb0"
  mouth_display: "/dev/fb1"
  resolution: [800, 480]
  fps: 30
  static_mode: true
  development_mode: true
  touch_enabled: false
```

### New Configuration
```yaml
display:
  mode: "dual_display"
  display_devices: ["/dev/fb0", "/dev/fb1"]
  resolution: [800, 480]
  fps: 30
  development_mode: true
  touch_enabled: false
```

**Key Changes:**
- ✅ `mode` added (required): Specifies display mode
- ✅ `display_devices` (list): Replaces separate eyes_display/mouth_display
- ❌ `static_mode` removed: No longer needed (handled by mode selection)
- ✅ Added `led_count` and `led_pin` for LED stripe mode

---

## Display State Information

Check display manager state with `get_state()`:

```python
state = display_manager.get_state()
# Returns:
{
    "is_initialized": True,
    "is_running": True,
    "mode": "dual_display",
    "num_displays": 2,
    "development_mode": True,
    "resolution": (800, 480),
    "fps": 30,
    "active_displays": [0, 1]  # Displays currently receiving frames
}
```

---

## Troubleshooting

### "pygame can only have one display window"
- **Issue:** Dual display mode in development only shows display 0
- **Solution:** This is a pygame limitation. Use production mode on Raspberry Pi for true dual displays, or run two separate processes

### "Invalid display index: X"
- **Issue:** AnimationEngine trying to send frames to non-existent display
- **Solution:** Check that `eyes_display_index` and `mouth_display_index` match your display mode (0 for single, 0-1 for dual)

### "LED Stripe not showing colors"
- **Current:** LED stripe mode only logs to console (placeholder)
- **Future:** Will be implemented with actual LED hardware support

### Display shows blank/black screen
- Check that frames are being received: Look for "Received frame for display X" logs
- Verify display resolution matches your hardware
- In development mode, ensure pygame window is visible

---

## Performance Considerations

### Display Mode Performance

| Mode | CPU Usage | Memory | Best For |
|------|-----------|---------|----------|
| Dual Display | Medium | High | Full facial animations |
| Single Display | Low | Medium | Simple animations, single screen |
| LED Stripe | Very Low | Very Low | Ambient effects, status indication |
| Disabled | Minimal | Minimal | Headless operation, testing |

### FPS Recommendations

- **Dual Display:** 30 FPS (smooth animations)
- **Single Display:** 30-60 FPS
- **LED Stripe:** 10-20 FPS (LEDs don't need high refresh rate)
- **Disabled:** N/A

---

## Future Enhancements

### Planned Features
- [ ] Actual LED stripe hardware integration (WS2812B/NeoPixel)
- [ ] Multi-window support for development dual display
- [ ] Display rotation and orientation settings
- [ ] Hardware acceleration for DSI displays
- [ ] Custom display layouts (grid, overlay, split-screen)
- [ ] Display-specific scaling and transformations
- [ ] Hot-swapping display modes at runtime

### LED Stripe Features (Future)
- [ ] Color patterns and animations
- [ ] Audio-reactive effects
- [ ] Gradient mapping from animation colors
- [ ] Per-LED control with custom effects
- [ ] Integration with music/audio spectrum

