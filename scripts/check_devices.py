#!/usr/bin/env python3
"""
Device Access Verification Script
Tests if Windows physical devices are properly proxied to the Docker container.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_device_access():
    """Check if devices are accessible in the container."""
    print("🔍 Checking Windows Device Access in Docker Container")
    print("=" * 60)
    
    # Check if running in Docker
    if not os.path.exists('/.dockerenv'):
        print("❌ This script should be run inside a Docker container")
        return False
    
    print("✅ Running inside Docker container")
    
    # Check audio devices
    print("\n🔊 Checking Audio Devices...")
    try:
        result = subprocess.run(['ls', '-la', '/dev/snd/'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Audio devices accessible:")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ Audio devices not accessible")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
    
    # Check camera devices
    print("\n📷 Checking Camera Devices...")
    try:
        result = subprocess.run(['ls', '-la', '/dev/video*'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Camera devices accessible:")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ Camera devices not accessible")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error checking camera devices: {e}")
    
    # Check display devices
    print("\n🖥️ Checking Display Devices...")
    try:
        result = subprocess.run(['ls', '-la', '/dev/fb*'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Display devices accessible:")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ Display devices not accessible")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error checking display devices: {e}")
    
    # Check X11 display
    print("\n🖼️ Checking X11 Display...")
    display = os.environ.get('DISPLAY', 'Not set')
    print(f"   DISPLAY: {display}")
    
    if os.path.exists('/tmp/.X11-unix'):
        print("✅ X11 socket accessible")
    else:
        print("❌ X11 socket not accessible")
    
    # Check environment variables
    print("\n⚙️ Checking Environment Variables...")
    env_vars = [
        'AUDIO_DEVICE', 'CAMERA_ENABLED', 'CAMERA_DEVICE',
        'DISPLAY', 'DISPLAY_STATIC_MODE', 'MOCK_MODE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    print("\n" + "=" * 60)
    print("🎯 Device Access Check Complete!")
    print("\n📋 Summary:")
    print("   - Audio devices: /dev/snd/*")
    print("   - Camera devices: /dev/video*")
    print("   - Display devices: /dev/fb*")
    print("   - X11 display: /tmp/.X11-unix")
    print("\n💡 If devices are not accessible:")
    print("   1. Ensure Docker is running with --privileged")
    print("   2. Check WSL2 audio/camera support")
    print("   3. Verify device permissions")
    print("   4. Use mock mode for testing: MOCK_MODE=true")

if __name__ == "__main__":
    check_device_access()


