"""
Orchestrator Package

Voice assistant orchestration - coordinates all services via the event bus.
"""

from .voice_assistant import VoiceAssistant

__all__ = ['VoiceAssistant']
