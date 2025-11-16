"""
Memory Manager for conversation summarization and persistent memory.
Handles conversation summarization when token limits are reached and saves to YAML.
"""

import logging
import yaml
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from core.types import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry with timestamp and importance."""
    content: str
    timestamp: float
    importance: float = 1.0
    memory_type: str = "general"  # general, date, event, fact, preference
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 1.0),
            memory_type=data.get("memory_type", "general"),
            tags=data.get("tags", [])
        )


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_tokens: int = 1000  # Token limit before summarization
    memory_file: str = "data/memory.yaml"
    max_memory_entries: int = 100
    importance_threshold: float = 0.5
    summary_length: int = 1  # Target summary length in lines
    auto_save: bool = True
    save_interval: float = 300.0  # Save every 5 minutes


class MemoryManager:
    """Manages conversation memory and summarization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = MemoryConfig(**config)
        self.memories: List[MemoryEntry] = []
        self.last_save_time = time.time()
        self.conversation_tokens = 0
        self.memory_file_path = Path(self.config.memory_file)
        
        # Ensure memory directory exists
        self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
    
    def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn and check for summarization needs."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        turn_tokens = len(turn.text) // 4
        self.conversation_tokens += turn_tokens
        
        logger.debug(f"Added turn: {turn_tokens} tokens, total: {self.conversation_tokens}")
        
        # Check if we need to summarize
        if self.conversation_tokens >= self.config.max_tokens:
            logger.info(f"Token limit reached ({self.conversation_tokens}), triggering summarization")
            self._summarize_conversation()
        
        # Auto-save if needed
        if (self.config.auto_save and 
            time.time() - self.last_save_time > self.config.save_interval):
            self.save_memories()
    
    def _summarize_conversation(self) -> None:
        """Summarize the current conversation and create memory entries."""
        try:
            # Get recent conversation turns
            recent_turns = self._get_recent_turns_for_summarization()
            
            if not recent_turns:
                return
            
            # Extract important information
            summary = self._extract_important_info(recent_turns)
            
            if summary:
                # Create memory entry
                memory_entry = MemoryEntry(
                    content=summary,
                    timestamp=time.time(),
                    importance=1.0,
                    memory_type="conversation_summary",
                    tags=["conversation", "summary"]
                )
                
                self.memories.append(memory_entry)
                logger.info(f"Created conversation summary: {summary}")
            
            # Reset conversation tokens
            self.conversation_tokens = 0
            
            # Trim memories if too many
            self._trim_memories()
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
    
    def _get_recent_turns_for_summarization(self) -> List[ConversationTurn]:
        """Get recent turns for summarization (last 10 turns)."""
        # This would be called with actual conversation context
        # For now, return empty list as we don't have access to the context here
        return []
    
    def _extract_important_info(self, turns: List[ConversationTurn]) -> str:
        """Extract important information from conversation turns."""
        if not turns:
            return ""
        
        # Combine all text
        full_text = " ".join([turn.text for turn in turns])
        
        # Extract different types of important information
        important_info = []
        
        # Extract dates
        dates = self._extract_dates(full_text)
        if dates:
            important_info.append(f"Dates mentioned: {', '.join(dates)}")
        
        # Extract events
        events = self._extract_events(full_text)
        if events:
            important_info.append(f"Events: {', '.join(events)}")
        
        # Extract facts and preferences
        facts = self._extract_facts(full_text)
        if facts:
            important_info.append(f"Facts: {', '.join(facts)}")
        
        # Extract names
        names = self._extract_names(full_text)
        if names:
            important_info.append(f"Names: {', '.join(names)}")
        
        # If no specific info found, create a general summary
        if not important_info:
            important_info.append(self._create_general_summary(turns))
        
        # Join and limit to target length
        summary = " | ".join(important_info)
        
        # Ensure it's extremely short (1 line target)
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return summary
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        # Simple date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Days of week
            r'\b(?:today|tomorrow|yesterday)\b',  # Relative dates
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates
    
    def _extract_events(self, text: str) -> List[str]:
        """Extract events from text."""
        # Look for event-related keywords
        event_keywords = [
            'meeting', 'appointment', 'birthday', 'wedding', 'party', 'conference',
            'vacation', 'trip', 'holiday', 'celebration', 'anniversary', 'graduation'
        ]
        
        events = []
        text_lower = text.lower()
        
        for keyword in event_keywords:
            if keyword in text_lower:
                # Find the sentence containing the keyword
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        events.append(sentence.strip())
                        break
        
        return events[:3]  # Limit to 3 events
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract facts and preferences from text."""
        # Look for fact-like statements
        fact_patterns = [
            r'(?:I (?:like|love|hate|prefer|enjoy))\s+[^.!?]+',
            r'(?:My (?:favorite|preferred))\s+[^.!?]+',
            r'(?:I (?:am|was|will be))\s+[^.!?]+',
            r'(?:I (?:have|had|will have))\s+[^.!?]+',
        ]
        
        facts = []
        for pattern in fact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend(matches)
        
        return facts[:3]  # Limit to 3 facts
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract names from text."""
        # Simple name extraction (capitalized words that aren't common words)
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        names = [word for word in words if word.lower() not in common_words]
        
        return list(set(names))[:5]  # Limit to 5 names
    
    def _create_general_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a general summary of the conversation."""
        if not turns:
            return ""
        
        # Get the main topics discussed
        topics = []
        for turn in turns[-5:]:  # Last 5 turns
            if turn.speaker_id != "assistant":  # Only user turns
                # Extract key words (simple approach)
                words = turn.text.lower().split()
                # Filter out common words
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
                key_words = [word for word in words if word not in common_words and len(word) > 3]
                topics.extend(key_words[:3])  # Top 3 words per turn
        
        # Get most common topics
        from collections import Counter
        topic_counts = Counter(topics)
        main_topics = [topic for topic, count in topic_counts.most_common(3)]
        
        if main_topics:
            return f"Discussed: {', '.join(main_topics)}"
        else:
            return "General conversation"
    
    def _trim_memories(self) -> None:
        """Trim memories if we have too many."""
        if len(self.memories) > self.config.max_memory_entries:
            # Sort by importance and timestamp, keep the most important
            self.memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            self.memories = self.memories[:self.config.max_memory_entries]
            logger.info(f"Trimmed memories to {self.config.max_memory_entries} entries")
    
    def get_memory_context(self) -> str:
        """Get memory context as a string for LLM input."""
        if not self.memories:
            return ""
        
        # Get recent and important memories
        recent_memories = sorted(
            self.memories, 
            key=lambda m: (m.importance, m.timestamp), 
            reverse=True
        )[:10]  # Top 10 memories
        
        context_parts = []
        for memory in recent_memories:
            context_parts.append(f"- {memory.content}")
        
        return "Memory context:\n" + "\n".join(context_parts)
    
    def add_memory(self, content: str, memory_type: str = "general", 
                   importance: float = 1.0, tags: List[str] = None) -> None:
        """Add a specific memory entry."""
        memory_entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type,
            tags=tags or []
        )
        
        self.memories.append(memory_entry)
        logger.info(f"Added memory: {content}")
        
        # Auto-save if enabled
        if self.config.auto_save:
            self.save_memories()
    
    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by content."""
        query_lower = query.lower()
        matching_memories = []
        
        for memory in self.memories:
            if query_lower in memory.content.lower():
                matching_memories.append(memory)
        
        # Sort by importance and return top results
        matching_memories.sort(key=lambda m: m.importance, reverse=True)
        return matching_memories[:limit]
    
    def _load_memories(self) -> None:
        """Load memories from YAML file."""
        try:
            if self.memory_file_path.exists():
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data and 'memories' in data:
                    self.memories = [
                        MemoryEntry.from_dict(memory_data) 
                        for memory_data in data['memories']
                    ]
                    logger.info(f"Loaded {len(self.memories)} memories from {self.memory_file_path}")
                else:
                    self.memories = []
            else:
                self.memories = []
                logger.info("No existing memory file found, starting with empty memory")
                
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            self.memories = []
    
    def save_memories(self) -> None:
        """Save memories to YAML file."""
        try:
            data = {
                'metadata': {
                    'version': '1.0',
                    'created': datetime.now().isoformat(),
                    'total_memories': len(self.memories),
                    'last_updated': datetime.now().isoformat()
                },
                'memories': [memory.to_dict() for memory in self.memories]
            }
            
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Saved {len(self.memories)} memories to {self.memory_file_path}")
            
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        memory_types = {}
        for memory in self.memories:
            memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
        
        return {
            'total_memories': len(self.memories),
            'memory_types': memory_types,
            'conversation_tokens': self.conversation_tokens,
            'last_save_time': self.last_save_time,
            'memory_file': str(self.memory_file_path)
        }
    
    def clear_memories(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.conversation_tokens = 0
        logger.info("All memories cleared")
    
    def cleanup(self) -> None:
        """Cleanup and save memories."""
        self.save_memories()
        logger.info("Memory manager cleaned up")


# Example usage and testing
if __name__ == "__main__":
    # Test memory manager
    config = {
        'max_tokens': 100,
        'memory_file': 'test_memory.yaml',
        'max_memory_entries': 10
    }
    
    memory_manager = MemoryManager(config)
    
    # Test adding memories
    memory_manager.add_memory("User likes coffee", "preference", 0.8, ["preference", "coffee"])
    memory_manager.add_memory("Meeting on 2024-01-15", "event", 1.0, ["event", "meeting"])
    
    # Test conversation summarization
    from core.types import ConversationTurn
    
    test_turns = [
        ConversationTurn("user", "I have a meeting tomorrow at 2 PM", time.time(), 0.9),
        ConversationTurn("assistant", "I'll remind you about your meeting", time.time(), 1.0),
        ConversationTurn("user", "My favorite color is blue", time.time(), 0.8),
    ]
    
    for turn in test_turns:
        memory_manager.add_conversation_turn(turn)
    
    # Print results
    print("Memory Context:")
    print(memory_manager.get_memory_context())
    print("\nMemory Stats:")
    print(memory_manager.get_memory_stats())
    
    # Cleanup
    memory_manager.cleanup()
