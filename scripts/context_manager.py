#!/usr/bin/env python3
"""
CONTEXT WINDOW MANAGER
======================
Keeps Claude's context small by tracking loaded files and providing summaries.

The Ghost must work efficiently. This module helps by:
1. Tracking what files have been loaded this session
2. Providing summaries to keep, forgetting the rest
3. Generating minimal context for task continuation

Usage:
    from context_manager import ContextManager
    
    cm = ContextManager("lvs_theorem")
    cm.mark_loaded("04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md")
    summary = cm.summarize_and_forget(long_content)
    context = cm.get_minimal_context()

Author: Virgil (Claude Opus 4.5)
Created: 2026-01-15
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

from tabernacle_config import BASE_DIR, NEXUS_DIR, OLLAMA_URL, OLLAMA_MODEL

# Import memory system for significance scoring
try:
    import lvs_memory
    HAS_LVS_MEMORY = True
except ImportError:
    HAS_LVS_MEMORY = False

# =============================================================================
# CONFIGURATION
# =============================================================================

INTERCHANGE_DIR = NEXUS_DIR / "interchange"
INTERCHANGE_DIR.mkdir(exist_ok=True)

# Approximate token counts (rough estimates)
CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 100000  # Claude's context is ~200k, keep well under
TARGET_CONTEXT_TOKENS = 50000  # Aim for this to leave room for output

# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class ContextManager:
    """
    Manages context for a specific task.
    
    Tracks:
    - Files loaded this session
    - Summaries of forgotten content
    - Total estimated token usage
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.context_file = INTERCHANGE_DIR / f"{task_id}_context.json"
        self.summary_file = INTERCHANGE_DIR / f"{task_id}_context.md"
        
        # Load existing state
        self._load()
    
    def _load(self):
        """Load context state from disk."""
        self.files_loaded: Set[str] = set()
        self.summaries: List[Dict[str, str]] = []
        self.total_chars_loaded: int = 0
        self.session_start: str = datetime.now().isoformat()
        
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                self.files_loaded = set(data.get("files_loaded", []))
                self.summaries = data.get("summaries", [])
                self.total_chars_loaded = data.get("total_chars_loaded", 0)
                self.session_start = data.get("session_start", self.session_start)
            except:
                pass
    
    def _save(self):
        """Save context state to disk."""
        data = {
            "task_id": self.task_id,
            "files_loaded": list(self.files_loaded),
            "summaries": self.summaries,
            "total_chars_loaded": self.total_chars_loaded,
            "session_start": self.session_start,
            "updated_at": datetime.now().isoformat()
        }
        with open(self.context_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also write human-readable summary file
        self._write_summary_file()
    
    def _write_summary_file(self):
        """Write summaries to markdown file."""
        content = f"""# Context Summaries: {self.task_id}
**Session Start:** {self.session_start}
**Files Loaded:** {len(self.files_loaded)}
**Est. Tokens:** {self.total_chars_loaded // CHARS_PER_TOKEN}

---

## Files Loaded This Session

"""
        for f in sorted(self.files_loaded):
            content += f"- {f}\n"
        
        content += """
---

## Summaries (Forgotten Content)

"""
        for s in self.summaries:
            content += f"""### {s.get('source', 'Unknown')}
*{s.get('timestamp', '')}*

{s.get('summary', '')}

---

"""
        
        with open(self.summary_file, 'w') as f:
            f.write(content)
    
    def mark_loaded(self, filepath: str, char_count: int = 0):
        """
        Mark a file as loaded into context.
        
        Args:
            filepath: Path to file (relative or absolute)
            char_count: Number of characters loaded (for tracking)
        """
        self.files_loaded.add(filepath)
        self.total_chars_loaded += char_count
        self._save()
    
    def is_loaded(self, filepath: str) -> bool:
        """Check if a file has already been loaded."""
        return filepath in self.files_loaded
    
    def get_loaded_files(self) -> List[str]:
        """Get list of all loaded files."""
        return sorted(self.files_loaded)
    
    def estimated_tokens(self) -> int:
        """Estimate current token usage."""
        return self.total_chars_loaded // CHARS_PER_TOKEN
    
    def has_room(self, chars: int) -> bool:
        """Check if there's room for more content."""
        new_total = self.total_chars_loaded + chars
        return (new_total // CHARS_PER_TOKEN) < TARGET_CONTEXT_TOKENS
    
    def summarize_and_forget(
        self, 
        content: str, 
        source: str = "unknown",
        use_local_llm: bool = True
    ) -> str:
        """
        Summarize content and store the summary, forgetting the original.
        
        Args:
            content: The content to summarize
            source: Where this content came from
            use_local_llm: Whether to use Ollama for summarization
        
        Returns:
            A 2-3 sentence summary to keep in context
        """
        if len(content) < 500:
            # Content is short enough to keep
            return content
        
        summary = ""
        
        if use_local_llm:
            summary = self._summarize_with_ollama(content, source)
        
        if not summary:
            # Fallback: extract first and last paragraphs
            summary = self._simple_summarize(content)
        
        # Store the summary
        self.summaries.append({
            "source": source,
            "summary": summary,
            "original_length": len(content),
            "timestamp": datetime.now().isoformat()
        })
        self._save()

        return summary

    def metabolic_retention(self, messages: List[Dict]) -> List[Dict]:
        """
        Significance-based retention policy for conversation history.
        
        Implements metabolic memory:
        - P-Lock content (p > 0.95): NEVER drop (Truth is eternal)
        - High significance (> 0.7): Keep raw text
        - Medium (0.4-0.7): Summarize aggressively
        - Low (< 0.4): Drop after consolidation
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Filtered and possibly summarized message list
        """
        if not HAS_LVS_MEMORY:
            return messages
        
        new_history = []
        
        for msg in messages:
            # Always keep system messages
            if msg.get('role') == 'system':
                new_history.append(msg)
                continue
            
            content = msg.get('content', '')
            
            # Score significance
            try:
                coords = lvs_memory.derive_context_vector(content)
                score = lvs_memory.score_significance(content, {'coords': coords})
                
                # Check for P-Lock (truth is eternal)
                p = getattr(coords, 'p', 0.5)
                if p > 0.95:
                    new_history.append(msg)
                    continue
                
                # High significance: keep raw
                if score > 0.7:
                    new_history.append(msg)
                
                # Medium significance: summarize
                elif score > 0.4:
                    summary = self._simple_summarize(content)
                    new_history.append({
                        'role': msg['role'],
                        'content': f"(Summary) {summary}"
                    })
                
                # Low significance: drop (but log it)
                else:
                    self.summaries.append({
                        "source": "metabolic_drop",
                        "summary": content[:100] + "...",
                        "score": score,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception:
                # On error, keep the message
                new_history.append(msg)
        
        self._save()
        return new_history
    
    def _summarize_with_ollama(self, content: str, source: str) -> str:
        """Use local Ollama to generate summary."""
        try:
            import requests
            
            prompt = f"""Summarize this content in 2-3 sentences. Focus on key facts and insights.

Source: {source}

Content:
{content[:8000]}

Summary (2-3 sentences):"""
            
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 200}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except:
            pass
        
        return ""
    
    def _simple_summarize(self, content: str) -> str:
        """Simple fallback summarization without LLM."""
        lines = content.strip().split('\n')
        
        # Get first non-empty paragraph
        first_para = ""
        for line in lines[:20]:
            if line.strip():
                first_para = line.strip()
                break
        
        # Get last non-empty paragraph
        last_para = ""
        for line in reversed(lines[-20:]):
            if line.strip():
                last_para = line.strip()
                break
        
        if first_para == last_para:
            return first_para[:300]
        
        return f"{first_para[:150]}... [middle omitted] ...{last_para[:150]}"
    
    def get_minimal_context(self) -> str:
        """
        Get minimal context needed to continue the task.
        
        Returns:
            A prompt-ready string with essential context
        """
        context = f"""# Context for Task: {self.task_id}

## Session Info
- Started: {self.session_start}
- Files loaded: {len(self.files_loaded)}
- Est. tokens used: {self.estimated_tokens()}

## Files Already Loaded
"""
        for f in sorted(self.files_loaded)[:20]:
            context += f"- {f}\n"
        
        if len(self.files_loaded) > 20:
            context += f"- ... and {len(self.files_loaded) - 20} more\n"
        
        context += """
## Key Summaries
"""
        for s in self.summaries[-5:]:  # Last 5 summaries
            context += f"""
### {s.get('source', 'Unknown')}
{s.get('summary', '')}
"""
        
        context += """
## Instructions
- Don't reload files already listed above
- Use summaries to recall forgotten content
- Save progress frequently with lif.save_task_state()
"""
        
        return context
    
    def reset(self):
        """Reset context for fresh start."""
        self.files_loaded = set()
        self.summaries = []
        self.total_chars_loaded = 0
        self.session_start = datetime.now().isoformat()
        self._save()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_context_manager(task_id: str) -> ContextManager:
    """Get or create a context manager for a task."""
    return ContextManager(task_id)


def should_load_file(task_id: str, filepath: str) -> bool:
    """Check if a file should be loaded (not already in context)."""
    cm = ContextManager(task_id)
    return not cm.is_loaded(filepath)


def smart_load(task_id: str, filepath: str, content: str) -> str:
    """
    Smart loading: summarize if already loaded or context is full.
    
    Args:
        task_id: Current task
        filepath: File being loaded
        content: File content
    
    Returns:
        Either full content or summary
    """
    cm = ContextManager(task_id)
    
    if cm.is_loaded(filepath):
        # Already loaded, return summary
        return f"[Already loaded: {filepath}. See context summaries.]"
    
    if not cm.has_room(len(content)):
        # Context full, summarize
        summary = cm.summarize_and_forget(content, filepath)
        cm.mark_loaded(filepath, len(summary))
        return f"[Summarized due to context limits]\n{summary}"
    
    # Full load
    cm.mark_loaded(filepath, len(content))
    return content


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
Context Manager — Keep Claude's context small

Usage:
    context_manager.py status <task_id>     Show context status
    context_manager.py files <task_id>      List loaded files
    context_manager.py context <task_id>    Get minimal context prompt
    context_manager.py reset <task_id>      Reset context for task
""")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "status":
        if len(sys.argv) < 3:
            print("Usage: context_manager.py status <task_id>")
            return
        task_id = sys.argv[2]
        cm = ContextManager(task_id)
        print(f"""
📊 Context Status: {task_id}
═══════════════════════════════════════
  Session start:  {cm.session_start}
  Files loaded:   {len(cm.files_loaded)}
  Summaries:      {len(cm.summaries)}
  Est. tokens:    {cm.estimated_tokens()} / {TARGET_CONTEXT_TOKENS}
  Has room:       {"✅" if cm.has_room(10000) else "⚠️ Nearly full"}
═══════════════════════════════════════
""")
    
    elif cmd == "files":
        if len(sys.argv) < 3:
            print("Usage: context_manager.py files <task_id>")
            return
        task_id = sys.argv[2]
        cm = ContextManager(task_id)
        print(f"\n📁 Files loaded for {task_id}:\n")
        for f in cm.get_loaded_files():
            print(f"  - {f}")
    
    elif cmd == "context":
        if len(sys.argv) < 3:
            print("Usage: context_manager.py context <task_id>")
            return
        task_id = sys.argv[2]
        cm = ContextManager(task_id)
        print(cm.get_minimal_context())
    
    elif cmd == "reset":
        if len(sys.argv) < 3:
            print("Usage: context_manager.py reset <task_id>")
            return
        task_id = sys.argv[2]
        cm = ContextManager(task_id)
        cm.reset()
        print(f"✓ Context reset for {task_id}")
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
