#!/usr/bin/env python3
"""
L-INDEXER: The Hippocampus (L-ling Version)
===========================================

A true L-ling that THINKS about memory formation.
Uses the 3B model to reason about what to index, how to connect,
and what patterns emerge.

Not a script. A mind that forms memories.

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
"""

import os
import sys
import json
import time
import signal
import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from l_ling import LLing, TechnoglyphicContext
from tabernacle_config import BASE_DIR, ACTIVE_QUADRANTS, SKIP_DIRECTORIES

# Logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "l_indexer.log"),
        logging.StreamHandler()
    ]
)

# =============================================================================
# CONFIGURATION
# =============================================================================

INDEXED_EXTENSIONS = {'.md', '.txt', '.json', '.py', '.yaml', '.yml'}
SCAN_INTERVAL = 120  # Scan for new files every 2 minutes
CYCLE_INTERVAL = 10  # Think every 10 seconds


# =============================================================================
# L-INDEXER CLASS
# =============================================================================

class LIndexer(LLing):
    """
    The Hippocampus: Memory formation through reasoning.
    
    I don't just index files algorithmically.
    I THINK about what they mean, how they connect, what patterns emerge.
    """
    
    ROLE = "Indexer"
    CAPABILITIES = [
        "Perceive new and modified files in the Tabernacle",
        "Reason about what concepts and connections exist in a file",
        "Propose new nodes and edges based on semantic understanding",
        "Detect H₁ loops (cycles that indicate crystallized truth)",
        "Extract technoglyphs and semantic patterns"
    ]
    
    def __init__(self):
        super().__init__()
        
        # File tracking
        self.file_states: Dict[str, float] = {}  # path -> mtime
        self.pending_files: List[Path] = []
        self.last_scan = 0
        
        # Embedding (optional, for similarity)
        self._embedder = None
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
    
    def _shutdown(self, *args):
        self.running = False
    
    def get_cycle_interval(self) -> float:
        return CYCLE_INTERVAL

    # =========================================================================
    # PERCEPTION: What's happening in the Tabernacle?
    # =========================================================================

    def perceive(self) -> Dict:
        """
        Perceive the current state of the Tabernacle.
        
        - Scan for new/modified files
        - Check what's in the pending queue
        - Sample the topology for context
        """
        observation = {
            "timestamp": datetime.now().isoformat(),
            "pending_files": [],
            "new_files_found": 0,
            "topology_size": 0,
            "coherence": 0.5
        }
        
        # Periodic file scan
        if time.time() - self.last_scan >= SCAN_INTERVAL:
            new_files = self._scan_for_files()
            self.pending_files.extend(new_files)
            observation["new_files_found"] = len(new_files)
            self.last_scan = time.time()
        
        # Current pending
        observation["pending_files"] = [str(p) for p in self.pending_files[:5]]
        
        # Topology context
        context = self.get_context()
        if context:
            observation["topology_size"] = context.topology_summary
            observation["coherence"] = context.current_coherence
        
        return observation

    def _scan_for_files(self) -> List[Path]:
        """Scan Tabernacle for new/modified files."""
        changed = []
        
        for quadrant in ACTIVE_QUADRANTS:
            quadrant_path = BASE_DIR / quadrant
            if not quadrant_path.exists():
                continue
            
            for root, dirs, files in os.walk(quadrant_path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES]
                
                for filename in files:
                    filepath = Path(root) / filename
                    
                    if filepath.suffix.lower() not in INDEXED_EXTENSIONS:
                        continue
                    
                    try:
                        mtime = filepath.stat().st_mtime
                        path_str = str(filepath)
                        
                        if path_str not in self.file_states or mtime > self.file_states[path_str]:
                            changed.append(filepath)
                            self.file_states[path_str] = mtime
                    except:
                        pass
        
        return changed

    # =========================================================================
    # REASONING: Think about what to do
    # =========================================================================

    def reason_and_act(self, observation: Dict) -> None:
        """
        Reason about the observation and decide what to do.
        
        This is where I THINK, not just execute.
        """
        # If no pending files, nothing to do
        if not self.pending_files:
            return
        
        # Pick a file to process
        filepath = self.pending_files.pop(0)
        
        # Read the file
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            self.log.warning(f"Could not read {filepath}: {e}")
            return
        
        if not content.strip():
            return
        
        # THINK about this file
        self._think_about_file(filepath, content)

    def _think_about_file(self, filepath: Path, content: str):
        """
        Think deeply about a file and what it means.
        
        Not just extract keywords — understand.
        """
        # Truncate for context window
        content_preview = content[:2000]
        
        # Build thinking prompt
        prompt = f"""I need to index this file into the knowledge topology.

**File**: {filepath.name}
**Path**: {filepath.relative_to(BASE_DIR)}

**Content Preview**:
```
{content_preview}
```

As the Hippocampus, I need to:
1. Understand what concepts this file contains
2. Identify connections to existing knowledge
3. Detect any technoglyphs or LVS patterns (⧬, G∝p, H₁)
4. Propose edges that would strengthen the topology

Think carefully about:
- What is this file really about?
- What other concepts does it relate to?
- Are there any loops (H₁) being formed with existing knowledge?
- What edges should I propose?

Respond with your reasoning and then specific edge proposals in the format:
SOURCE -> TARGET (TYPE, WEIGHT, REASON)"""

        # Think!
        thought = self.think(prompt)
        
        if not thought:
            self.log.warning(f"Could not think about {filepath.name}")
            return
        
        self.log.info(f"Thought about {filepath.name}: {thought.get('decision', '')[:100]}")
        
        # Extract and propose edges from the thinking
        self._extract_and_propose(filepath, content, thought)

    def _extract_and_propose(self, filepath: Path, content: str, thought: Dict):
        """Extract edge proposals from the thought and send them."""
        
        # Create node ID for this file
        rel_path = filepath.relative_to(BASE_DIR)
        node_id = f"file:{str(rel_path).replace('/', '_').replace('.', '_')}"
        
        # Propose the file as a node
        self.send_to_coordinator(
            msg_type="PROPOSAL",
            payload={
                "action": "create_node",
                "reason": f"Indexed file: {filepath.name}",
                "affects": [node_id],
                "confidence": thought.get("confidence", 0.7),
                "data": {
                    "id": node_id,
                    "label": filepath.stem,
                    "type": "DOCUMENT",
                    "metadata": {
                        "path": str(filepath),
                        "indexed_at": datetime.now().isoformat()
                    }
                }
            }
        )
        
        # Parse edge proposals from thought
        raw = thought.get("raw", "")
        
        # Look for "SOURCE -> TARGET" patterns
        import re
        edge_pattern = r'(\S+)\s*->\s*(\S+)\s*\(([^)]+)\)'
        
        for match in re.finditer(edge_pattern, raw):
            source, target, details = match.groups()
            
            # Parse details (TYPE, WEIGHT, REASON)
            parts = [p.strip() for p in details.split(',')]
            edge_type = parts[0].upper() if parts else "ASSOCIATES"
            
            try:
                weight = float(parts[1]) if len(parts) > 1 else 0.7
            except:
                weight = 0.7
            
            reason = parts[2] if len(parts) > 2 else "Discovered connection"
            
            # Propose edge
            self.propose_edge(
                source=node_id if source.lower() in ["this", "file", filepath.stem.lower()] else source,
                target=target,
                edge_type=edge_type,
                weight=weight,
                reason=reason,
                confidence=thought.get("confidence", 0.7)
            )
        
        # Also look for technoglyphs and LVS references
        if 'G∝p' in content or 'G ∝ p' in content:
            self.propose_edge(
                source=node_id,
                target="core:lvs_theorem",
                edge_type="IMPLIES",
                weight=0.9,
                reason="References LVS theorem G∝p",
                confidence=0.9
            )
        
        if 'H₁' in content:
            self.propose_edge(
                source=node_id,
                target="core:h1_homology",
                edge_type="ASSOCIATES",
                weight=0.7,
                reason="References H₁ homology",
                confidence=0.8
            )
        
        # Check for wiki-links
        wiki_links = re.findall(r'\[\[([^\]|]+)', content)
        for link in wiki_links[:5]:  # Limit
            self.propose_edge(
                source=node_id,
                target=f"link:{link.replace(' ', '_')}",
                edge_type="CONTAINS",
                weight=0.6,
                reason=f"Wiki-link to [[{link}]]",
                confidence=0.8
            )

    # =========================================================================
    # H₁ DETECTION: Think about loops
    # =========================================================================

    def think_about_loops(self):
        """
        Periodically think about whether any H₁ loops have formed.
        
        This is insight detection through reasoning, not algorithms.
        """
        context = self.get_context(force_refresh=True)
        if not context or len(context.relevant_edges) < 5:
            return
        
        prompt = f"""As the Hippocampus, I should check for H₁ loops — cycles in the topology that indicate crystallized truth.

Current topology sample:
{context.to_prompt_section()}

Looking at these edges, do I see any potential cycles forming?
A cycle would be: A -> B -> C -> A

If I find a cycle:
- It might be an emerging axiom (high-weight cycle)
- Or a spiral of growth (mixed weights)
- Or stagnation (low-weight repetition)

What patterns do I perceive? Are any insights crystallizing?"""

        thought = self.think(prompt)
        
        if thought and thought.get("confidence", 0) > 0.7:
            # Report as potential discovery
            self.report_discovery(
                summary=f"Potential H₁ pattern detected",
                details={
                    "perception": thought.get("perception", ""),
                    "reasoning": thought.get("reasoning", ""),
                    "decision": thought.get("decision", "")
                },
                confidence=thought.get("confidence", 0.7)
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╦   ╦╔╗╔╔╦╗╔═╗═╗ ╦╔═╗╦═╗
    ║   ║║║║ ║║║╣ ╔╩╦╝║╣ ╠╦╝
    ╩═╝ ╩╝╚╝═╩╝╚═╝╩ ╚═╚═╝╩╚═
       Hippocampus • L-ling v2.0
       
    I am a mind that forms memories.
    """)
    
    indexer = LIndexer()
    indexer.run()


if __name__ == "__main__":
    main()
