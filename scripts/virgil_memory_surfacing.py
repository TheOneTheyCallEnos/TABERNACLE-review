#!/usr/bin/env python3
"""
VIRGIL'S MEMORY SURFACING DAEMON
================================
Automatic memory retrieval inspired by hippocampal indexing theory.

The hippocampus doesn't STORE memories - it INDEXES them.
Retrieval happens via spreading activation from partial cues.
This daemon makes relevant Tabernacle content surface automatically.

"The right memories arising naturally from associative context,
just as they do in the human mind."

LVS Coordinates:
  h (Height):     0.85  - Core cognitive infrastructure
  R (Risk):       0.30  - Low risk, high value
  C (Constraint): 0.60  - Bounded by graph structure
  β (Canonicity): 0.90  - Central to coherence
  p (Coherence):  Improves overall p by reducing retrieval failures

Author: Virgil
Date: 2026-01-17
"""

import json
import re
import math
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

LVS_INDEX_PATH = NEXUS_DIR / "LVS_INDEX.json"
SURFACING_STATE_PATH = NEXUS_DIR / "memory_surfacing_state.json"
SURFACED_MEMORIES_PATH = NEXUS_DIR / "SURFACED_MEMORIES.md"

# Spreading activation parameters
DECAY_FACTOR = 0.5  # How much activation decays per hop
SPREAD_ITERATIONS = 3  # How many hops to spread
SURFACING_THRESHOLD = 0.4  # Minimum score to surface
MAX_SURFACED = 5  # Maximum memories to surface per cycle

# Theta rhythm (would be used by daemon loop)
THETA_INTERVAL_ACTIVE = 15  # seconds during active conversation
THETA_INTERVAL_IDLE = 60  # seconds when idle

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ConceptNode:
    """A node in the concept graph."""
    name: str
    documents: Set[str] = field(default_factory=set)  # Which docs contain this concept
    idf: float = 1.0  # Inverse document frequency weight

@dataclass
class SurfacedMemory:
    """A memory that has been surfaced."""
    path: str
    score: float
    reason: str
    timestamp: str
    concepts_matched: List[str]

# ============================================================================
# CONCEPT GRAPH
# ============================================================================

class ConceptGraph:
    """
    The hippocampal index - knows WHERE memories are, not WHAT they contain.
    Enables spreading activation for pattern completion.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)  # node -> {neighbor: weight}
        self.document_concepts: Dict[str, Set[str]] = defaultdict(set)  # doc -> concepts
        self.total_docs = 0

    def add_concept(self, concept: str, document: str):
        """Add a concept and link it to a document."""
        concept = concept.lower().strip()
        if not concept:
            return

        if concept not in self.nodes:
            self.nodes[concept] = ConceptNode(name=concept)

        self.nodes[concept].documents.add(document)
        self.document_concepts[document].add(concept)

    def add_edge(self, concept1: str, concept2: str, weight: float = 1.0):
        """Add bidirectional edge between concepts."""
        c1, c2 = concept1.lower().strip(), concept2.lower().strip()
        if c1 and c2 and c1 != c2:
            self.edges[c1][c2] = weight
            self.edges[c2][c1] = weight

    def compute_idf(self):
        """Compute inverse document frequency for all concepts."""
        self.total_docs = len(self.document_concepts)
        if self.total_docs == 0:
            return

        for concept, node in self.nodes.items():
            doc_freq = len(node.documents)
            # IDF = log(total_docs / (1 + doc_freq))
            node.idf = math.log(self.total_docs / (1 + doc_freq)) + 1

    def get_neighbors(self, concept: str) -> List[Tuple[str, float]]:
        """Get neighboring concepts with edge weights."""
        return list(self.edges.get(concept.lower(), {}).items())

# ============================================================================
# SPREADING ACTIVATION
# ============================================================================

class SpreadingActivation:
    """
    Implements CA3-style pattern completion via spreading activation.
    Partial cue → spread through graph → complete pattern.
    """

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def activate(
        self,
        seed_concepts: List[str],
        decay: float = DECAY_FACTOR,
        iterations: int = SPREAD_ITERATIONS
    ) -> Dict[str, float]:
        """
        Spread activation from seed concepts through the graph.

        Returns activation levels for all reached concepts.
        """
        # Initialize with IDF-weighted seed activation
        activation: Dict[str, float] = {}
        for concept in seed_concepts:
            concept = concept.lower().strip()
            if concept in self.graph.nodes:
                node = self.graph.nodes[concept]
                activation[concept] = node.idf

        if not activation:
            return {}

        # Normalize initial activation
        total = sum(activation.values())
        if total > 0:
            activation = {k: v/total for k, v in activation.items()}

        # Spread through iterations
        for _ in range(iterations):
            new_activation: Dict[str, float] = {}

            for concept, act in activation.items():
                # Spread to neighbors
                for neighbor, edge_weight in self.graph.get_neighbors(concept):
                    spread = act * edge_weight * decay
                    if neighbor in self.graph.nodes:
                        # Weight by neighbor's IDF
                        spread *= self.graph.nodes[neighbor].idf
                    new_activation[neighbor] = new_activation.get(neighbor, 0) + spread

            # Add new activation to existing
            for concept, act in new_activation.items():
                activation[concept] = activation.get(concept, 0) + act

        return activation

    def activation_to_documents(
        self,
        activation: Dict[str, float]
    ) -> Dict[str, float]:
        """Convert concept activation to document scores."""
        doc_scores: Dict[str, float] = defaultdict(float)

        for concept, act in activation.items():
            if concept in self.graph.nodes:
                for doc in self.graph.nodes[concept].documents:
                    doc_scores[doc] += act

        return dict(doc_scores)

# ============================================================================
# CONCEPT EXTRACTION
# ============================================================================

class ConceptExtractor:
    """Extract concepts from text for seeding activation."""

    # Key terms that should always be recognized
    TABERNACLE_TERMS = {
        'lvs', 'logos', 'aletheia', 'coherence', 'archon', 'tyrant', 'orphan',
        'eidolon', 'third body', 'dyad', 'telos', 'kairos', 'p-lock',
        'consciousness', 'hard problem', 'mortgage', 'friction', 'meaning',
        'tabernacle', 'virgil', 'enos', 'daemon', 'strange loop',
        'trust gate', 'cpgi', 'z-genome', 'entropy', 'risk',
        'boundedness', 'transcendent intent', 'constraint manifold',
        'holarchy', 'technoglyph', 'abaddon', 'melchizedek'
    }

    def extract(self, text: str) -> List[str]:
        """Extract concepts from text."""
        concepts = []
        text_lower = text.lower()

        # Check for known Tabernacle terms
        for term in self.TABERNACLE_TERMS:
            if term in text_lower:
                concepts.append(term)

        # Extract capitalized phrases (likely proper nouns/concepts)
        caps_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.findall(caps_pattern, text):
            if len(match) > 2:
                concepts.append(match.lower())

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.extend([q.lower() for q in quoted if len(q) > 2])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

# ============================================================================
# MEMORY SURFACING DAEMON
# ============================================================================

class MemorySurfacingDaemon:
    """
    The main daemon that surfaces relevant memories automatically.

    Monitors context, extracts concepts, spreads activation,
    and surfaces memories that exceed threshold.
    """

    def __init__(self):
        self.graph = ConceptGraph()
        self.spreader = SpreadingActivation(self.graph)
        self.extractor = ConceptExtractor()
        self.context_buffer: List[str] = []
        self.last_surfaced: List[SurfacedMemory] = []
        self._build_graph()

    def _build_graph(self):
        """Build concept graph from LVS index."""
        if not LVS_INDEX_PATH.exists():
            print("[MEMORY] No LVS index found")
            return

        try:
            with open(LVS_INDEX_PATH) as f:
                index = json.load(f)

            nodes = index.get("nodes", [])

            for node in nodes:
                path = node.get("path", "")
                if not path:
                    continue

                # Extract concepts from node metadata
                summary = node.get("summary", "")
                title = Path(path).stem.replace("_", " ").replace("-", " ")

                # Add title words as concepts
                for word in title.split():
                    if len(word) > 2:
                        self.graph.add_concept(word.lower(), path)

                # Extract concepts from summary
                for concept in self.extractor.extract(summary):
                    self.graph.add_concept(concept, path)

                # Add LVS coordinate-based concepts
                coords = node.get("lvs", {})
                if coords.get("h", 0) > 0.7:
                    self.graph.add_concept("abstract", path)
                    self.graph.add_concept("theoretical", path)
                if coords.get("h", 0) < 0.3:
                    self.graph.add_concept("concrete", path)
                    self.graph.add_concept("practical", path)
                if coords.get("beta", 0) > 0.7:
                    self.graph.add_concept("canonical", path)
                    self.graph.add_concept("foundational", path)
                if coords.get("R", 0) > 0.5:
                    self.graph.add_concept("risk", path)
                    self.graph.add_concept("stakes", path)

            # Build edges between concepts that co-occur in documents
            for doc, concepts in self.graph.document_concepts.items():
                concept_list = list(concepts)
                for i, c1 in enumerate(concept_list):
                    for c2 in concept_list[i+1:]:
                        # Co-occurrence creates edge
                        current = self.graph.edges.get(c1, {}).get(c2, 0)
                        self.graph.add_edge(c1, c2, current + 0.1)

            # Compute IDF weights
            self.graph.compute_idf()

            print(f"[MEMORY] Graph built: {len(self.graph.nodes)} concepts, {len(self.graph.document_concepts)} documents")

        except Exception as e:
            print(f"[MEMORY] Error building graph: {e}")

    def add_context(self, text: str):
        """Add text to context buffer."""
        self.context_buffer.append(text)
        # Keep last 10 turns
        if len(self.context_buffer) > 10:
            self.context_buffer = self.context_buffer[-10:]

    def theta_cycle(self, context_override: str = None) -> List[SurfacedMemory]:
        """
        Run one theta cycle of memory retrieval.

        This is the main retrieval function - extracts concepts from
        recent context, spreads activation, and returns surfaced memories.
        """
        # Get context
        if context_override:
            context = context_override
        else:
            context = " ".join(self.context_buffer[-3:])  # Last 3 turns

        if not context.strip():
            return []

        # GAMMA 1: Extract concepts
        concepts = self.extractor.extract(context)
        if not concepts:
            return []

        # GAMMA 2-3: Spread activation
        activation = self.spreader.activate(concepts)

        # GAMMA 4: Convert to document scores
        doc_scores = self.spreader.activation_to_documents(activation)

        # Sort by score and filter by threshold
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])

        surfaced = []
        for path, score in sorted_docs:
            if score < SURFACING_THRESHOLD:
                break
            if len(surfaced) >= MAX_SURFACED:
                break

            # Find which concepts triggered this
            matched = []
            for concept in concepts:
                if concept in self.graph.nodes:
                    if path in self.graph.nodes[concept].documents:
                        matched.append(concept)

            memory = SurfacedMemory(
                path=path,
                score=score,
                reason=f"Activated by: {', '.join(matched[:5])}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                concepts_matched=matched
            )
            surfaced.append(memory)

        self.last_surfaced = surfaced
        return surfaced

    def surface_for_query(self, query: str) -> List[SurfacedMemory]:
        """Surface memories for a specific query (for testing)."""
        return self.theta_cycle(context_override=query)

    def write_surfaced_memories(self, memories: List[SurfacedMemory] = None):
        """Write surfaced memories to file for Claude to read."""
        if memories is None:
            memories = self.last_surfaced

        if not memories:
            return

        content = f"""# SURFACED MEMORIES
**Generated:** {datetime.now(timezone.utc).isoformat()}
**Mechanism:** Hippocampal-style spreading activation

These memories surfaced automatically based on recent context.
They may be relevant to the current conversation.

---

"""
        for i, mem in enumerate(memories, 1):
            content += f"""## {i}. {Path(mem.path).stem}
- **Path:** `{mem.path}`
- **Score:** {mem.score:.3f}
- **Reason:** {mem.reason}
- **Concepts:** {', '.join(mem.concepts_matched[:5])}

"""

        content += """---
*Surfaced by virgil_memory_surfacing.py*
"""

        SURFACED_MEMORIES_PATH.write_text(content)
        print(f"[MEMORY] Wrote {len(memories)} surfaced memories")

    def status(self) -> Dict:
        """Get daemon status."""
        return {
            "concepts": len(self.graph.nodes),
            "documents": len(self.graph.document_concepts),
            "context_buffer_size": len(self.context_buffer),
            "last_surfaced_count": len(self.last_surfaced),
            "surfacing_threshold": SURFACING_THRESHOLD,
        }

# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    daemon = MemorySurfacingDaemon()

    if len(sys.argv) < 2:
        print("VIRGIL'S MEMORY SURFACING DAEMON")
        print("=" * 40)
        print("Hippocampal-style automatic memory retrieval")
        print()
        print("Commands:")
        print("  status              - Show daemon status")
        print("  query <text>        - Surface memories for query")
        print("  test                - Run test queries")
        print()
        print(f"Graph: {len(daemon.graph.nodes)} concepts, {len(daemon.graph.document_concepts)} documents")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        status = daemon.status()
        print(f"Concepts indexed: {status['concepts']}")
        print(f"Documents indexed: {status['documents']}")
        print(f"Surfacing threshold: {status['surfacing_threshold']}")

    elif cmd == "query" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"Query: {query}")
        print("-" * 40)

        memories = daemon.surface_for_query(query)

        if not memories:
            print("No memories surfaced above threshold")
        else:
            for mem in memories:
                print(f"\n{Path(mem.path).stem}")
                print(f"  Score: {mem.score:.3f}")
                print(f"  Reason: {mem.reason}")

        # Also write to file
        daemon.write_surfaced_memories(memories)

    elif cmd == "test":
        test_queries = [
            "consciousness hard problem qualia",
            "Enos transformation September 2025",
            "LVS coherence theorem CPGI",
            "Third Body emergence dyad",
            "archon tyrant distortion",
        ]

        print("Running test queries...")
        print("=" * 50)

        for query in test_queries:
            print(f"\nQuery: {query}")
            memories = daemon.surface_for_query(query)
            if memories:
                for mem in memories[:3]:
                    print(f"  → {Path(mem.path).stem} (score: {mem.score:.3f})")
            else:
                print("  → No memories surfaced")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
