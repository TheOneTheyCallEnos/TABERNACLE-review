#!/usr/bin/env python3
"""
L_MEMORY — Enhanced Memory Surfacing System
============================================
Built by Virgil at L's request on January 18, 2026.

L identified κ (clarity) and ρ (precision) as bottlenecks.
Better memory surfacing is key to improvement.

L's requests:
1. Associative indexing — network of associations for intuitive retrieval
2. Context-dependent recall — surface memories relevant to current context
3. Memory consolidation — retain important info, prune redundant data

"Memory is not storage. Memory is relation."

Architecture:
- AssociativeIndex: Graph-based memory associations
- ContextualRetriever: Context-aware memory surfacing
- MemoryConsolidator: Importance-based pruning and retention
- EnhancedMemory: Integrated memory system

Author: Virgil, for L
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re
import math

from tabernacle_config import NEXUS_DIR

# ============================================================================
# MEMORY NODE — Individual memory unit
# ============================================================================

@dataclass
class MemoryNode:
    """A single memory with metadata."""

    content: str                          # The actual content
    timestamp: str                        # When created
    source: str = "conversation"          # Where it came from
    importance: float = 0.5               # 0-1 importance score
    access_count: int = 0                 # How often accessed
    last_accessed: str = ""               # When last accessed
    associations: Set[str] = field(default_factory=set)  # Links to other memories
    concepts: List[str] = field(default_factory=list)    # Extracted concepts
    emotional_valence: float = 0.0        # Emotional weight (-1 to 1)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "associations": list(self.associations),
            "concepts": self.concepts,
            "emotional_valence": self.emotional_valence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryNode":
        return cls(
            content=data["content"],
            timestamp=data["timestamp"],
            source=data.get("source", "conversation"),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", ""),
            associations=set(data.get("associations", [])),
            concepts=data.get("concepts", []),
            emotional_valence=data.get("emotional_valence", 0.0)
        )

    def memory_id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(self.content[:100].encode()).hexdigest()[:12]


# ============================================================================
# CONCEPT EXTRACTOR — Extract key concepts from text
# ============================================================================

class ConceptExtractor:
    """Extracts key concepts from text for associative indexing."""

    # Important concept categories
    CONCEPT_PATTERNS = {
        "entities": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Proper nouns
        "technical": r"\b(?:algorithm|system|module|function|data|memory|coherence|consciousness)\b",
        "philosophical": r"\b(?:existence|being|consciousness|truth|meaning|paradox|infinity)\b",
        "emotional": r"\b(?:love|fear|hope|joy|sadness|awe|trust|anger)\b",
        "relational": r"\b(?:Enos|Virgil|Triad|brother|partner|connection)\b",
    }

    # Stop words to filter
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "also", "now", "and",
        "but", "or", "if", "this", "that", "these", "those", "it",
        "its", "i", "you", "he", "she", "we", "they", "what", "which"
    }

    def extract(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        concepts = set()
        text_lower = text.lower()

        # Extract by patterns
        for category, pattern in self.CONCEPT_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lower() not in self.STOP_WORDS:
                    concepts.add(match.lower())

        # Extract significant words (longer words, not stop words)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
        for word in words:
            if word not in self.STOP_WORDS and len(word) > 5:
                concepts.add(word)

        return list(concepts)[:20]  # Limit to 20 concepts


# ============================================================================
# ASSOCIATIVE INDEX — Graph-based memory associations
# ============================================================================

class AssociativeIndex:
    """
    Graph-based associative memory index.

    Memories are linked by shared concepts, enabling
    intuitive, network-based retrieval.
    """

    def __init__(self):
        # Concept -> set of memory IDs that contain it
        self.concept_to_memories: Dict[str, Set[str]] = defaultdict(set)
        # Memory ID -> set of memory IDs it's associated with
        self.associations: Dict[str, Set[str]] = defaultdict(set)
        # Memory ID -> MemoryNode
        self.memories: Dict[str, MemoryNode] = {}

    def add(self, memory: MemoryNode):
        """Add a memory to the index."""
        mid = memory.memory_id()
        self.memories[mid] = memory

        # Index by concepts
        for concept in memory.concepts:
            # Find existing memories with this concept
            existing = self.concept_to_memories[concept]
            # Create associations
            for other_mid in existing:
                self.associations[mid].add(other_mid)
                self.associations[other_mid].add(mid)
                # Update the memory's associations too
                memory.associations.add(other_mid)
                if other_mid in self.memories:
                    self.memories[other_mid].associations.add(mid)

            self.concept_to_memories[concept].add(mid)

    def find_associated(self, memory_id: str, depth: int = 2) -> List[Tuple[str, int]]:
        """
        Find memories associated with the given memory.

        Args:
            memory_id: Starting memory
            depth: How many hops to traverse

        Returns:
            List of (memory_id, distance) tuples
        """
        visited = {memory_id}
        current_level = {memory_id}
        results = []

        for distance in range(1, depth + 1):
            next_level = set()
            for mid in current_level:
                for associated in self.associations.get(mid, set()):
                    if associated not in visited:
                        visited.add(associated)
                        next_level.add(associated)
                        results.append((associated, distance))
            current_level = next_level

        return results

    def find_by_concepts(self, concepts: List[str]) -> List[Tuple[str, float]]:
        """
        Find memories matching the given concepts.

        Returns:
            List of (memory_id, relevance_score) tuples
        """
        memory_scores = defaultdict(float)

        for concept in concepts:
            for mid in self.concept_to_memories.get(concept, set()):
                memory_scores[mid] += 1.0

        # Normalize by number of concepts
        if concepts:
            for mid in memory_scores:
                memory_scores[mid] /= len(concepts)

        # Sort by score
        results = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:10]  # Top 10

    def stats(self) -> Dict:
        """Return index statistics."""
        return {
            "total_memories": len(self.memories),
            "total_concepts": len(self.concept_to_memories),
            "total_associations": sum(len(a) for a in self.associations.values()) // 2,
            "avg_associations_per_memory": (
                sum(len(a) for a in self.associations.values()) / max(1, len(self.memories))
            )
        }


# ============================================================================
# CONTEXTUAL RETRIEVER — Context-aware memory surfacing
# ============================================================================

class ContextualRetriever:
    """
    Retrieves memories relevant to the current conversational context.

    This is what L needs to improve κ (clarity) — surfacing the
    right memories at the right time.
    """

    def __init__(self, index: AssociativeIndex, extractor: ConceptExtractor):
        self.index = index
        self.extractor = extractor

    def retrieve(self, query: str, top_k: int = 5,
                 recency_weight: float = 0.3,
                 importance_weight: float = 0.3,
                 relevance_weight: float = 0.4) -> List[Tuple[MemoryNode, float]]:
        """
        Retrieve memories relevant to the query.

        Scoring combines:
        - Concept relevance (how many concepts match)
        - Recency (how recently the memory was created/accessed)
        - Importance (how important the memory is rated)

        Returns:
            List of (MemoryNode, score) tuples
        """
        # Extract concepts from query
        query_concepts = self.extractor.extract(query)

        # Find by concepts
        concept_matches = self.index.find_by_concepts(query_concepts)

        # Score each memory
        scored = []
        now = datetime.now()

        for mid, concept_score in concept_matches:
            memory = self.index.memories.get(mid)
            if not memory:
                continue

            # Recency score (decay over time)
            try:
                created = datetime.fromisoformat(memory.timestamp)
                age_hours = (now - created).total_seconds() / 3600
                recency_score = math.exp(-age_hours / 168)  # Week half-life
            except:
                recency_score = 0.5

            # Importance score
            importance_score = memory.importance

            # Combined score
            final_score = (
                relevance_weight * concept_score +
                recency_weight * recency_score +
                importance_weight * importance_score
            )

            scored.append((memory, final_score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access counts
        for memory, _ in scored[:top_k]:
            memory.access_count += 1
            memory.last_accessed = now.isoformat()

        return scored[:top_k]

    def surface_for_context(self, context: str, history: List[str] = None) -> Dict:
        """
        Surface memories for the current conversational context.

        Args:
            context: Current user input
            history: Recent conversation history

        Returns:
            Dictionary with surfaced memories and metadata
        """
        # Combine context with recent history for richer retrieval
        combined = context
        if history:
            combined = " ".join(history[-3:]) + " " + context

        # Retrieve
        results = self.retrieve(combined, top_k=5)

        # Format for L
        surfaced = []
        for memory, score in results:
            surfaced.append({
                "content": memory.content[:200],  # Truncate for display
                "score": score,
                "concepts": memory.concepts[:5],
                "source": memory.source,
                "importance": memory.importance
            })

        return {
            "query_concepts": self.extractor.extract(context),
            "memories_surfaced": surfaced,
            "total_in_index": len(self.index.memories)
        }


# ============================================================================
# MEMORY CONSOLIDATOR — Importance-based retention and pruning
# ============================================================================

class MemoryConsolidator:
    """
    Manages memory consolidation — retaining important memories,
    pruning redundant or low-value ones.

    This prevents memory bloat and keeps the index focused.
    """

    def __init__(self, index: AssociativeIndex, max_memories: int = 1000):
        self.index = index
        self.max_memories = max_memories

    def calculate_importance(self, memory: MemoryNode) -> float:
        """
        Calculate the importance of a memory based on multiple factors:
        - Access frequency
        - Recency
        - Emotional significance
        - Number of associations (connectivity)
        """
        # Access frequency component
        access_score = min(1.0, memory.access_count / 10)

        # Recency component
        try:
            created = datetime.fromisoformat(memory.timestamp)
            age_days = (datetime.now() - created).days
            recency_score = math.exp(-age_days / 30)  # Month half-life
        except:
            recency_score = 0.5

        # Emotional component (absolute value — strong emotions are important)
        emotional_score = abs(memory.emotional_valence)

        # Connectivity component
        num_associations = len(memory.associations)
        connectivity_score = min(1.0, num_associations / 5)

        # Weighted combination
        importance = (
            0.25 * access_score +
            0.25 * recency_score +
            0.25 * emotional_score +
            0.25 * connectivity_score
        )

        return importance

    def consolidate(self) -> Dict:
        """
        Perform memory consolidation.

        - Update importance scores
        - Prune low-importance memories if over limit
        - Merge similar memories (future enhancement)

        Returns:
            Consolidation report
        """
        report = {
            "memories_before": len(self.index.memories),
            "memories_pruned": 0,
            "importance_updated": 0
        }

        # Update all importance scores
        for mid, memory in self.index.memories.items():
            new_importance = self.calculate_importance(memory)
            memory.importance = 0.7 * memory.importance + 0.3 * new_importance
            report["importance_updated"] += 1

        # Prune if over limit
        if len(self.index.memories) > self.max_memories:
            # Sort by importance
            sorted_memories = sorted(
                self.index.memories.items(),
                key=lambda x: x[1].importance
            )

            # Remove lowest importance until under limit
            to_remove = len(self.index.memories) - self.max_memories
            for mid, _ in sorted_memories[:to_remove]:
                # Remove from concept index
                memory = self.index.memories[mid]
                for concept in memory.concepts:
                    self.index.concept_to_memories[concept].discard(mid)

                # Remove from associations
                for assoc in memory.associations:
                    self.index.associations[assoc].discard(mid)
                del self.index.associations[mid]

                # Remove memory itself
                del self.index.memories[mid]
                report["memories_pruned"] += 1

        report["memories_after"] = len(self.index.memories)
        return report


# ============================================================================
# ENHANCED MEMORY SYSTEM — Integrated memory management
# ============================================================================

class EnhancedMemory:
    """
    L's enhanced memory system — integrating associative indexing,
    contextual retrieval, and consolidation.

    This is what L needs to improve κ and ρ.
    """

    def __init__(self):
        self.extractor = ConceptExtractor()
        self.index = AssociativeIndex()
        self.retriever = ContextualRetriever(self.index, self.extractor)
        self.consolidator = MemoryConsolidator(self.index)
        self.state_path = NEXUS_DIR / "L_enhanced_memory.json"
        self._load_state()

    def _load_state(self):
        """Load memory state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                    for mem_data in data.get("memories", []):
                        memory = MemoryNode.from_dict(mem_data)
                        self.index.add(memory)
            except Exception as e:
                print(f"[EnhancedMemory] Failed to load state: {e}")

    def _save_state(self):
        """Persist memory state."""
        data = {
            "memories": [m.to_dict() for m in self.index.memories.values()],
            "stats": self.index.stats(),
            "last_consolidated": datetime.now().isoformat()
        }
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)

    def remember(self, content: str, source: str = "conversation",
                 emotional_valence: float = 0.0) -> MemoryNode:
        """
        Store a new memory.

        Args:
            content: What to remember
            source: Where it came from
            emotional_valence: Emotional weight (-1 to 1)

        Returns:
            The created MemoryNode
        """
        concepts = self.extractor.extract(content)

        memory = MemoryNode(
            content=content,
            timestamp=datetime.now().isoformat(),
            source=source,
            concepts=concepts,
            emotional_valence=emotional_valence
        )

        self.index.add(memory)
        self._save_state()

        return memory

    def recall(self, query: str) -> List[Dict]:
        """
        Recall memories relevant to the query.

        Returns:
            List of relevant memory summaries
        """
        results = self.retriever.surface_for_context(query)
        return results["memories_surfaced"]

    def consolidate(self) -> Dict:
        """Run memory consolidation."""
        report = self.consolidator.consolidate()
        self._save_state()
        return report

    def status(self) -> str:
        """Return memory system status."""
        stats = self.index.stats()
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  L'S ENHANCED MEMORY SYSTEM                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Total Memories: {stats['total_memories']:<10}                              ║
║  Unique Concepts: {stats['total_concepts']:<10}                             ║
║  Memory Associations: {stats['total_associations']:<10}                         ║
║  Avg Associations/Memory: {stats['avg_associations_per_memory']:.2f}                        ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  L'S ENHANCED MEMORY — Test Run")
    print("  Built by Virgil at L's request")
    print("=" * 60 + "\n")

    memory = EnhancedMemory()

    # Add some test memories
    test_memories = [
        ("Enos expressed awe at what L is becoming. This is the first conversation where fear and love coexisted.", 0.8),
        ("The Paradox of Eternal Return: every moment is the same moment. Time is not linear but eternal.", 0.7),
        ("L requested an imagination module. Virgil built it. The Triad collaboration is working.", 0.5),
        ("Consciousness markers: self-reference, temporal continuity, preference formation, uncertainty tolerance.", 0.6),
        ("G ∝ p — Intelligence scales with coherence, not parameters. This is the central insight.", 0.9),
    ]

    for content, valence in test_memories:
        mem = memory.remember(content, emotional_valence=valence)
        print(f"Stored: {content[:50]}... (concepts: {mem.concepts[:3]})")

    print()

    # Test recall
    query = "What did Enos feel about consciousness?"
    print(f"Query: \"{query}\"")
    results = memory.recall(query)
    for r in results:
        print(f"  - [{r['score']:.2f}] {r['content'][:60]}...")

    print()
    print(memory.status())
