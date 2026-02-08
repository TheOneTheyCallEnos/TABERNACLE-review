#!/usr/bin/env python3
"""
RIE RELATIONAL MEMORY v2
========================
Enhanced edge-primary memory with:
- NEW concept discovery (not just known concepts)
- Temporal decay (edges weaken over time)
- Edge pruning (garbage collection)
- Bidirectional LVS sync
- Hebbian ceiling (log-scale weight growth)
- Coherence-weighted learning (high-p edges are stronger)

Fixes from v1:
1. Concept extraction now creates NEW nodes for unknown words
2. Edges decay based on time since last activation
3. Pruning removes old, weak edges
4. LVS index changes sync bidirectionally
5. Weights use log scale to prevent unbounded growth

Author: Virgil
Date: 2026-01-18
Status: ENHANCED — Addressing P-Lock bottlenecks
"""

import json
import math
import hashlib
import sys
import fcntl
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import re

# BiologicalEdge import - needs path manipulation for multi-context use
sys.path.insert(0, str(Path(__file__).parent))
from biological_edge import BiologicalEdge

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

RELATIONAL_MEMORY_PATH = NEXUS_DIR / "rie_relational_memory.json"
LVS_INDEX_PATH = NEXUS_DIR / "LVS_INDEX.json"

# Memory parameters
ACTIVATION_DECAY = 0.7          # How much activation decays per hop
SPREAD_ITERATIONS = 3           # How many hops to spread
HEBBIAN_LEARNING_RATE = 0.1     # How fast relations strengthen
COHERENCE_WEIGHT = 2.0          # How much coherence boosts retrieval
MIN_ACTIVATION_THRESHOLD = 0.1  # Minimum to surface

# NEW v2 PARAMETERS
TEMPORAL_DECAY_DAYS = 7.0       # Half-life for edge decay (days)
MAX_WEIGHT = 10.0               # Maximum edge weight (ceiling)
MIN_WEIGHT_PRUNE = 0.05         # Edges below this are pruned
MIN_ACTIVATION_PRUNE = 30       # Days without activation before prune candidate
CONCEPT_MIN_LENGTH = 3          # Minimum word length for concepts
STOPWORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they',
    'this', 'that', 'with', 'from', 'would', 'there', 'their', 'what', 'about',
    'which', 'when', 'make', 'like', 'just', 'over', 'such', 'into', 'than',
    'them', 'some', 'could', 'other', 'then', 'its', 'also', 'these', 'very',
    'after', 'most', 'only', 'those', 'being', 'because', 'through', 'between',
    'does', 'each', 'how', 'where', 'while', 'should', 'here', 'more', 'will'
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Node:
    """A node is just an anchor point — minimal structure."""
    id: str
    label: str
    node_type: str = "concept"  # concept, document, entity
    created_at: str = ""
    activation: float = 0.0
    # NEW: track discovery context
    discovered_in_context: str = ""
    # NEW: Link to LVS vector chunks for hybrid retrieval
    lvs_chunk_ids: List[str] = field(default_factory=list)

@dataclass
class Edge:
    """The primary unit of intelligence."""
    id: str
    source_id: str
    target_id: str
    relation_type: str

    weight: float = 1.0
    coherence_at_formation: float = 0.5

    created_at: str = ""
    last_activated: str = ""
    activation_count: int = 0

    formation_context: str = ""
    current_activation: float = 0.0
    
    # Hub resistance: higher = slower decay (topological memory)
    # 1.0 = normal (7-day half-life), 3.0 = major hub (21-day half-life)
    hub_resistance: float = 1.0
    
    # CACHED datetime to avoid repeated parsing (CPU fix 2026-01-21)
    _last_activated_dt: datetime = field(default=None, repr=False)

    def effective_weight(self, current_time: datetime = None) -> float:
        """
        Compute effective weight with coherence boost AND temporal decay.
        High-coherence relations are more trustworthy.
        Recent relations are stronger than old ones.
        
        FIX 2026-01-21: Cache parsed datetime to avoid O(n) string parsing
        on every edge traversal during spread_activation(). With 55k edges
        and 3 iterations, this was causing 99% CPU for 12+ minutes.
        """
        base = self.weight * (1 + COHERENCE_WEIGHT * self.coherence_at_formation)

        # Apply temporal decay
        if current_time and self.last_activated:
            try:
                # USE CACHED DATETIME (the fix)
                if self._last_activated_dt is None:
                    self._last_activated_dt = datetime.fromisoformat(
                        self.last_activated.replace('Z', '+00:00')
                    )
                    if self._last_activated_dt.tzinfo is None:
                        self._last_activated_dt = self._last_activated_dt.replace(tzinfo=timezone.utc)
                
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)

                days_since = (current_time - self._last_activated_dt).total_seconds() / 86400
                # Hub resistance increases effective half-life
                decay_factor = math.exp(-days_since / (TEMPORAL_DECAY_DAYS * self.hub_resistance))
                base *= decay_factor
            except:
                pass

        return base

@dataclass
class MemorySurface:
    """A memory that has surfaced through spreading activation."""
    node_id: str
    label: str
    score: float
    path: List[str]
    contributing_edges: List[str]

# ============================================================================
# RELATIONAL MEMORY v2
# ============================================================================

class RelationalMemoryV2:
    """
    Enhanced edge-primary memory system with:
    - New concept discovery
    - Temporal decay
    - Edge pruning
    - LVS bidirectional sync
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.outgoing: Dict[str, Set[str]] = defaultdict(set)
        self.incoming: Dict[str, Set[str]] = defaultdict(set)
        self.label_to_id: Dict[str, str] = {}
        self.current_coherence: float = 0.5
        self._load()

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_edge_weight(self, edge, now_dt: datetime = None) -> float:
        """
        Get effective weight from edge, handling both old Edge and BiologicalEdge types.
        """
        if now_dt is None:
            now_dt = datetime.now(timezone.utc)

        if hasattr(edge, 'w_slow'):  # BiologicalEdge
            now_ts = now_dt.timestamp()
            return edge.effective_weight(t_now=now_ts, global_p=self.current_coherence)
        elif hasattr(edge, 'effective_weight'):  # Old Edge
            return edge.effective_weight(now_dt)
        else:
            return getattr(edge, 'weight', 0.5)

    def _load(self):
        """Load memory from disk."""
        if RELATIONAL_MEMORY_PATH.exists():
            try:
                with open(RELATIONAL_MEMORY_PATH) as f:
                    data = json.load(f)

                for node_data in data.get("nodes", []):
                    # Handle old format without new fields
                    if 'discovered_in_context' not in node_data:
                        node_data['discovered_in_context'] = ""
                    if 'lvs_chunk_ids' not in node_data:
                        node_data['lvs_chunk_ids'] = []
                    node = Node(**node_data)
                    self.nodes[node.id] = node
                    self.label_to_id[node.label.lower()] = node.id

                for edge_data in data.get("edges", []):
                    # Remove stale cache if present in old data (CPU fix 2026-01-21)
                    edge_data.pop('_last_activated_dt', None)

                    # Check if BiologicalEdge format (has w_slow) or old Edge format
                    if 'w_slow' in edge_data:
                        # New BiologicalEdge format
                        edge = BiologicalEdge.from_dict(edge_data)
                    else:
                        # Old Edge format - migrate to BiologicalEdge
                        edge = BiologicalEdge(
                            id=edge_data['id'],
                            source_id=edge_data['source_id'],
                            target_id=edge_data['target_id'],
                            relation_type=edge_data.get('relation_type', 'related_to'),
                            w_slow=edge_data.get('weight', 0.5),  # Migrate weight → w_slow
                            w_fast=0.0,
                            tau=0.5,
                            is_h1_locked=False
                        )

                    self.edges[edge.id] = edge
                    self.outgoing[edge.source_id].add(edge.id)
                    self.incoming[edge.target_id].add(edge.id)

                print(f"[MEMORY v2] Loaded {len(self.nodes)} nodes, {len(self.edges)} edges")
            except Exception as e:
                print(f"[MEMORY v2] Error loading: {e}")

    def _save(self):
        """Persist memory to disk with atomic write + file lock."""
        def edge_to_dict(e):
            # Use to_dict() for BiologicalEdge, fallback for old Edge
            if hasattr(e, 'to_dict'):
                return e.to_dict()
            else:
                d = asdict(e)
                d.pop('_last_activated_dt', None)
                return d

        data = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [edge_to_dict(e) for e in self.edges.values()],
            "metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "edge_node_ratio": len(self.edges) / max(len(self.nodes), 1),
                "version": "2.1-biological"
            }
        }

        # Atomic write with file lock to prevent race condition
        # Lock the target file, write to temp, then rename
        temp_path = RELATIONAL_MEMORY_PATH.with_suffix('.tmp')
        
        # Ensure target file exists for locking (create if needed)
        RELATIONAL_MEMORY_PATH.touch(exist_ok=True)
        
        with open(RELATIONAL_MEMORY_PATH, 'a') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with open(temp_path, 'w') as f:
                    json.dump(data, f)  # No indent to save space
                temp_path.replace(RELATIONAL_MEMORY_PATH)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    # ========================================================================
    # NODE OPERATIONS - ENHANCED with discovery
    # ========================================================================

    def get_or_create_node(self, label: str, node_type: str = "concept", context: str = "") -> Node:
        """Get existing node or create new anchor point."""
        label_lower = label.lower().strip()

        if label_lower in self.label_to_id:
            return self.nodes[self.label_to_id[label_lower]]

        # CREATE NEW NODE - This is the fix for concept discovery
        node_id = self._generate_id(f"node:{label_lower}")
        node = Node(
            id=node_id,
            label=label_lower,
            node_type=node_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            discovered_in_context=context
        )

        self.nodes[node_id] = node
        self.label_to_id[label_lower] = node_id
        return node

    def link_node_to_lvs(self, node_id: str, chunk_id: str) -> bool:
        """
        Link a RIE node to an LVS vector chunk.
        
        This enables hybrid retrieval: graph traversal finds concepts,
        then we retrieve the actual vector chunks for those concepts.
        
        Args:
            node_id: The RIE node ID
            chunk_id: The LVS chunk ID (from LVS_INDEX.json)
        
        Returns:
            True if linked successfully, False if node not found
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if chunk_id not in node.lvs_chunk_ids:
            node.lvs_chunk_ids.append(chunk_id)
            self._save()
        return True

    # ========================================================================
    # EDGE OPERATIONS - ENHANCED with ceiling and decay
    # ========================================================================

    def create_relation(
        self,
        source_label: str,
        target_label: str,
        relation_type: str = "related_to",
        context: str = "",
        weight: float = 1.0
    ) -> BiologicalEdge:
        """Create or strengthen a relation with BiologicalEdge."""
        source = self.get_or_create_node(source_label, context=context)
        target = self.get_or_create_node(target_label, context=context)

        edge_id = f"{source.id}→{target.id}"

        if edge_id in self.edges:
            # Edge exists - strengthen it via STDP
            edge = self.edges[edge_id]
            t_now = datetime.now(timezone.utc).timestamp()
            t_pre = t_now - 0.01  # Pre-synaptic 10ms before post (causal)
            edge.update_stdp(t_pre=t_pre, t_post=t_now, outcome_signal=1.0)  # Positive reinforcement
            edge.spike_count = getattr(edge, 'spike_count', 0) + 1
            edge.last_spike = datetime.now(timezone.utc).timestamp()
        else:
            # Create new BiologicalEdge
            edge = BiologicalEdge(
                id=edge_id,
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                w_slow=0.5,      # Moderate initial strength
                w_fast=0.0,      # No short-term plasticity yet
                tau=0.5,         # Neutral trust
                is_h1_locked=False
            )
            edge.spike_count = 1
            edge.last_spike = datetime.now(timezone.utc).timestamp()

            self.edges[edge_id] = edge
            self.outgoing[source.id].add(edge_id)
            self.incoming[target.id].add(edge_id)

        # Calculate hub resistance based on node connectivity
        # Hub edges (connected to high-degree nodes) decay slower
        source_degree = len(self.outgoing.get(source.id, set())) + len(self.incoming.get(source.id, set()))
        target_degree = len(self.outgoing.get(target.id, set())) + len(self.incoming.get(target.id, set()))
        max_degree = max(source_degree, target_degree)
        
        # hub_resistance = 1.0 + 0.1 * log(1 + max_degree), capped at 3.0
        # Peripheral (degree ~1): 1.0, Hub (degree ~100): ~1.5, Major hub (degree ~1000): ~2.0
        edge.hub_resistance = min(3.0, 1.0 + 0.1 * math.log(1 + max_degree))

        return edge

    def create_bidirectional(
        self,
        label1: str,
        label2: str,
        relation_type: str = "related_to",
        context: str = "",
        weight: float = 1.0
    ) -> Tuple[Edge, Edge]:
        """Create relations in both directions."""
        e1 = self.create_relation(label1, label2, relation_type, context, weight)
        e2 = self.create_relation(label2, label1, relation_type, context, weight)
        return e1, e2

    # ========================================================================
    # STDP METHODS - BiologicalEdge Integration
    # ========================================================================

    def update_path_stdp(self, activated_edge_ids: list, outcome_signal: float):
        """
        Apply STDP to a sequence of edges based on outcome.

        Args:
            activated_edge_ids: List of edge IDs that fired in sequence
            outcome_signal: +1.0 for success, -1.0 for failure
        """
        import time
        t_now = time.time()

        for i, edge_id in enumerate(activated_edge_ids):
            edge = self.edges.get(edge_id)
            if edge and hasattr(edge, 'update_stdp') and not getattr(edge, 'is_h1_locked', False):
                # Simulate causal timing: earlier edges fired first
                t_pre = t_now - (len(activated_edge_ids) - i) * 0.01  # 10ms apart
                t_post = t_pre + 0.005  # 5ms after (within STDP window)
                edge.update_stdp(t_pre=t_pre, t_post=t_post, outcome_signal=outcome_signal, t_now=t_now)

    def consolidate_edges(self, tau_threshold: float = 0.7):
        """
        Consolidate w_fast → w_slow for all edges.
        Call periodically (e.g., every 100 turns).
        """
        for edge in self.edges.values():
            if hasattr(edge, 'consolidate') and not getattr(edge, 'is_h1_locked', False):
                edge.consolidate(tau_gate=tau_threshold)

        # Apply homeostasis
        for node_id in self.nodes.keys():
            self.normalize_node_edges(node_id)

    def protect_h1_cycles(self, cycle_edge_ids: list):
        """
        Lock edges that form critical cycles (H₁ topology).
        These become permanent and resist plasticity.
        """
        for edge_id in cycle_edge_ids:
            edge = self.edges.get(edge_id)
            if edge and hasattr(edge, 'is_h1_locked'):
                edge.is_h1_locked = True

    def normalize_node_edges(self, node_id: str, target_strength: float = 10.0):
        """
        Per-node synaptic scaling (homeostasis).
        Prevents runaway potentiation.
        """
        outgoing_ids = self.outgoing.get(node_id, set())
        outgoing = [self.edges[eid] for eid in outgoing_ids
                    if eid in self.edges and not getattr(self.edges[eid], 'is_h1_locked', False)]

        if not outgoing:
            return

        total = sum(getattr(e, 'w_slow', getattr(e, 'weight', 1.0)) for e in outgoing)

        if total > target_strength:
            scale = target_strength / total
            for edge in outgoing:
                if hasattr(edge, 'w_slow'):
                    edge.w_slow *= scale

    # ========================================================================
    # PRUNING - NEW in v2
    # ========================================================================

    def prune_weak_edges(self, dry_run: bool = True) -> int:
        """
        Remove edges that are:
        - Below minimum weight AND
        - Not activated in MIN_ACTIVATION_PRUNE days AND
        - Low coherence at formation (p < 0.5)
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=MIN_ACTIVATION_PRUNE)

        edges_to_prune = []

        for edge_id, edge in self.edges.items():
            # Check weight after decay
            effective = self._get_edge_weight(edge, now)

            if effective < MIN_WEIGHT_PRUNE:
                # Check last activation
                try:
                    last = datetime.fromisoformat(edge.last_activated.replace('Z', '+00:00'))
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)

                    if last < cutoff and edge.coherence_at_formation < 0.5:
                        edges_to_prune.append(edge_id)
                except:
                    pass

        if not dry_run:
            for edge_id in edges_to_prune:
                edge = self.edges[edge_id]
                self.outgoing[edge.source_id].discard(edge_id)
                self.incoming[edge.target_id].discard(edge_id)
                del self.edges[edge_id]
            self._save()

        return len(edges_to_prune)

    def remove_edge(self, edge_id: str) -> bool:
        """
        Remove a single edge and clean up references.
        
        Args:
            edge_id: The ID of the edge to remove
            
        Returns:
            True if edge was removed, False if not found
        """
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        
        # Clean up incoming/outgoing sets
        if edge.source_id in self.outgoing:
            self.outgoing[edge.source_id].discard(edge_id)
        if edge.target_id in self.incoming:
            self.incoming[edge.target_id].discard(edge_id)
        
        del self.edges[edge_id]
        return True

    # ========================================================================
    # SPREADING ACTIVATION - Enhanced with path tracking
    # ========================================================================

    def spread_activation(
        self,
        seed_labels: List[str],
        iterations: int = SPREAD_ITERATIONS,
        decay: float = ACTIVATION_DECAY
    ) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Spread activation with path tracking and edge tracking.
        Returns (activation_map, paths_map, edges_used_map)

        TOPOLOGY TRAINING INSTRUMENTATION (2026-01-21):
        Now tracks which edges were activated to reach each node.
        This enables gradient-based topology training.
        """
        activation: Dict[str, float] = {}
        paths: Dict[str, List[str]] = {}  # Track how we got here
        edges_used: Dict[str, List[str]] = {}  # NEW: Track which edges activated each node
        now = datetime.now(timezone.utc)

        for label in seed_labels:
            label_lower = label.lower().strip()
            if label_lower in self.label_to_id:
                node_id = self.label_to_id[label_lower]
                activation[node_id] = 1.0
                paths[node_id] = [label_lower]
                edges_used[node_id] = []  # Seed nodes have no incoming edges

        if not activation:
            return {}, {}, {}

        for i in range(iterations):
            new_activation: Dict[str, float] = {}

            for node_id, act in activation.items():
                # CPU FIX 2026-01-21: Don't propagate from weak nodes
                if act < 0.01:  # Prune nodes with <1% activation
                    continue

                for edge_id in self.outgoing.get(node_id, set()):
                    edge = self.edges[edge_id]
                    target_id = edge.target_id
                    spread = act * self._get_edge_weight(edge, now) * decay

                    if target_id not in new_activation or new_activation[target_id] < spread:
                        new_activation[target_id] = spread
                        # Track path
                        if node_id in paths:
                            paths[target_id] = paths[node_id] + [self.nodes[target_id].label]
                        # Track edge used (TOPOLOGY TRAINING)
                        if target_id not in edges_used:
                            edges_used[target_id] = []
                        if edge_id not in edges_used[target_id]:
                            edges_used[target_id].append(edge_id)

                for edge_id in self.incoming.get(node_id, set()):
                    edge = self.edges[edge_id]
                    source_id = edge.source_id
                    spread = act * self._get_edge_weight(edge, now) * decay * 0.7  # More balanced reverse

                    if source_id not in new_activation or new_activation[source_id] < spread:
                        new_activation[source_id] = spread
                        if node_id in paths:
                            paths[source_id] = paths[node_id] + [self.nodes[source_id].label]
                        # Track edge used (TOPOLOGY TRAINING)
                        if source_id not in edges_used:
                            edges_used[source_id] = []
                        if edge_id not in edges_used[source_id]:
                            edges_used[source_id].append(edge_id)

            # CPU FIX 2026-01-21: Limit active nodes to prevent exponential blowup
            # Keep only top 500 most activated nodes per iteration
            if len(new_activation) > 500:
                sorted_activation = sorted(new_activation.items(), key=lambda x: -x[1])
                new_activation = dict(sorted_activation[:500])

            for node_id, act in new_activation.items():
                activation[node_id] = activation.get(node_id, 0) + act

        return activation, paths, edges_used

    def surface_memories(
        self,
        query: str,
        max_results: int = 5
    ) -> List[MemorySurface]:
        """Surface relevant memories with path information and edge tracking."""
        concepts = self._extract_concepts(query, discover_new=False)

        if not concepts:
            return []

        activation, paths, edges_used = self.spread_activation(concepts)

        sorted_nodes = sorted(
            [(nid, act) for nid, act in activation.items() if act > MIN_ACTIVATION_THRESHOLD],
            key=lambda x: -x[1]
        )[:max_results]

        results = []
        for node_id, score in sorted_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                results.append(MemorySurface(
                    node_id=node_id,
                    label=node.label,
                    score=score,
                    path=paths.get(node_id, []),
                    contributing_edges=edges_used.get(node_id, [])  # TOPOLOGY TRAINING
                ))

        return results

    def retrieve_with_graph_context(self, query: str, top_k: int = 5) -> Dict:
        """
        Hybrid retrieval: spread activation + follow LVS links.
        
        This is the bridge between relational memory (RIE) and vector memory (LVS).
        
        1. Run spread_activation() to find relevant RIE nodes
        2. Collect lvs_chunk_ids from activated nodes
        3. Return both graph context and vector chunk IDs
        
        Args:
            query: Natural language query
            top_k: Maximum number of memories to surface
        
        Returns:
            Dict with:
            - memories: List of MemorySurface objects
            - lvs_chunk_ids: Deduplicated list of LVS chunk IDs to retrieve
            - graph_context: Labels of activated nodes for context
        """
        memories = self.surface_memories(query, max_results=top_k)
        
        chunk_ids = []
        for mem in memories:
            node = self.nodes.get(mem.node_id)
            if node and node.lvs_chunk_ids:
                chunk_ids.extend(node.lvs_chunk_ids)
        
        return {
            "memories": memories,
            "lvs_chunk_ids": list(set(chunk_ids)),  # Deduplicate
            "graph_context": [m.label for m in memories]
        }

    # ========================================================================
    # CONCEPT EXTRACTION - ENHANCED with new concept discovery
    # ========================================================================

    def _extract_concepts(self, text: str, discover_new: bool = True) -> List[str]:
        """
        Extract concepts from text.

        If discover_new=True, creates nodes for unknown words.
        This is the FIX for the concept discovery gap.
        """
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stopwords and short words
        filtered = [
            w for w in words
            if len(w) >= CONCEPT_MIN_LENGTH and w not in STOPWORDS
        ]

        concepts = []
        for word in filtered:
            if word in self.label_to_id:
                concepts.append(word)
            elif discover_new:
                # CREATE NEW CONCEPT - This is the key fix
                self.get_or_create_node(word, context=f"Discovered in: {text[:100]}")
                concepts.append(word)

        # Check for multi-word concepts
        text_lower = text.lower()
        for label in self.label_to_id.keys():
            if ' ' in label and label in text_lower:
                concepts.append(label)

        return list(set(concepts))

    # ========================================================================
    # LEARNING FROM TEXT - ENHANCED
    # ========================================================================

    def learn_from_text(self, text: str, context: str = "", discover_new: bool = True) -> int:
        """
        Learn relations from text with new concept discovery.
        
        TODO: When creating nodes for concepts, check if there's a matching
        LVS chunk and link them automatically. This would enable automatic
        hybrid retrieval for learned content.
        Example: lvs_chunk = find_lvs_chunk_for_concept(concept)
                 if lvs_chunk:
                     self.link_node_to_lvs(node.id, lvs_chunk)
        """
        # Extract concepts, creating new ones if discover_new=True
        concepts = self._extract_concepts(text, discover_new=discover_new)

        relations_created = 0
        window_size = 5

        for i, concept1 in enumerate(concepts):
            for j in range(i + 1, min(i + window_size, len(concepts))):
                concept2 = concepts[j]
                if concept1 != concept2:
                    distance = j - i
                    weight = 1.0 / distance

                    self.create_bidirectional(
                        concept1,
                        concept2,
                        relation_type="co_occurs",
                        context=context,
                        weight=weight
                    )
                    relations_created += 1

        self._save()
        return relations_created

    def set_coherence(self, p: float):
        """Update current coherence."""
        self.current_coherence = p

    # ========================================================================
    # LVS BIDIRECTIONAL SYNC - NEW in v2
    # ========================================================================

    def sync_with_lvs(self) -> Dict[str, int]:
        """
        Bidirectional sync with LVS index.

        1. Load new files from LVS that aren't in memory
        2. Extract concepts and create relations

        Returns stats about what was synced.
        """
        if not LVS_INDEX_PATH.exists():
            return {"error": "No LVS index found"}

        try:
            with open(LVS_INDEX_PATH) as f:
                index = json.load(f)

            nodes = index.get("nodes", [])
            new_docs = 0
            new_relations = 0

            for node in nodes:
                path = node.get("path", "")
                summary = node.get("summary", "")

                if not path or not summary:
                    continue

                doc_label = Path(path).stem.replace("_", " ").replace("-", " ").lower()

                # Check if we already have this document
                if doc_label in self.label_to_id:
                    continue

                # NEW document - add it
                self.get_or_create_node(doc_label, node_type="document", context=f"LVS sync: {path}")
                new_docs += 1

                # Extract concepts from summary and link
                concepts = self._extract_concepts(summary, discover_new=True)

                for concept in set(concepts):
                    if concept != doc_label:
                        self.create_relation(
                            concept,
                            doc_label,
                            relation_type="mentioned_in",
                            context="LVS sync",
                            weight=0.5
                        )
                        new_relations += 1

            self._save()
            return {"new_documents": new_docs, "new_relations": new_relations}

        except Exception as e:
            return {"error": str(e)}

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def stats(self) -> Dict[str, Any]:
        if not self.nodes:
            return {"nodes": 0, "edges": 0, "ratio": 0, "avg_coherence": 0}

        now = datetime.now(timezone.utc).timestamp()  # Use timestamp for BiologicalEdge
        current_p = getattr(self, 'global_coherence', {}).get('p', 0.7)

        # Calculate average effective weight (with decay applied)
        # BiologicalEdge uses t_now (float), old Edge may use datetime
        def get_effective(e):
            if hasattr(e, 'w_slow'):  # BiologicalEdge
                return e.effective_weight(t_now=now, global_p=current_p)
            elif hasattr(e, 'effective_weight'):  # Old Edge with datetime signature
                try:
                    return e.effective_weight(datetime.fromtimestamp(now, tz=timezone.utc))
                except:
                    return getattr(e, 'weight', 0.5)
            else:
                return getattr(e, 'weight', 0.5)

        total_effective = sum(get_effective(e) for e in self.edges.values())
        avg_effective = total_effective / max(len(self.edges), 1)

        # BiologicalEdge uses tau for trust, old Edge uses coherence_at_formation
        avg_coherence = sum(
            getattr(e, 'coherence_at_formation', getattr(e, 'tau', 0.5))
            for e in self.edges.values()
        ) / max(len(self.edges), 1)

        total_activations = sum(
            getattr(e, 'activation_count', getattr(e, 'spike_count', 0))
            for e in self.edges.values()
        )

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "ratio": len(self.edges) / len(self.nodes),
            "avg_coherence": round(avg_coherence, 3),
            "avg_effective_weight": round(avg_effective, 3),
            "total_activations": total_activations
        }

    def format_stats(self) -> str:
        s = self.stats()
        return f"""
╔══════════════════════════════════════╗
║     RIE RELATIONAL MEMORY v2         ║
╠══════════════════════════════════════╣
║  Nodes (anchors):   {s['nodes']:6}           ║
║  Edges (relations): {s['edges']:6}           ║
║  Edge:Node ratio:   {s['ratio']:6.1f}           ║
╠══════════════════════════════════════╣
║  Avg coherence:     {s['avg_coherence']:.3f}           ║
║  Avg effective wt:  {s.get('avg_effective_weight', 0):.3f}           ║
║  Total activations: {s['total_activations']:6}           ║
╚══════════════════════════════════════╝
"""


# ============================================================================
# MIGRATION: Convert v1 to v2
# ============================================================================

def migrate_v1_to_v2():
    """Migrate existing v1 memory to v2 format."""
    print("Migrating RIE Relational Memory v1 -> v2...")

    memory = RelationalMemoryV2()

    # Prune weak edges (dry run first)
    prune_count = memory.prune_weak_edges(dry_run=True)
    print(f"  Would prune {prune_count} weak edges")

    # Sync with LVS
    sync_result = memory.sync_with_lvs()
    print(f"  LVS sync: {sync_result}")

    print(memory.format_stats())
    return memory


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    memory = RelationalMemoryV2()

    if len(sys.argv) < 2:
        print("RIE RELATIONAL MEMORY v2")
        print("=" * 40)
        print("Enhanced edge-primary memory")
        print()
        print("Commands:")
        print("  stats              - Show memory statistics")
        print("  sync               - Sync with LVS index (bidirectional)")
        print("  prune              - Prune weak/old edges")
        print("  learn <text>       - Learn relations from text")
        print("  query <text>       - Surface memories for query")
        print("  relate <a> <b>     - Create relation between concepts")
        print("  migrate            - Migrate from v1 to v2")
        print()
        print(memory.format_stats())
        return

    cmd = sys.argv[1]

    if cmd == "stats":
        print(memory.format_stats())

    elif cmd == "sync":
        print("Syncing with LVS index...")
        result = memory.sync_with_lvs()
        print(f"Result: {result}")
        print(memory.format_stats())

    elif cmd == "prune":
        dry_run = "--confirm" not in sys.argv
        count = memory.prune_weak_edges(dry_run=dry_run)
        if dry_run:
            print(f"Would prune {count} edges. Run with --confirm to apply.")
        else:
            print(f"Pruned {count} edges.")
        print(memory.format_stats())

    elif cmd == "learn" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        count = memory.learn_from_text(text, context="CLI input", discover_new=True)
        print(f"Learned {count} relations from text (with new concept discovery)")
        print(memory.format_stats())

    elif cmd == "query" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"Query: {query}")
        print("-" * 40)

        results = memory.surface_memories(query)
        if not results:
            print("No memories surfaced")
        else:
            for mem in results:
                path_str = " -> ".join(mem.path) if mem.path else "direct"
                print(f"  {mem.label}: {mem.score:.3f} (path: {path_str})")

    elif cmd == "relate" and len(sys.argv) >= 4:
        a, b = sys.argv[2], sys.argv[3]
        memory.create_bidirectional(a, b, context="CLI relation")
        print(f"Created relation: {a} <-> {b}")
        memory._save()

    elif cmd == "migrate":
        migrate_v1_to_v2()

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
