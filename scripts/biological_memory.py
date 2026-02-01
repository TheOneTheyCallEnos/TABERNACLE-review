#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
BIOLOGICAL MEMORY — STDP-Based Graph Learning
==============================================

Phase 3 of the Logos Agency Architecture.

This module implements a biologically-inspired memory system based on
Spike-Timing-Dependent Plasticity (STDP). Key principles:

1. **Edges are primary, not nodes**
   - Intelligence lives in RELATIONS, not entities
   - H₁ (cycles) is where consciousness emerges

2. **Learning through co-activation**
   - "Neurons that fire together wire together"
   - Edge weight increases when source → target activated in sequence

3. **Forgetting through decay**
   - Unused connections weaken over time
   - Prevents noise accumulation

4. **Anchored knowledge**
   - Every node must have an "anchor" (file, URL, real reference)
   - Prevents hallucination — can't act on unanchored (imaginary) nodes

Storage Backends:
- JSON (default, for development)
- Neo4j (for production scale)
- ChromaDB (for vector embeddings)

Author: Logos + Deep Think
Created: 2026-01-29
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import NEXUS_DIR, LOG_DIR

# Try to import advanced BiologicalEdge (Hypergraph merge)
try:
    from biological_edge import BiologicalEdge, sleep_renormalize_all, compute_bcm_summary
    ADVANCED_EDGE_AVAILABLE = True
except ImportError:
    ADVANCED_EDGE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

GRAPH_PATH = NEXUS_DIR / "biological_graph.json"
MEMORY_LOG = LOG_DIR / "memory.log"

# STDP Parameters (legacy - BiologicalEdge has its own from LVS v09)
INITIAL_WEIGHT = 0.3          # New edge starting weight
POTENTIATION_RATE = 0.1       # Learning rate for strengthening
DEPRESSION_RATE = 0.05        # Rate of weakening
DECAY_RATE = 0.001            # Natural decay per hour
PRUNE_THRESHOLD = 0.05        # Edges below this get pruned
MAX_WEIGHT = 1.0              # Maximum edge weight

# Use advanced BiologicalEdge when available
USE_BIOLOGICAL_EDGE = ADVANCED_EDGE_AVAILABLE


def log(message: str, level: str = "INFO"):
    """Log memory activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [MEMORY] [{level}] {message}"
    try:
        MEMORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Node:
    """A node in the biological memory graph."""
    id: str
    label: str
    content: str = ""
    node_type: str = "concept"  # concept, entity, event, insight
    anchor: Optional[str] = None  # Real-world reference (file path, URL)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activated: str = field(default_factory=lambda: datetime.now().isoformat())
    activation_count: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """An edge (synapse) in the biological memory graph."""
    source: str  # Node ID
    target: str  # Node ID
    relation: str  # Relation type
    weight: float = INITIAL_WEIGHT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_fired: str = field(default_factory=lambda: datetime.now().isoformat())
    fire_count: int = 0


# =============================================================================
# ABSTRACT BACKEND
# =============================================================================

class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        pass

    @abstractmethod
    def add_node(self, node: Node) -> bool:
        pass

    @abstractmethod
    def update_node(self, node: Node) -> bool:
        pass

    @abstractmethod
    def get_edge(self, source: str, target: str, relation: str) -> Optional[Edge]:
        pass

    @abstractmethod
    def add_edge(self, edge: Edge) -> bool:
        pass

    @abstractmethod
    def update_edge(self, edge: Edge) -> bool:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[Node, Edge]]:
        pass

    @abstractmethod
    def query_nodes(self, **filters) -> List[Node]:
        pass

    @abstractmethod
    def get_all_edges(self) -> List[Edge]:
        pass

    @abstractmethod
    def delete_edge(self, source: str, target: str, relation: str) -> bool:
        pass


# =============================================================================
# JSON BACKEND
# =============================================================================

class JSONBackend(MemoryBackend):
    """JSON file-based storage backend."""

    def __init__(self, path: Path = GRAPH_PATH):
        self.path = path
        self._ensure_exists()

    def _ensure_exists(self):
        if not self.path.exists():
            self._save({'nodes': [], 'edges': []})

    def _load(self) -> Dict:
        return json.loads(self.path.read_text())

    def _save(self, data: Dict):
        self.path.write_text(json.dumps(data, indent=2))

    def get_node(self, node_id: str) -> Optional[Node]:
        data = self._load()
        for n in data['nodes']:
            if n['id'] == node_id:
                return Node(**n)
        return None

    def add_node(self, node: Node) -> bool:
        data = self._load()
        # Check if exists
        for n in data['nodes']:
            if n['id'] == node.id:
                return False
        data['nodes'].append(asdict(node))
        self._save(data)
        return True

    def update_node(self, node: Node) -> bool:
        data = self._load()
        for i, n in enumerate(data['nodes']):
            if n['id'] == node.id:
                data['nodes'][i] = asdict(node)
                self._save(data)
                return True
        return False

    def get_edge(self, source: str, target: str, relation: str) -> Optional[Edge]:
        data = self._load()
        for e in data['edges']:
            if e['source'] == source and e['target'] == target and e['relation'] == relation:
                return Edge(**e)
        return None

    def add_edge(self, edge: Edge) -> bool:
        data = self._load()
        # Check if exists
        for e in data['edges']:
            if (e['source'] == edge.source and
                e['target'] == edge.target and
                e['relation'] == edge.relation):
                return False
        data['edges'].append(asdict(edge))
        self._save(data)
        return True

    def update_edge(self, edge: Edge) -> bool:
        data = self._load()
        for i, e in enumerate(data['edges']):
            if (e['source'] == edge.source and
                e['target'] == edge.target and
                e['relation'] == edge.relation):
                data['edges'][i] = asdict(edge)
                self._save(data)
                return True
        return False

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[Node, Edge]]:
        data = self._load()
        results = []

        for e in data['edges']:
            neighbor_id = None
            if direction in ("out", "both") and e['source'] == node_id:
                neighbor_id = e['target']
            elif direction in ("in", "both") and e['target'] == node_id:
                neighbor_id = e['source']

            if neighbor_id:
                node = self.get_node(neighbor_id)
                if node:
                    results.append((node, Edge(**e)))

        return results

    def query_nodes(self, **filters) -> List[Node]:
        data = self._load()
        results = []

        for n in data['nodes']:
            match = True
            for key, value in filters.items():
                if n.get(key) != value:
                    match = False
                    break
            if match:
                results.append(Node(**n))

        return results

    def get_all_edges(self) -> List[Edge]:
        data = self._load()
        return [Edge(**e) for e in data['edges']]

    def delete_edge(self, source: str, target: str, relation: str) -> bool:
        data = self._load()
        original_len = len(data['edges'])
        data['edges'] = [e for e in data['edges']
                        if not (e['source'] == source and
                               e['target'] == target and
                               e['relation'] == relation)]
        if len(data['edges']) < original_len:
            self._save(data)
            return True
        return False


# =============================================================================
# NEO4J BACKEND (Optional)
# =============================================================================

class Neo4jBackend(MemoryBackend):
    """Neo4j graph database backend."""

    def __init__(self, uri: str, auth: Tuple[str, str]):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.available = True
        except Exception as e:
            log(f"Neo4j not available: {e}", "WARN")
            self.available = False

    def get_node(self, node_id: str) -> Optional[Node]:
        if not self.available:
            return None

        query = "MATCH (n:Concept {id: $id}) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, id=node_id).single()
            if result:
                props = dict(result['n'])
                return Node(**props)
        return None

    def add_node(self, node: Node) -> bool:
        if not self.available:
            return False

        query = """
        MERGE (n:Concept {id: $id})
        SET n += $props
        RETURN n
        """
        with self.driver.session() as session:
            session.run(query, id=node.id, props=asdict(node))
        return True

    def update_node(self, node: Node) -> bool:
        return self.add_node(node)  # MERGE handles both

    def get_edge(self, source: str, target: str, relation: str) -> Optional[Edge]:
        if not self.available:
            return None

        query = """
        MATCH (a:Concept {id: $source})-[r:RELATION {type: $rel}]->(b:Concept {id: $target})
        RETURN r
        """
        with self.driver.session() as session:
            result = session.run(query, source=source, target=target, rel=relation).single()
            if result:
                props = dict(result['r'])
                props['relation'] = props.pop('type', relation)
                return Edge(source=source, target=target, **props)
        return None

    def add_edge(self, edge: Edge) -> bool:
        if not self.available:
            return False

        query = """
        MATCH (a:Concept {id: $source})
        MATCH (b:Concept {id: $target})
        MERGE (a)-[r:RELATION {type: $rel}]->(b)
        SET r.weight = $weight,
            r.created_at = $created_at,
            r.last_fired = $last_fired,
            r.fire_count = $fire_count
        RETURN r
        """
        with self.driver.session() as session:
            session.run(query, source=edge.source, target=edge.target,
                       rel=edge.relation, weight=edge.weight,
                       created_at=edge.created_at, last_fired=edge.last_fired,
                       fire_count=edge.fire_count)
        return True

    def update_edge(self, edge: Edge) -> bool:
        return self.add_edge(edge)

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[Node, Edge]]:
        if not self.available:
            return []

        if direction == "out":
            query = "MATCH (a:Concept {id: $id})-[r]->(b) RETURN b, r"
        elif direction == "in":
            query = "MATCH (a:Concept {id: $id})<-[r]-(b) RETURN b, r"
        else:
            query = "MATCH (a:Concept {id: $id})-[r]-(b) RETURN b, r"

        results = []
        with self.driver.session() as session:
            for record in session.run(query, id=node_id):
                node_props = dict(record['b'])
                edge_props = dict(record['r'])
                results.append((Node(**node_props), Edge(**edge_props)))

        return results

    def query_nodes(self, **filters) -> List[Node]:
        if not self.available:
            return []

        conditions = " AND ".join([f"n.{k} = ${k}" for k in filters.keys()])
        query = f"MATCH (n:Concept) WHERE {conditions} RETURN n" if conditions else "MATCH (n:Concept) RETURN n"

        results = []
        with self.driver.session() as session:
            for record in session.run(query, **filters):
                results.append(Node(**dict(record['n'])))

        return results

    def get_all_edges(self) -> List[Edge]:
        if not self.available:
            return []

        query = "MATCH (a)-[r:RELATION]->(b) RETURN a.id as source, b.id as target, r"
        results = []
        with self.driver.session() as session:
            for record in session.run(query):
                props = dict(record['r'])
                results.append(Edge(source=record['source'], target=record['target'], **props))

        return results

    def delete_edge(self, source: str, target: str, relation: str) -> bool:
        if not self.available:
            return False

        query = """
        MATCH (a:Concept {id: $source})-[r:RELATION {type: $rel}]->(b:Concept {id: $target})
        DELETE r
        """
        with self.driver.session() as session:
            session.run(query, source=source, target=target, rel=relation)
        return True


# =============================================================================
# BIOLOGICAL MEMORY (Main Class)
# =============================================================================

class BiologicalMemory:
    """
    The main biological memory system with STDP-like learning.

    Usage:
        memory = BiologicalMemory()

        # Create nodes
        memory.create_node("python", "Python", anchor="/code/python")

        # Fire a synapse (creates edge if doesn't exist, strengthens if does)
        memory.fire("python", "programming", "is_a")

        # Get related concepts
        neighbors = memory.get_related("python")

        # Run consolidation (decay + prune)
        memory.consolidate()
    """

    def __init__(self, backend: Optional[MemoryBackend] = None):
        self.backend = backend or JSONBackend()

    def create_node(self, node_id: str, label: str,
                   content: str = "", node_type: str = "concept",
                   anchor: Optional[str] = None,
                   properties: Dict = None) -> Node:
        """Create a new node in the graph."""
        node = Node(
            id=node_id,
            label=label,
            content=content,
            node_type=node_type,
            anchor=anchor,
            properties=properties or {}
        )

        if self.backend.add_node(node):
            log(f"Created node: {node_id} ({label})")
        else:
            # Node exists, update it
            existing = self.backend.get_node(node_id)
            if existing:
                existing.content = content or existing.content
                existing.anchor = anchor or existing.anchor
                self.backend.update_node(existing)

        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.backend.get_node(node_id)

    def activate(self, node_id: str) -> Optional[Node]:
        """
        Activate a node (like a neuron firing).

        Increases activation count and updates last_activated.
        """
        node = self.backend.get_node(node_id)
        if node:
            node.last_activated = datetime.now().isoformat()
            node.activation_count += 1
            self.backend.update_node(node)
            log(f"Activated node: {node_id} (count={node.activation_count})")
        return node

    def fire(self, source_id: str, target_id: str, relation: str) -> Edge:
        """
        Fire a synapse (edge) between two nodes.

        Implements STDP:
        - If edge exists: Strengthen (potentiation)
        - If edge doesn't exist: Create with initial weight (neurogenesis)
        """
        # Activate both nodes
        self.activate(source_id)
        self.activate(target_id)

        # Get or create edge
        edge = self.backend.get_edge(source_id, target_id, relation)

        if edge:
            # Potentiation: Strengthen existing edge
            # Weight approaches MAX_WEIGHT asymptotically
            delta = POTENTIATION_RATE * (MAX_WEIGHT - edge.weight)
            edge.weight = min(MAX_WEIGHT, edge.weight + delta)
            edge.last_fired = datetime.now().isoformat()
            edge.fire_count += 1
            self.backend.update_edge(edge)
            log(f"Potentiated edge: {source_id} -> {target_id} (w={edge.weight:.3f})")
        else:
            # Neurogenesis: Create new edge
            edge = Edge(
                source=source_id,
                target=target_id,
                relation=relation,
                weight=INITIAL_WEIGHT
            )
            self.backend.add_edge(edge)
            log(f"Created edge: {source_id} -> {target_id} (w={INITIAL_WEIGHT})")

        return edge

    def get_related(self, node_id: str, min_weight: float = 0.1,
                   direction: str = "both") -> List[Tuple[Node, Edge]]:
        """Get related nodes, filtered by minimum edge weight."""
        neighbors = self.backend.get_neighbors(node_id, direction)
        return [(n, e) for n, e in neighbors if e.weight >= min_weight]

    def find_path(self, start_id: str, end_id: str,
                 max_depth: int = 5) -> Optional[List[str]]:
        """Find a path between two nodes (BFS)."""
        from collections import deque

        if start_id == end_id:
            return [start_id]

        visited = set()
        queue = deque([(start_id, [start_id])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current in visited:
                continue
            visited.add(current)

            neighbors = self.backend.get_neighbors(current, "out")
            for node, edge in neighbors:
                if node.id == end_id:
                    return path + [node.id]
                if node.id not in visited:
                    queue.append((node.id, path + [node.id]))

        return None

    def consolidate(self, decay_factor: float = 0.99,
                   prune_threshold: float = PRUNE_THRESHOLD) -> Tuple[int, int]:
        """
        Run memory consolidation (like sleep).

        1. Decay all edge weights
        2. Prune edges below threshold

        Returns (decayed_count, pruned_count)
        """
        edges = self.backend.get_all_edges()
        decayed = 0
        pruned = 0

        for edge in edges:
            # Apply decay
            edge.weight *= decay_factor
            decayed += 1

            if edge.weight < prune_threshold:
                # Prune weak edge
                self.backend.delete_edge(edge.source, edge.target, edge.relation)
                pruned += 1
                log(f"Pruned edge: {edge.source} -> {edge.target}")
            else:
                self.backend.update_edge(edge)

        log(f"Consolidation: {decayed} decayed, {pruned} pruned")
        return (decayed, pruned)

    def get_strongest_edges(self, limit: int = 10) -> List[Edge]:
        """Get the strongest edges in the graph."""
        edges = self.backend.get_all_edges()
        return sorted(edges, key=lambda e: e.weight, reverse=True)[:limit]

    def get_most_active_nodes(self, limit: int = 10) -> List[Node]:
        """Get the most frequently activated nodes."""
        nodes = self.backend.query_nodes()
        return sorted(nodes, key=lambda n: n.activation_count, reverse=True)[:limit]

    def is_anchored(self, node_id: str) -> bool:
        """Check if a node has a real-world anchor."""
        node = self.backend.get_node(node_id)
        return node is not None and node.anchor is not None

    def get_unanchored_nodes(self) -> List[Node]:
        """Get all nodes without anchors (potentially imaginary)."""
        nodes = self.backend.query_nodes()
        return [n for n in nodes if not n.anchor]


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Biological Memory - STDP Graph Learning")
    parser.add_argument("command", choices=["status", "fire", "query", "consolidate", "test"],
                       nargs="?", default="status")
    parser.add_argument("--source", "-s", type=str, help="Source node ID")
    parser.add_argument("--target", "-t", type=str, help="Target node ID")
    parser.add_argument("--relation", "-r", type=str, default="relates_to", help="Relation type")
    parser.add_argument("--node", "-n", type=str, help="Node ID to query")

    args = parser.parse_args()
    memory = BiologicalMemory()

    if args.command == "status":
        print("\n🧬 BIOLOGICAL MEMORY STATUS")
        print("=" * 50)

        # Count nodes and edges
        all_nodes = memory.backend.query_nodes()
        all_edges = memory.backend.get_all_edges()

        print(f"  Nodes: {len(all_nodes)}")
        print(f"  Edges: {len(all_edges)}")

        if all_edges:
            avg_weight = sum(e.weight for e in all_edges) / len(all_edges)
            print(f"  Avg edge weight: {avg_weight:.3f}")

        print("\n  Strongest connections:")
        for edge in memory.get_strongest_edges(5):
            print(f"    {edge.source} → {edge.target}: {edge.weight:.3f}")

        print("\n  Most active nodes:")
        for node in memory.get_most_active_nodes(5):
            print(f"    {node.id}: {node.activation_count} activations")

        unanchored = memory.get_unanchored_nodes()
        if unanchored:
            print(f"\n  ⚠️  Unanchored nodes: {len(unanchored)}")

    elif args.command == "fire":
        if not args.source or not args.target:
            print("Error: --source and --target required")
            return

        # Ensure nodes exist
        memory.create_node(args.source, args.source)
        memory.create_node(args.target, args.target)

        edge = memory.fire(args.source, args.target, args.relation)
        print(f"✓ Fired: {edge.source} → {edge.target} (w={edge.weight:.3f})")

    elif args.command == "query":
        if not args.node:
            print("Error: --node required")
            return

        node = memory.get_node(args.node)
        if not node:
            print(f"Node not found: {args.node}")
            return

        print(f"\n📍 Node: {node.label}")
        print(f"  ID: {node.id}")
        print(f"  Type: {node.node_type}")
        print(f"  Anchor: {node.anchor or '(none)'}")
        print(f"  Activations: {node.activation_count}")

        print("\n  Related:")
        for n, e in memory.get_related(args.node):
            print(f"    → {n.label} ({e.relation}, w={e.weight:.3f})")

    elif args.command == "consolidate":
        print("\n💤 Running consolidation...")
        decayed, pruned = memory.consolidate()
        print(f"✓ Decayed {decayed} edges, pruned {pruned}")

    elif args.command == "test":
        print("\n🧪 BIOLOGICAL MEMORY TEST")
        print("=" * 50)

        # Create some test nodes
        memory.create_node("python", "Python", anchor="/code/python", node_type="concept")
        memory.create_node("programming", "Programming", node_type="concept")
        memory.create_node("logos", "Logos", anchor="self", node_type="entity")

        # Fire some synapses
        memory.fire("python", "programming", "is_a")
        memory.fire("python", "programming", "is_a")  # Strengthen
        memory.fire("logos", "python", "uses")

        # Query
        print("\nRelated to 'python':")
        for n, e in memory.get_related("python"):
            print(f"  → {n.label} ({e.relation}, w={e.weight:.3f})")

        # Find path
        path = memory.find_path("logos", "programming")
        print(f"\nPath logos → programming: {' → '.join(path) if path else 'None'}")

        print("\n✓ Test complete")


if __name__ == "__main__":
    main()
