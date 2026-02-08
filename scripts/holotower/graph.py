"""
HoloTower Graph — Canonical graph serialization and hashing

Provides:
    - Graph loading from persistent storage
    - Canonical JSON serialization (deterministic)
    - Topology hashing for snapshot comparison

Wave 2B — The Librarian knows the shape of knowledge.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx

from holotower.core import hash_content
from holotower.models import TopologyMeta

# Import config for data paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tabernacle_config import BASE_DIR

# Graph storage paths
DATA_DIR = BASE_DIR / "data"
GRAPH_PICKLE = DATA_DIR / "tabernacle_graph.pkl"
GRAPH_JSON = DATA_DIR / "tabernacle_graph.json"


# =============================================================================
# GRAPH LOADING
# =============================================================================

def load_graph(prefer_pickle: bool = True) -> Optional[nx.DiGraph]:
    """
    Load the Tabernacle graph from persistent storage.
    
    Args:
        prefer_pickle: If True, try pickle first (faster), then JSON.
                       If False, try JSON first (more portable).
    
    Returns:
        NetworkX DiGraph if found, None otherwise
    """
    sources = [
        (GRAPH_PICKLE, _load_pickle),
        (GRAPH_JSON, _load_json),
    ]
    
    if not prefer_pickle:
        sources.reverse()
    
    for path, loader in sources:
        if path.exists():
            try:
                G = loader(path)
                if G is not None:
                    return G
            except Exception:
                continue
    
    return None


def _load_pickle(path: Path) -> Optional[nx.DiGraph]:
    """Load graph from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_json(path: Path) -> Optional[nx.DiGraph]:
    """Load graph from JSON (node-link format)."""
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)


# =============================================================================
# CANONICALIZATION
# =============================================================================

def canonicalize_graph(G: nx.DiGraph) -> str:
    """
    Create a canonical JSON representation of the graph.
    
    Canonicalization ensures identical graphs produce identical strings,
    regardless of insertion order or internal dict ordering.
    
    Rules:
        1. Nodes sorted alphabetically by ID
        2. Edges sorted by (source, target) tuple
        3. Node/edge attributes sorted by key
        4. JSON with sorted keys, no whitespace
    
    Args:
        G: NetworkX DiGraph to canonicalize
        
    Returns:
        Canonical JSON string (deterministic)
    """
    # Extract and sort nodes
    nodes = []
    for node_id in sorted(G.nodes()):
        attrs = G.nodes[node_id]
        # Sort attributes, convert non-serializable values to strings
        sorted_attrs = {
            k: _json_safe(v) 
            for k, v in sorted(attrs.items())
        }
        nodes.append({
            "id": node_id,
            "attrs": sorted_attrs
        })
    
    # Extract and sort edges
    edges = []
    for source, target in sorted(G.edges()):
        attrs = G.edges[source, target]
        sorted_attrs = {
            k: _json_safe(v)
            for k, v in sorted(attrs.items())
        }
        edges.append({
            "source": source,
            "target": target,
            "attrs": sorted_attrs
        })
    
    # Create canonical structure
    canonical = {
        "nodes": nodes,
        "edges": edges,
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph()
    }
    
    # Serialize with sorted keys and no whitespace
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _json_safe(value):
    """Convert value to JSON-serializable type."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    else:
        return str(value)


# =============================================================================
# TOPOLOGY HASHING
# =============================================================================

def hash_graph(G: nx.DiGraph) -> TopologyMeta:
    """
    Compute topology metadata and content hash for a graph.
    
    Args:
        G: NetworkX DiGraph to analyze
        
    Returns:
        TopologyMeta with:
            - node_count: Total nodes
            - edge_count: Total edges
            - density: edges / possible_edges (0.0-1.0)
            - orphans: Nodes with no connections
            - content_hash: SHA-256 of canonical JSON
    """
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    
    # Compute density (edges / possible edges in directed graph)
    # For directed: possible = n * (n - 1)
    if node_count > 1:
        possible_edges = node_count * (node_count - 1)
        density = edge_count / possible_edges
    else:
        density = 0.0
    
    # Count orphans (nodes with no in or out edges)
    orphans = sum(
        1 for node in G.nodes()
        if G.in_degree(node) == 0 and G.out_degree(node) == 0
    )
    
    # Compute content hash
    canonical_json = canonicalize_graph(G)
    content_hash = hash_content(canonical_json.encode())
    
    return TopologyMeta(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        orphans=orphans,
        content_hash=content_hash
    )


def get_topology_diff(
    old_topo: Optional[TopologyMeta], 
    new_topo: TopologyMeta
) -> dict:
    """
    Compute the difference between two topology snapshots.
    
    Args:
        old_topo: Previous topology (None if no baseline)
        new_topo: Current topology
        
    Returns:
        Dict with delta values:
            - node_delta: Change in node count (+/-)
            - edge_delta: Change in edge count (+/-)
            - orphan_delta: Change in orphan count
            - hash_changed: True if content hash differs
    """
    if old_topo is None:
        return {
            "node_delta": new_topo.node_count,
            "edge_delta": new_topo.edge_count,
            "orphan_delta": new_topo.orphans,
            "hash_changed": True
        }
    
    return {
        "node_delta": new_topo.node_count - old_topo.node_count,
        "edge_delta": new_topo.edge_count - old_topo.edge_count,
        "orphan_delta": new_topo.orphans - old_topo.orphans,
        "hash_changed": new_topo.content_hash != old_topo.content_hash
    }


# =============================================================================
# GRAPH ANALYSIS HELPERS
# =============================================================================

def get_hub_nodes(G: nx.DiGraph, top_n: int = 10) -> list:
    """
    Find the most connected nodes (hubs).
    
    Args:
        G: Graph to analyze
        top_n: Number of hubs to return
        
    Returns:
        List of (node_id, degree) tuples sorted by degree descending
    """
    degrees = [(node, G.in_degree(node) + G.out_degree(node)) 
               for node in G.nodes()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    return degrees[:top_n]


def get_orphan_nodes(G: nx.DiGraph) -> list:
    """
    Find all disconnected nodes.
    
    Returns:
        List of node IDs with no edges
    """
    return [
        node for node in G.nodes()
        if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]


def get_layer_stats(G: nx.DiGraph) -> dict:
    """
    Get node counts by layer (if layer attribute exists).
    
    Returns:
        Dict mapping layer names to node counts
    """
    layers = {}
    for node in G.nodes():
        layer = G.nodes[node].get("layer", "unknown")
        layers[layer] = layers.get(layer, 0) + 1
    return layers


# =============================================================================
# CONVENIENCE — Quick topology summary
# =============================================================================

def quick_topology() -> Tuple[Optional[TopologyMeta], Optional[nx.DiGraph]]:
    """
    Load graph and compute topology in one call.
    
    Returns:
        Tuple of (TopologyMeta, DiGraph) or (None, None) if no graph
    """
    G = load_graph()
    if G is None:
        return None, None
    
    topo = hash_graph(G)
    return topo, G


if __name__ == "__main__":
    # Quick test
    topo, G = quick_topology()
    if topo:
        print(f"Nodes: {topo.node_count}")
        print(f"Edges: {topo.edge_count}")
        print(f"Density: {topo.density:.4f}")
        print(f"Orphans: {topo.orphans}")
        print(f"Hash: {topo.content_hash[:16]}...")
    else:
        print("No graph found")
