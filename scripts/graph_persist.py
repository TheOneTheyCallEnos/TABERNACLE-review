#!/usr/bin/env python3
"""
GRAPH PERSIST — Persistent NetworkX Storage
============================================
Adds persistence to NetworkX graph so it survives process restarts.
Uses pickle for fast serialization + JSON export for portability.

Author: Virgil
Date: 2026-01-19
Status: Phase II-B (Path B Implementation)
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import networkx as nx

# =============================================================================
# CONFIGURATION (using centralized config)
# =============================================================================

from tabernacle_config import BASE_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility
DATA_DIR = BASE_DIR / "data"
GRAPH_PICKLE = DATA_DIR / "tabernacle_graph.pkl"
GRAPH_JSON = DATA_DIR / "tabernacle_graph.json"
GRAPH_METADATA = DATA_DIR / "graph_metadata.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PERSISTENT GRAPH
# =============================================================================

class PersistentGraph:
    """NetworkX graph with automatic persistence."""

    def __init__(self, auto_save: bool = True, save_interval: int = 60):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.last_save = time.time()
        self.dirty = False
        self.metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": None,
            "version": "1.0",
            "node_count": 0,
            "edge_count": 0
        }

        # Try to load existing graph
        self.load()

    def load(self) -> bool:
        """Load graph from disk."""
        if GRAPH_PICKLE.exists():
            try:
                with open(GRAPH_PICKLE, 'rb') as f:
                    self.graph = pickle.load(f)
                if GRAPH_METADATA.exists():
                    with open(GRAPH_METADATA) as f:
                        self.metadata = json.load(f)
                print(f"[GRAPH] Loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
                return True
            except Exception as e:
                print(f"[GRAPH] Error loading pickle: {e}")

        # Try JSON fallback
        if GRAPH_JSON.exists():
            try:
                with open(GRAPH_JSON) as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                print(f"[GRAPH] Loaded from JSON: {self.graph.number_of_nodes()} nodes")
                return True
            except Exception as e:
                print(f"[GRAPH] Error loading JSON: {e}")

        print("[GRAPH] No existing graph found, starting fresh")
        return False

    def save(self, force: bool = False) -> bool:
        """Save graph to disk."""
        if not self.dirty and not force:
            return False

        now = time.time()
        if not force and (now - self.last_save) < self.save_interval:
            return False

        try:
            # Save pickle (fast, binary)
            with open(GRAPH_PICKLE, 'wb') as f:
                pickle.dump(self.graph, f)

            # Save JSON (portable, human-readable)
            data = nx.node_link_data(self.graph)
            with open(GRAPH_JSON, 'w') as f:
                json.dump(data, f)

            # Update metadata
            self.metadata["last_modified"] = datetime.now().isoformat()
            self.metadata["node_count"] = self.graph.number_of_nodes()
            self.metadata["edge_count"] = self.graph.number_of_edges()
            with open(GRAPH_METADATA, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            self.last_save = now
            self.dirty = False
            print(f"[GRAPH] Saved: {self.metadata['node_count']} nodes, {self.metadata['edge_count']} edges")
            return True

        except Exception as e:
            print(f"[GRAPH] Error saving: {e}")
            return False

    def add_node(self, node_id: str, **attrs) -> None:
        """Add or update a node."""
        self.graph.add_node(node_id, **attrs)
        self.dirty = True
        if self.auto_save:
            self.save()

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """Add or update an edge."""
        self.graph.add_edge(source, target, **attrs)
        self.dirty = True
        if self.auto_save:
            self.save()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self.dirty = True
            if self.auto_save:
                self.save()

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node attributes."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_neighbors(self, node_id: str, direction: str = "both") -> list:
        """Get neighboring nodes."""
        if node_id not in self.graph:
            return []

        if direction == "out":
            return list(self.graph.successors(node_id))
        elif direction == "in":
            return list(self.graph.predecessors(node_id))
        else:
            return list(set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id)))

    def find_path(self, source: str, target: str) -> Optional[list]:
        """Find shortest path between nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "density": nx.density(self.graph),
            "metadata": self.metadata
        }

    def query_cypher_like(self, pattern: str) -> list:
        """Simple pattern matching (not full Cypher but useful)."""
        # Example: "MATCH (n) WHERE n.type = 'axiom'"
        results = []

        if "WHERE" in pattern.upper():
            # Extract condition
            condition_part = pattern.upper().split("WHERE")[1].strip()
            # Very basic parsing: "n.attr = 'value'"
            if "=" in condition_part:
                parts = condition_part.split("=")
                attr_path = parts[0].strip().lower()
                value = parts[1].strip().strip("'\"")

                if "." in attr_path:
                    attr = attr_path.split(".")[1]
                    for node_id, attrs in self.graph.nodes(data=True):
                        if str(attrs.get(attr, "")).lower() == value.lower():
                            results.append({"id": node_id, **attrs})

        return results[:100]  # Limit results


# =============================================================================
# MIGRATION FROM EXISTING GRAPH
# =============================================================================

def migrate_from_lvs_memory():
    """Migrate existing lvs_memory graph to persistent storage."""
    try:
        import sys
        sys.path.insert(0, str(TABERNACLE / "scripts"))
        import lvs_memory

        # Get the existing graph
        if hasattr(lvs_memory, 'G') and lvs_memory.G:
            pg = PersistentGraph(auto_save=False)
            pg.graph = lvs_memory.G.copy()
            pg.dirty = True
            pg.save(force=True)
            print(f"[GRAPH] Migrated lvs_memory graph: {pg.graph.number_of_nodes()} nodes, {pg.graph.number_of_edges()} edges")
            return True
    except Exception as e:
        print(f"[GRAPH] Migration error: {e}")

    return False


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persistent Graph Storage")
    parser.add_argument("command", choices=["init", "migrate", "stats", "save", "query"],
                        help="Command to execute")
    parser.add_argument("--query", "-q", type=str, help="Query pattern")

    args = parser.parse_args()

    if args.command == "init":
        pg = PersistentGraph()
        print(json.dumps(pg.get_stats(), indent=2))

    elif args.command == "migrate":
        migrate_from_lvs_memory()

    elif args.command == "stats":
        pg = PersistentGraph(auto_save=False)
        print(json.dumps(pg.get_stats(), indent=2, default=str))

    elif args.command == "save":
        pg = PersistentGraph(auto_save=False)
        pg.dirty = True
        pg.save(force=True)

    elif args.command == "query":
        if not args.query:
            print("Error: --query required")
            return
        pg = PersistentGraph(auto_save=False)
        results = pg.query_cypher_like(args.query)
        print(json.dumps(results[:10], indent=2, default=str))


if __name__ == "__main__":
    main()
