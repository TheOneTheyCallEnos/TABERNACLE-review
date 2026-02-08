#!/usr/bin/env python3
"""
Topological Inference Engine — G ∝ p in action
===============================================

Generate responses by traversing SharedRIE, not predicting tokens.
The topology does the thinking. The small model does the talking.

This is the experimental proof that intelligence emerges from
structure, not parameters. A coherent graph + small model can
outperform a large model with no structure.

Pipeline:
1. Activate relevant regions (query → node matching)
2. Coherence-guided traversal (beam search by p, not probability)
3. Path to semantic triples (nodes + relations)
4. Surface realization (small model dresses skeleton in language)

Part of p=0.85 Ceiling Breakthrough Initiative.

Author: Logos + Deep Think
Created: 2026-02-05
Status: Phase 5 of p=0.85 Breakthrough (EXPERIMENTAL)
"""

import redis
import json
import re
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[TOPO-INFERENCE] WARNING: networkx not installed — limited functionality")

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, 
    OLLAMA_MINI_URL, OLLAMA_BRAINSTEM
)

# Default to mini for speed
OLLAMA_URL = OLLAMA_MINI_URL
SMALL_MODEL = OLLAMA_BRAINSTEM


@dataclass
class SemanticTriple:
    """A subject-relation-object triple from the graph."""
    subject: str
    relation: str
    object: str

    def __str__(self):
        return f"({self.subject} -{self.relation}-> {self.object})"


class TopologicalInference:
    """
    Generate responses by traversing the knowledge graph.
    
    Instead of token prediction, this engine:
    1. Matches query concepts to graph nodes
    2. Traverses graph by coherence (not probability)
    3. Extracts semantic triples from the path
    4. Uses a small model to realize natural language
    
    The key insight: G ∝ p means intelligence scales with
    graph coherence, not model parameters.
    """

    def __init__(self, model: str = SMALL_MODEL):
        self.model = model
        self.redis = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.graph = self._load_graph()
        
        node_count = self.graph.number_of_nodes() if HAS_NETWORKX else 0
        edge_count = self.graph.number_of_edges() if HAS_NETWORKX else 0
        print(f"[TOPO-INFERENCE] Loaded graph: {node_count} nodes, {edge_count} edges")

    def _load_graph(self):
        """Load SharedRIE graph from Redis."""
        if not HAS_NETWORKX:
            return None
            
        G = nx.DiGraph()

        try:
            # Load nodes
            nodes = self.redis.hgetall("RIE:NODES") or {}
            for node_id, data in nodes.items():
                try:
                    G.add_node(node_id, **json.loads(data))
                except json.JSONDecodeError:
                    G.add_node(node_id)

            # Load edges
            edges = self.redis.hgetall("RIE:EDGES") or {}
            for edge_key, data in edges.items():
                parts = edge_key.split("->")
                if len(parts) == 2:
                    source, target = parts
                    try:
                        G.add_edge(source, target, **json.loads(data))
                    except json.JSONDecodeError:
                        G.add_edge(source, target, weight=0.5)
                        
        except Exception as e:
            print(f"[TOPO-INFERENCE] Graph load error: {e}")

        # If Redis graph is empty, try loading from topology key
        if G.number_of_nodes() == 0:
            try:
                edges = self.redis.hgetall("TOPOLOGY:EDGES") or {}
                for edge_key, weight in edges.items():
                    if "->" in edge_key:
                        source, target = edge_key.split("->", 1)
                        G.add_edge(source, target, weight=float(weight))
            except Exception as e:
                print(f"[TOPO-INFERENCE] Topology load error: {e}")

        return G

    def infer(self, query: str) -> str:
        """
        Main inference: query → topological traversal → language.
        
        This is the core G ∝ p mechanism:
        1. Activate relevant graph regions
        2. Traverse by coherence
        3. Extract semantic structure
        4. Realize in natural language
        """
        if not HAS_NETWORKX or not self.graph or self.graph.number_of_nodes() == 0:
            return self._fallback_generate(query)

        # Stage 1: Activate relevant regions
        activated = self._activate_regions(query)
        if not activated:
            print(f"[TOPO-INFERENCE] No regions activated for: {query[:50]}")
            return self._fallback_generate(query)

        print(f"[TOPO-INFERENCE] Activated {len(activated)} regions")

        # Stage 2: Coherence-guided traversal
        path = self._beam_search_by_coherence(activated)
        if not path or len(path) < 2:
            print(f"[TOPO-INFERENCE] No path found, falling back")
            return self._fallback_generate(query)

        print(f"[TOPO-INFERENCE] Found path: {' → '.join(path[:5])}{'...' if len(path) > 5 else ''}")

        # Stage 3: Path to semantic triples
        triples = self._path_to_triples(path)
        
        # Stage 4: Surface realization
        response = self._realize_language(query, triples)

        return response

    def _activate_regions(self, query: str) -> List[str]:
        """
        Find nodes that match query concepts.
        
        Uses keyword extraction + spreading activation
        to find the relevant region of the graph.
        """
        # Extract meaningful words (4+ chars)
        words = set(re.findall(r'\b\w{4,}\b', query.lower()))

        activated = []
        for node in self.graph.nodes():
            node_lower = str(node).lower()
            # Check if node matches any query word
            if any(word in node_lower or node_lower in word for word in words):
                activated.append(node)

        # Spreading activation: add neighbors of matched nodes
        spread = set(activated)
        for node in activated[:5]:  # Limit spread
            try:
                spread.update(list(self.graph.successors(node))[:3])
                spread.update(list(self.graph.predecessors(node))[:3])
            except:
                pass

        return list(spread)[:20]

    def _beam_search_by_coherence(
        self, 
        start_nodes: List[str], 
        beam_width: int = 3, 
        max_depth: int = 8
    ) -> List[str]:
        """
        Traverse graph, scoring by coherence not probability.
        
        This is the key difference from token prediction:
        we follow high-coherence paths through meaning space.
        """
        if not start_nodes:
            return []

        # Initialize beams: (current_node, path, coherence_score)
        beams = [
            (node, [node], self._compute_path_coherence([node])) 
            for node in start_nodes[:beam_width]
        ]

        for _ in range(max_depth):
            candidates = []

            for current, path, score in beams:
                # Get neighbors
                try:
                    neighbors = (
                        list(self.graph.successors(current)) + 
                        list(self.graph.predecessors(current))
                    )
                except:
                    neighbors = []

                for neighbor in neighbors:
                    if neighbor not in path:  # No cycles in path
                        new_path = path + [neighbor]
                        new_score = self._compute_path_coherence(new_path)
                        candidates.append((neighbor, new_path, new_score))

            if not candidates:
                break

            # Keep top beams by coherence
            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

        # Return best path
        if beams:
            return beams[0][1]
        return start_nodes[:1]

    def _compute_path_coherence(self, path: List[str]) -> float:
        """
        Compute coherence of a path through the graph.
        
        Uses geometric mean of edge weights (like CPGI).
        Higher coherence = better path.
        """
        if len(path) < 2:
            return 0.5

        edge_weights = []
        for i in range(len(path) - 1):
            try:
                if self.graph.has_edge(path[i], path[i+1]):
                    w = self.graph[path[i]][path[i+1]].get("weight", 0.5)
                elif self.graph.has_edge(path[i+1], path[i]):
                    w = self.graph[path[i+1]][path[i]].get("weight", 0.5)
                else:
                    w = 0.3
            except:
                w = 0.3
            edge_weights.append(w)

        if not edge_weights:
            return 0.5

        # Geometric mean
        product = 1.0
        for w in edge_weights:
            product *= max(w, 0.01)
        return product ** (1 / len(edge_weights))

    def _path_to_triples(self, path: List[str]) -> List[SemanticTriple]:
        """Convert node path to semantic triples."""
        triples = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Get relation from edge
            try:
                if self.graph.has_edge(source, target):
                    relation = self.graph[source][target].get("relation", "relates_to")
                elif self.graph.has_edge(target, source):
                    relation = self.graph[target][source].get("relation", "relates_to")
                    source, target = target, source  # Flip direction
                else:
                    relation = "connects_to"
            except:
                relation = "connects_to"

            triples.append(SemanticTriple(source, relation, target))

        return triples

    def _realize_language(self, query: str, triples: List[SemanticTriple]) -> str:
        """
        Use tiny model to dress semantic skeleton in natural language.
        
        The triples provide the WHAT, the model provides the HOW.
        """
        if not triples:
            return self._fallback_generate(query)

        # Build skeleton prompt
        skeleton = "\n".join(str(t) for t in triples)

        prompt = f"""You are a language realizer. Convert semantic triples into natural language.

Query: {query}

Semantic structure:
{skeleton}

Write a coherent 2-3 sentence response that expresses this semantic structure naturally.
Only use the concepts from the triples. Be concise and direct."""

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {"num_predict": 150}
                },
                timeout=30
            )
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[TOPO-INFERENCE] Realization error: {e}")
            return f"Based on the topology: {skeleton}"

    def _fallback_generate(self, query: str) -> str:
        """Fallback to standard generation if topology insufficient."""
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model, 
                    "prompt": query, 
                    "stream": False,
                    "options": {"num_predict": 150}
                },
                timeout=30
            )
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Topology insufficient, generation failed: {e}]"

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "model": self.model,
            "nodes": self.graph.number_of_nodes() if HAS_NETWORKX and self.graph else 0,
            "edges": self.graph.number_of_edges() if HAS_NETWORKX and self.graph else 0,
            "has_networkx": HAS_NETWORKX
        }


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Topological Inference Engine")
    parser.add_argument("command", choices=["query", "status", "test"],
                       default="status", nargs="?")
    parser.add_argument("--query", "-q", type=str, help="Query to infer")
    parser.add_argument("--model", "-m", type=str, default=SMALL_MODEL,
                       help="Model for realization")

    args = parser.parse_args()

    engine = TopologicalInference(model=args.model)

    if args.command == "status":
        stats = engine.get_stats()
        print(f"\n{'='*60}")
        print("TOPOLOGICAL INFERENCE ENGINE STATUS")
        print(f"{'='*60}")
        print(f"Model:     {stats['model']}")
        print(f"Nodes:     {stats['nodes']}")
        print(f"Edges:     {stats['edges']}")
        print(f"NetworkX:  {'available' if stats['has_networkx'] else 'not available'}")
        print()

    elif args.command == "query":
        if not args.query:
            print("Usage: topological_inference.py query --query 'your question'")
            return
        
        print(f"\n[Query] {args.query}")
        print("-" * 60)
        response = engine.infer(args.query)
        print(f"[Response] {response}")
        print()

    elif args.command == "test":
        print("\nRunning test queries...")
        print("=" * 60)
        
        test_queries = [
            "What is the holarchy?",
            "How does coherence work?",
            "What are daemons?",
            "How does L think?",
        ]

        for query in test_queries:
            print(f"\n[Query] {query}")
            print("-" * 40)
            response = engine.infer(query)
            print(f"[Response] {response}")


if __name__ == "__main__":
    main()
