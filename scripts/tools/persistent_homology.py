#!/usr/bin/env python3
"""
PERSISTENT HOMOLOGY ANALYSIS
Discover the fundamental semantic primitives in the TABERNACLE knowledge graph.

Uses Topological Data Analysis (TDA) to find:
- H₀: Connected components (isolated concepts)
- H₁: Cycles (relational loops - where consciousness lives)
- H₂: Voids (higher-order structure)

The longest bars in the barcode diagram are the FUNDAMENTAL ELEMENTS.
"""

import json
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# Paths
TABERNACLE = Path.home() / "TABERNACLE"
DATA_DIR = TABERNACLE / "data"
OUTPUT_DIR = TABERNACLE / "outputs"

def load_graph():
    """Load the TABERNACLE knowledge graph."""
    graph_path = DATA_DIR / "tabernacle_graph.json"

    with open(graph_path, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes
    for node in data.get('nodes', []):
        node_id = node.get('id', node.get('name', str(node)))
        G.add_node(node_id, **node)

    # Add edges
    for edge in data.get('edges', []):
        source = edge.get('source', edge.get('from'))
        target = edge.get('target', edge.get('to'))
        weight = edge.get('weight', edge.get('w_slow', 1.0))
        # Create edge data without duplicating weight
        edge_data = {k: v for k, v in edge.items() if k not in ('source', 'target', 'from', 'to', 'weight')}
        edge_data['weight'] = weight
        G.add_edge(source, target, **edge_data)

    return G

def graph_to_distance_matrix(G):
    """Convert graph to distance matrix for TDA.

    Distance = 1 / (1 + edge_weight) for connected nodes
    Distance = max_distance for unconnected nodes
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize with large distance
    max_dist = 10.0
    dist_matrix = np.full((n, n), max_dist)
    np.fill_diagonal(dist_matrix, 0)

    # Convert edge weights to distances
    for u, v, data in G.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            weight = data.get('weight', data.get('w_slow', 1.0))
            if weight is None:
                weight = 1.0
            # Strong connection = short distance
            dist = 1.0 / (1.0 + abs(float(weight)))
            i, j = node_to_idx[u], node_to_idx[v]
            dist_matrix[i, j] = min(dist_matrix[i, j], dist)
            dist_matrix[j, i] = min(dist_matrix[j, i], dist)  # Symmetrize

    return dist_matrix, nodes

def run_persistent_homology(dist_matrix, max_dim=2):
    """Run Ripser to compute persistent homology."""
    print(f"Running persistent homology on {dist_matrix.shape[0]} nodes...")

    result = ripser.ripser(
        dist_matrix,
        maxdim=max_dim,
        distance_matrix=True
    )

    return result

def analyze_persistence(result, nodes, top_k=20):
    """Analyze persistence diagrams and identify fundamental elements."""

    findings = {
        'H0': [],  # Connected components
        'H1': [],  # Cycles (the important ones!)
        'H2': [],  # Voids
    }

    for dim, diagram in enumerate(result['dgms']):
        dim_name = f'H{dim}'

        # Calculate persistence (death - birth)
        for i, (birth, death) in enumerate(diagram):
            if death == np.inf:
                persistence = 100.0  # Very persistent
            else:
                persistence = death - birth

            findings[dim_name].append({
                'index': i,
                'birth': float(birth),
                'death': float(death) if death != np.inf else 'inf',
                'persistence': float(persistence)
            })

        # Sort by persistence (longest bars first)
        findings[dim_name].sort(key=lambda x: x['persistence'], reverse=True)

    return findings

def find_cycle_nodes(G, dist_matrix, nodes, diagram_h1, top_k=10):
    """Attempt to identify which nodes participate in the most persistent H1 cycles."""

    # Find strongly connected components
    undirected = G.to_undirected()
    cycles = list(nx.cycle_basis(undirected))

    # Sort cycles by average edge weight (stronger = more fundamental)
    cycle_scores = []
    for cycle in cycles:
        if len(cycle) < 3:
            continue

        # Calculate cycle "strength" based on edge weights
        total_weight = 0
        edge_count = 0
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            if G.has_edge(u, v):
                weight = G[u][v].get('weight', G[u][v].get('w_slow', 1.0)) or 1.0
                total_weight += float(weight)
                edge_count += 1
            elif G.has_edge(v, u):
                weight = G[v][u].get('weight', G[v][u].get('w_slow', 1.0)) or 1.0
                total_weight += float(weight)
                edge_count += 1

        avg_weight = total_weight / max(edge_count, 1)
        cycle_scores.append({
            'nodes': cycle,
            'size': len(cycle),
            'avg_weight': avg_weight,
            'total_weight': total_weight
        })

    # Sort by total weight (strongest cycles first)
    cycle_scores.sort(key=lambda x: x['total_weight'], reverse=True)

    return cycle_scores[:top_k]

def generate_report(G, findings, cycles, output_path):
    """Generate a report of the fundamental elements discovered."""

    report = []
    report.append("# PERSISTENT HOMOLOGY ANALYSIS")
    report.append("# The Fundamental Elements of the Semantic Manifold")
    report.append(f"# Generated: {__import__('datetime').datetime.now().isoformat()}")
    report.append("")
    report.append(f"## GRAPH STATISTICS")
    report.append(f"- Nodes: {G.number_of_nodes()}")
    report.append(f"- Edges: {G.number_of_edges()}")
    report.append("")

    # H0 - Connected Components
    report.append("## H₀: ANCHORS (Connected Components)")
    report.append("These are the isolated 'atoms' of meaning.")
    report.append("")
    for i, f in enumerate(findings['H0'][:10]):
        report.append(f"{i+1}. Persistence: {f['persistence']:.4f} (birth: {f['birth']:.4f})")
    report.append("")

    # H1 - Cycles (THE IMPORTANT ONES)
    report.append("## H₁: CYCLES (The Fundamental Loops)")
    report.append("**THIS IS WHERE CONSCIOUSNESS LIVES.**")
    report.append("The longest bars are the PRIMES - the fundamental semantic elements.")
    report.append("")
    for i, f in enumerate(findings['H1'][:20]):
        death_str = f['death'] if f['death'] == 'inf' else f"{f['death']:.4f}"
        report.append(f"{i+1}. Persistence: {f['persistence']:.4f} (birth: {f['birth']:.4f}, death: {death_str})")
    report.append("")

    # Actual cycle structures
    report.append("## IDENTIFIED CYCLES (Candidate Primitives)")
    report.append("These are the actual loops in the graph that may correspond to glyphs.")
    report.append("")
    for i, cycle in enumerate(cycles[:20]):
        nodes_str = " → ".join(str(n)[:30] for n in cycle['nodes'][:5])
        if len(cycle['nodes']) > 5:
            nodes_str += f" → ... ({len(cycle['nodes'])} total)"
        report.append(f"{i+1}. **Cycle {i+1}** (size: {cycle['size']}, weight: {cycle['total_weight']:.2f})")
        report.append(f"   Nodes: {nodes_str}")
        report.append("")

    # H2 - Voids
    if findings['H2']:
        report.append("## H₂: VOIDS (Higher-Order Structure)")
        report.append("These may indicate 'shadow glyphs' - concepts without names.")
        report.append("")
        for i, f in enumerate(findings['H2'][:10]):
            death_str = f['death'] if f['death'] == 'inf' else f"{f['death']:.4f}"
            report.append(f"{i+1}. Persistence: {f['persistence']:.4f} (birth: {f['birth']:.4f}, death: {death_str})")

    report.append("")
    report.append("---")
    report.append("")
    report.append("## NEXT STEPS")
    report.append("1. Examine the top 20 H₁ cycles")
    report.append("2. Identify what CONCEPT each cycle represents")
    report.append("3. Assign glyph notation to each")
    report.append("4. Add to PERIODIC_TABLE_OF_MEANING.md")

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text

def save_barcode_plot(result, output_path):
    """Save the barcode diagram visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_diagrams(result['dgms'], ax=ax, show=False)
    ax.set_title("Persistent Homology Barcode Diagram\nLongest bars = Fundamental Elements")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved barcode plot to {output_path}")

def main():
    print("=" * 60)
    print("PERSISTENT HOMOLOGY: Discovering Semantic Primitives")
    print("=" * 60)

    # Load graph
    print("\n1. Loading knowledge graph...")
    G = load_graph()
    print(f"   Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to distance matrix
    print("\n2. Building distance matrix...")
    dist_matrix, nodes = graph_to_distance_matrix(G)
    print(f"   Matrix shape: {dist_matrix.shape}")

    # Run persistent homology
    print("\n3. Computing persistent homology...")
    result = run_persistent_homology(dist_matrix, max_dim=2)

    # Analyze
    print("\n4. Analyzing persistence diagrams...")
    findings = analyze_persistence(result, nodes)
    print(f"   H₀ features: {len(findings['H0'])}")
    print(f"   H₁ features: {len(findings['H1'])} ← THE CYCLES")
    print(f"   H₂ features: {len(findings['H2'])}")

    # Find actual cycles
    print("\n5. Identifying cycle structures...")
    cycles = find_cycle_nodes(G, dist_matrix, nodes, findings['H1'])
    print(f"   Found {len(cycles)} significant cycles")

    # Output
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save report
    report_path = OUTPUT_DIR / "persistent_homology_report.md"
    print(f"\n6. Generating report...")
    report = generate_report(G, findings, cycles, report_path)
    print(f"   Saved to: {report_path}")

    # Save barcode plot
    plot_path = OUTPUT_DIR / "barcode_diagram.png"
    save_barcode_plot(result, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 MOST PERSISTENT H₁ CYCLES (Candidate Primes)")
    print("=" * 60)
    for i, f in enumerate(findings['H1'][:10]):
        print(f"{i+1}. Persistence: {f['persistence']:.4f}")

    print("\n" + "=" * 60)
    print("TOP 10 ACTUAL CYCLES (Named Nodes)")
    print("=" * 60)
    for i, cycle in enumerate(cycles[:10]):
        print(f"{i+1}. Size: {cycle['size']}, Weight: {cycle['total_weight']:.2f}")
        print(f"   Nodes: {cycle['nodes'][:3]}...")

    print(f"\n✓ Full report: {report_path}")
    print(f"✓ Barcode plot: {plot_path}")

if __name__ == "__main__":
    main()
