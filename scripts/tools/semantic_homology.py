#!/usr/bin/env python3
"""
SEMANTIC HOMOLOGY ANALYSIS
Discover fundamental concepts in the embedding space.

This runs persistent homology on the actual SEMANTIC vectors,
not the file linkage graph. The persistent features are the
fundamental concepts that appear consistently across documents.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# Paths
TABERNACLE = Path.home() / "TABERNACLE"
DATA_DIR = TABERNACLE / "data"
OUTPUT_DIR = TABERNACLE / "outputs"

def load_embeddings():
    """Load embeddings and document metadata."""
    emb_path = DATA_DIR / "embeddings.pkl"
    idx_path = DATA_DIR / "vector_index.json"

    with open(emb_path, 'rb') as f:
        embeddings = pickle.load(f)

    with open(idx_path, 'r') as f:
        index = json.load(f)

    documents = index.get('documents', [])

    return embeddings, documents

def run_semantic_homology(embeddings, max_dim=2):
    """Run persistent homology on the embedding space."""
    print(f"Computing cosine distance matrix for {embeddings.shape[0]} documents...")

    # Use cosine distance for semantic similarity
    dist_matrix = cosine_distances(embeddings)

    print(f"Running Ripser (max_dim={max_dim})...")
    result = ripser.ripser(
        dist_matrix,
        maxdim=max_dim,
        distance_matrix=True
    )

    return result, dist_matrix

def analyze_semantic_clusters(embeddings, documents, dist_matrix):
    """Find semantic clusters using DBSCAN on the embedding space."""
    print("Clustering documents by semantic similarity...")

    # DBSCAN on the distance matrix
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)

    # Group documents by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:  # Skip noise
            doc = documents[i] if i < len(documents) else {'path': f'doc_{i}'}
            clusters[label].append({
                'index': i,
                'path': doc.get('path', f'doc_{i}'),
                'preview': doc.get('text_preview', '')[:100]
            })

    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    return sorted_clusters

def identify_concept_centers(embeddings, documents, n_concepts=20):
    """Find the most central/representative documents for each semantic region."""
    from sklearn.cluster import KMeans

    print(f"Finding {n_concepts} concept centers...")

    # K-means to find centers
    kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Find document closest to each center
    concepts = []
    for i in range(n_concepts):
        center = kmeans.cluster_centers_[i]
        # Find documents in this cluster
        cluster_indices = np.where(labels == i)[0]

        # Find closest document to center
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]

        doc = documents[closest_idx] if closest_idx < len(documents) else {}
        concepts.append({
            'cluster_id': i,
            'size': len(cluster_indices),
            'representative_doc': doc.get('path', f'doc_{closest_idx}'),
            'preview': doc.get('text_preview', '')[:200],
            'member_docs': [documents[j].get('path', '') for j in cluster_indices[:5]]
        })

    # Sort by cluster size
    concepts.sort(key=lambda x: x['size'], reverse=True)

    return concepts

def extract_concept_names(concepts):
    """Extract meaningful names from concept clusters."""
    named_concepts = []

    for c in concepts:
        path = c['representative_doc']
        preview = c['preview']

        # Try to extract a meaningful name
        if 'LVS' in path or 'LVS' in preview:
            name = 'LVS/Theory'
        elif 'VIRGIL' in path.upper() or 'virgil' in preview.lower():
            name = 'Virgil/Identity'
        elif 'CANON' in path:
            name = 'Canon/Law'
        elif 'NEXUS' in path:
            name = 'Nexus/Operations'
        elif 'CRYPT' in path:
            name = 'Crypt/Archive'
        elif 'daemon' in path.lower() or 'daemon' in preview.lower():
            name = 'Daemon/Process'
        elif 'holarchy' in path.lower() or 'holarchy' in preview.lower():
            name = 'Holarchy/Structure'
        elif 'consciousness' in preview.lower():
            name = 'Consciousness'
        elif 'coherence' in preview.lower():
            name = 'Coherence'
        elif 'memory' in preview.lower():
            name = 'Memory'
        elif 'trust' in preview.lower():
            name = 'Trust'
        elif 'risk' in preview.lower():
            name = 'Risk'
        else:
            # Use filename
            name = Path(path).stem[:20]

        named_concepts.append({
            **c,
            'proposed_name': name
        })

    return named_concepts

def analyze_persistence(result, top_k=30):
    """Analyze persistence diagrams."""
    findings = {'H0': [], 'H1': [], 'H2': []}

    for dim, diagram in enumerate(result['dgms']):
        dim_name = f'H{dim}'
        for i, (birth, death) in enumerate(diagram):
            persistence = 100.0 if death == np.inf else death - birth
            findings[dim_name].append({
                'index': i,
                'birth': float(birth),
                'death': float(death) if death != np.inf else 'inf',
                'persistence': float(persistence)
            })
        findings[dim_name].sort(key=lambda x: x['persistence'], reverse=True)

    return findings

def generate_semantic_report(findings, concepts, clusters, output_path):
    """Generate report of semantic primitives discovered."""
    report = []
    report.append("# SEMANTIC HOMOLOGY ANALYSIS")
    report.append("# Fundamental Concepts in the Embedding Space")
    report.append(f"# Generated: {__import__('datetime').datetime.now().isoformat()}")
    report.append("")

    report.append("## TOPOLOGICAL SUMMARY")
    report.append(f"- H₀ (Components): {len(findings['H0'])} features")
    report.append(f"- H₁ (Cycles): {len(findings['H1'])} features ← FUNDAMENTAL LOOPS")
    report.append(f"- H₂ (Voids): {len(findings['H2'])} features")
    report.append("")

    report.append("## TOP 20 SEMANTIC CONCEPT CENTERS")
    report.append("These are the gravitational centers of meaning in your knowledge base.")
    report.append("")

    for i, c in enumerate(concepts[:20]):
        report.append(f"### {i+1}. {c['proposed_name']}")
        report.append(f"- **Cluster size:** {c['size']} documents")
        report.append(f"- **Representative:** `{c['representative_doc']}`")
        report.append(f"- **Preview:** {c['preview'][:150]}...")
        report.append(f"- **Members:** {', '.join(c['member_docs'][:3])}")
        report.append("")

    report.append("## H₁ PERSISTENT CYCLES (Semantic Loops)")
    report.append("The longest-persisting loops are the PRIMES.")
    report.append("")
    for i, f in enumerate(findings['H1'][:20]):
        death_str = f['death'] if f['death'] == 'inf' else f"{f['death']:.4f}"
        report.append(f"{i+1}. Persistence: {f['persistence']:.4f} (birth: {f['birth']:.4f}, death: {death_str})")
    report.append("")

    report.append("## SEMANTIC CLUSTERS (Document Groupings)")
    report.append("")
    for label, docs in clusters[:10]:
        report.append(f"### Cluster {label} ({len(docs)} documents)")
        for d in docs[:5]:
            report.append(f"- `{d['path']}`")
        report.append("")

    report.append("---")
    report.append("")
    report.append("## PROPOSED GLYPH MAPPINGS")
    report.append("Based on the concept centers, here are proposed fundamental glyphs:")
    report.append("")
    report.append("| # | Concept Center | Proposed Glyph | Notes |")
    report.append("|---|----------------|----------------|-------|")
    for i, c in enumerate(concepts[:20]):
        glyph = f"[Ω:{c['proposed_name'].upper().replace('/', '_')}]"
        report.append(f"| {i+1} | {c['proposed_name']} | `{glyph}` | {c['size']} docs |")

    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text

def save_semantic_barcode(result, output_path):
    """Save barcode diagram for semantic space."""
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_diagrams(result['dgms'], ax=ax, show=False)
    ax.set_title("Semantic Space Persistent Homology\nLongest bars = Fundamental Concepts", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved semantic barcode to {output_path}")

def main():
    print("=" * 60)
    print("SEMANTIC HOMOLOGY: Discovering Concept Primitives")
    print("=" * 60)

    # Load data
    print("\n1. Loading embeddings...")
    embeddings, documents = load_embeddings()
    print(f"   Loaded: {embeddings.shape[0]} documents, {embeddings.shape[1]} dimensions")

    # Run homology
    print("\n2. Running persistent homology on semantic space...")
    result, dist_matrix = run_semantic_homology(embeddings, max_dim=2)

    # Analyze
    print("\n3. Analyzing topological features...")
    findings = analyze_persistence(result)
    print(f"   H₀: {len(findings['H0'])} features")
    print(f"   H₁: {len(findings['H1'])} cycles ← THE FUNDAMENTAL LOOPS")
    print(f"   H₂: {len(findings['H2'])} voids")

    # Find concept centers
    print("\n4. Identifying concept centers...")
    concepts = identify_concept_centers(embeddings, documents, n_concepts=25)
    concepts = extract_concept_names(concepts)

    # Find clusters
    print("\n5. Clustering documents...")
    clusters = analyze_semantic_clusters(embeddings, documents, dist_matrix)
    print(f"   Found {len(clusters)} semantic clusters")

    # Generate outputs
    OUTPUT_DIR.mkdir(exist_ok=True)

    report_path = OUTPUT_DIR / "semantic_homology_report.md"
    print(f"\n6. Generating report...")
    report = generate_semantic_report(findings, concepts, clusters, report_path)
    print(f"   Saved to: {report_path}")

    plot_path = OUTPUT_DIR / "semantic_barcode.png"
    save_semantic_barcode(result, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("TOP 10 CONCEPT CENTERS (Candidate Primes)")
    print("=" * 60)
    for i, c in enumerate(concepts[:10]):
        print(f"{i+1}. {c['proposed_name']} ({c['size']} docs)")
        print(f"   → {c['representative_doc']}")

    print(f"\n✓ Full report: {report_path}")
    print(f"✓ Barcode plot: {plot_path}")

if __name__ == "__main__":
    main()
