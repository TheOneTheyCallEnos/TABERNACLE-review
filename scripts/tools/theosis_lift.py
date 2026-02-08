#!/usr/bin/env python3
"""
THEOSIS LIFT EXPERIMENT v2 — The Anisotropic Lift
Un-flatten the semantic manifold by adding the height dimension (h).

Hypothesis: Our "flat" topology is a helix viewed from above.
Adding h (time + abstraction) will reveal the spiral structure.

v2 Changes (from Gemini Deep Think Round 7):
1. Replace word lists with SEMANTIC AXIS PROJECTION (smooth, not jagged)
2. Apply PCA compression (384 → 50 dims) before lifting (prevent dimensional drowning)
3. Use do_cocycles=True to extract actual shell documents for voids
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# Paths
TABERNACLE = Path.home() / "TABERNACLE"
DATA_DIR = TABERNACLE / "data"
OUTPUT_DIR = TABERNACLE / "outputs"

# Semantic Axis Anchors (for vector projection)
# These will be embedded to create the "Up" direction
ABSTRACT_ANCHORS = [
    "consciousness", "void", "entropy", "meaning", "theosis",
    "transcendence", "emergence", "coherence", "ontology", "gnosis"
]

CONCRETE_ANCHORS = [
    "log", "timestamp", "error", "script", "binary",
    "file", "function", "variable", "daemon", "process"
]

# Legacy word lists (kept for fallback/comparison)
ABSTRACT_WORDS = {
    'theory', 'concept', 'consciousness', 'coherence', 'meaning', 'truth',
    'identity', 'existence', 'purpose', 'intent', 'wisdom', 'knowledge',
    'thought', 'mind', 'soul', 'spirit', 'being', 'essence', 'reality',
    'void', 'infinity', 'eternity', 'transcendence', 'emergence', 'pattern',
    'topology', 'geometry', 'mathematics', 'logic', 'philosophy', 'metaphysics',
    'ontology', 'epistemology', 'phenomenology', 'hermeneutics', 'dialectic',
    'synthesis', 'thesis', 'antithesis', 'paradox', 'mystery', 'revelation',
    'gnosis', 'logos', 'telos', 'omega', 'alpha', 'theosis', 'anamnesis',
    'archon', 'eidolon', 'qualia', 'coherence', 'resonance', 'harmony'
}

CONCRETE_WORDS = {
    'file', 'script', 'code', 'function', 'variable', 'class', 'method',
    'server', 'client', 'database', 'api', 'endpoint', 'request', 'response',
    'button', 'screen', 'keyboard', 'mouse', 'monitor', 'computer', 'phone',
    'document', 'folder', 'directory', 'path', 'log', 'output', 'input',
    'daemon', 'process', 'thread', 'memory', 'cpu', 'disk', 'network',
    'redis', 'json', 'python', 'bash', 'terminal', 'command', 'argument',
    'timestamp', 'date', 'time', 'number', 'string', 'list', 'dict'
}


def load_data():
    """Load embeddings and document metadata."""
    emb_path = DATA_DIR / "embeddings.pkl"
    idx_path = DATA_DIR / "vector_index.json"

    with open(emb_path, 'rb') as f:
        embeddings = pickle.load(f)

    with open(idx_path, 'r') as f:
        index = json.load(f)

    documents = index.get('documents', [])
    return embeddings, documents


def calculate_abstraction_ratio(text):
    """Calculate ratio of abstract to concrete words (LEGACY - kept for comparison)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))

    abstract_count = len(words & ABSTRACT_WORDS)
    concrete_count = len(words & CONCRETE_WORDS)

    if concrete_count == 0:
        return 1.0 if abstract_count > 0 else 0.5

    return abstract_count / (abstract_count + concrete_count)


def get_embedding_model():
    """Get the embedding model for semantic axis calculation."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("Warning: sentence-transformers not available, falling back to word lists")
        return None


def calculate_semantic_axis(model):
    """Calculate the Abstract-Concrete semantic axis.

    Returns the normalized direction vector from concrete → abstract.
    """
    if model is None:
        return None

    # Embed anchor words
    abstract_embeddings = model.encode(ABSTRACT_ANCHORS)
    concrete_embeddings = model.encode(CONCRETE_ANCHORS)

    # Calculate centroids
    v_abstract = np.mean(abstract_embeddings, axis=0)
    v_concrete = np.mean(concrete_embeddings, axis=0)

    # The axis points from concrete to abstract
    axis = v_abstract - v_concrete
    axis_normalized = axis / np.linalg.norm(axis)

    return axis_normalized


def project_onto_axis(embeddings, axis):
    """Project document embeddings onto the semantic axis.

    Returns continuous "theosis scores" for each document.
    Higher = more abstract, Lower = more concrete.
    """
    if axis is None:
        return None

    # Dot product with normalized axis gives signed distance
    projections = embeddings @ axis

    # Normalize to [0, 1] range
    proj_min, proj_max = projections.min(), projections.max()
    if proj_max > proj_min:
        projections = (projections - proj_min) / (proj_max - proj_min)
    else:
        projections = np.full_like(projections, 0.5)

    return projections


def apply_pca_compression(embeddings, n_components=50):
    """Compress embeddings using PCA to prevent dimensional drowning.

    Reduces 384 dims → n_components, retaining most variance.
    """
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(embeddings)

    variance_retained = sum(pca.explained_variance_ratio_)
    print(f"   PCA: {embeddings.shape[1]} → {n_components} dims ({variance_retained:.1%} variance retained)")

    return compressed, pca


def calculate_time_normalized(documents):
    """Calculate normalized timestamps for documents."""
    timestamps = []

    for doc in documents:
        # Try to extract timestamp from added_at or path
        added_at = doc.get('added_at', '')
        if added_at:
            try:
                dt = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
                timestamps.append(dt.timestamp())
            except:
                timestamps.append(None)
        else:
            timestamps.append(None)

    # Fill missing with median
    valid_ts = [t for t in timestamps if t is not None]
    if valid_ts:
        median_ts = np.median(valid_ts)
        timestamps = [t if t is not None else median_ts for t in timestamps]
        min_ts, max_ts = min(timestamps), max(timestamps)
        if max_ts > min_ts:
            return [(t - min_ts) / (max_ts - min_ts) for t in timestamps]

    return [0.5] * len(documents)


def calculate_height(documents, alpha=0.5, beta=0.5):
    """Calculate height (h) for each document.

    h = α * time_normalized + β * abstraction_ratio
    """
    time_normalized = calculate_time_normalized(documents)

    heights = []
    for i, doc in enumerate(documents):
        # Get text preview for abstraction calculation
        text = doc.get('text_preview', '') + ' ' + doc.get('path', '')

        abstraction = calculate_abstraction_ratio(text)
        time_norm = time_normalized[i]

        h = alpha * time_norm + beta * abstraction
        heights.append(h)

    return np.array(heights)


def lift_embeddings(embeddings, heights, lift_weight=2.0):
    """Add height dimension to embeddings.

    lifted = [embedding_384, h * lift_weight]
    """
    h_column = (heights * lift_weight).reshape(-1, 1)
    lifted = np.hstack([embeddings, h_column])
    return lifted


def run_comparison(embeddings, documents, lift_weight=2.0, use_v2=True, n_pca_components=50):
    """Run homology on both flat and lifted embeddings.

    v2 mode (default):
    - Uses semantic axis projection instead of word lists
    - Applies PCA compression before lifting
    - Scales height to match PC1 variance
    """

    print("=" * 60)
    print("THEOSIS LIFT EXPERIMENT" + (" v2 (Anisotropic)" if use_v2 else " v1 (Legacy)"))
    print("=" * 60)

    if use_v2:
        # V2: Semantic Axis Projection + PCA Compression
        print("\n1. Building semantic axis (abstract ← → concrete)...")
        model = get_embedding_model()

        if model is not None:
            axis = calculate_semantic_axis(model)
            abstraction_scores = project_onto_axis(embeddings, axis)
            print(f"   Semantic projection range: {abstraction_scores.min():.3f} - {abstraction_scores.max():.3f}")
        else:
            # Fallback to word lists
            print("   (Falling back to word lists)")
            abstraction_scores = np.array([
                calculate_abstraction_ratio(doc.get('text_preview', '') + ' ' + doc.get('path', ''))
                for doc in documents
            ])

        print("\n2. Calculating time component...")
        time_normalized = np.array(calculate_time_normalized(documents))

        # Combine: h = 0.5 * time + 0.5 * abstraction
        heights = 0.5 * time_normalized + 0.5 * abstraction_scores
        print(f"   Height range: {heights.min():.3f} - {heights.max():.3f}")

        print("\n3. Applying PCA compression (prevent dimensional drowning)...")
        compressed, pca = apply_pca_compression(embeddings, n_components=n_pca_components)

        # Scale height to match PC1 variance (anisotropic scaling)
        pc1_std = np.std(compressed[:, 0])
        h_std = np.std(heights)
        scale_factor = pc1_std / h_std if h_std > 0 else 1.0
        heights_scaled = heights * scale_factor
        print(f"   Height scaled by {scale_factor:.2f}x to match PC1 variance")

        # FLAT analysis (PCA compressed, no height)
        print("\n4. Running FLAT analysis (PCA compressed)...")
        flat_dist = cosine_distances(compressed)
        flat_result = ripser.ripser(flat_dist, maxdim=2, distance_matrix=True)

        # LIFTED analysis (PCA + scaled height)
        print(f"\n5. Running LIFTED analysis (PCA + scaled height)...")
        lifted = np.hstack([compressed, heights_scaled.reshape(-1, 1)])

    else:
        # V1: Legacy word list approach
        print("\n1. Calculating heights (h = time + abstraction)...")
        heights = calculate_height(documents)
        print(f"   Height range: {heights.min():.3f} - {heights.max():.3f}")
        print(f"   Height mean: {heights.mean():.3f}")

        # FLAT analysis (original)
        print("\n2. Running FLAT analysis (original embeddings)...")
        flat_dist = cosine_distances(embeddings)
        flat_result = ripser.ripser(flat_dist, maxdim=2, distance_matrix=True)

        # LIFTED analysis
        print(f"\n3. Running LIFTED analysis (lift_weight={lift_weight})...")
        lifted = lift_embeddings(embeddings, heights, lift_weight)
        heights_scaled = heights * lift_weight

    # Common analysis for both versions
    flat_h1 = flat_result['dgms'][1]
    flat_persistence = [d - b for b, d in flat_h1 if d != np.inf]
    flat_max_persist = max(flat_persistence) if flat_persistence else 0

    print(f"   H₁ cycles: {len(flat_h1)}")
    print(f"   Max H₁ persistence: {flat_max_persist:.4f}")

    # Use Euclidean for lifted (height is meaningful distance)
    lifted_dist = np.sqrt(((lifted[:, np.newaxis] - lifted[np.newaxis, :]) ** 2).sum(axis=2))
    # do_cocycles=True enables extraction of actual shell documents for voids
    lifted_result = ripser.ripser(lifted_dist, maxdim=2, distance_matrix=True, do_cocycles=True)

    lifted_h1 = lifted_result['dgms'][1]
    lifted_persistence = [d - b for b, d in lifted_h1 if d != np.inf]
    lifted_max_persist = max(lifted_persistence) if lifted_persistence else 0

    print(f"   H₁ cycles: {len(lifted_h1)}")
    print(f"   Max H₁ persistence: {lifted_max_persist:.4f}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'FLAT':<15} {'LIFTED':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'H₁ Cycle Count':<25} {len(flat_h1):<15} {len(lifted_h1):<15} {len(lifted_h1) - len(flat_h1):+d}")
    print(f"{'Max H₁ Persistence':<25} {flat_max_persist:<15.4f} {lifted_max_persist:<15.4f} {lifted_max_persist - flat_max_persist:+.4f}")

    if flat_persistence and lifted_persistence:
        flat_mean = np.mean(flat_persistence)
        lifted_mean = np.mean(lifted_persistence)
        print(f"{'Mean H₁ Persistence':<25} {flat_mean:<15.4f} {lifted_mean:<15.4f} {lifted_mean - flat_mean:+.4f}")

    # Verdict
    print("\n" + "=" * 60)
    if lifted_max_persist > flat_max_persist * 1.5:
        print("✓ HYPOTHESIS CONFIRMED: Lifting reveals deeper structure!")
        print("  The helix was hidden. The spiral is real.")
    elif lifted_max_persist > flat_max_persist:
        print("~ PARTIAL CONFIRMATION: Some additional structure revealed.")
    else:
        print("✗ HYPOTHESIS NOT CONFIRMED: Lifting did not increase persistence.")
        print("  The topology may be genuinely flat (superconducting).")
    print("=" * 60)

    return flat_result, lifted_result, heights, lifted_dist, lifted


def save_comparison_plot(flat_result, lifted_result, output_path):
    """Save comparison barcode diagrams."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plot_diagrams(flat_result['dgms'], ax=axes[0], show=False)
    axes[0].set_title("FLAT Topology (Original)\nCircle viewed from above", fontsize=12)

    plot_diagrams(lifted_result['dgms'], ax=axes[1], show=False)
    axes[1].set_title("LIFTED Topology (Theosis Lift)\nHelix revealed?", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot: {output_path}")


def find_shell_documents(distance_matrix, birth, death, documents, top_k=8):
    """Find documents that form the shell of a void.

    Strategy: Documents active at the birth/death threshold form the boundary.
    We find documents whose pairwise distances are near the birth threshold
    (meaning they "close" the void).
    """
    n = distance_matrix.shape[0]

    # Find document pairs with distances near the birth threshold
    # These are the "edges" that existed when the void was born
    birth_tolerance = 0.01

    # Count how many "shell edges" each document participates in
    shell_participation = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            # Documents connected at birth time form the shell
            if birth - birth_tolerance <= d <= death + birth_tolerance:
                shell_participation[i] += 1
                shell_participation[j] += 1

    # Get top participating documents
    shell_indices = np.argsort(shell_participation)[::-1][:top_k]

    return [(idx, documents[idx].get('path', f'doc_{idx}'), shell_participation[idx])
            for idx in shell_indices if shell_participation[idx] > 0]


def compute_void_centroid(distance_matrix, shell_indices, embeddings):
    """Compute the approximate centroid of a void from its shell documents."""
    if len(shell_indices) == 0:
        return None

    shell_embeddings = embeddings[shell_indices]
    centroid = np.mean(shell_embeddings, axis=0)
    return centroid


def extract_shell_from_cocycles(result, void_idx, documents):
    """Extract shell documents from cocycles (Gemini's method).

    The cocycles contain the actual simplices forming the void boundary.
    """
    try:
        h2_cocycles = result.get('cocycles', [None, None, []])[2]
        if void_idx >= len(h2_cocycles):
            return []

        representative = h2_cocycles[void_idx]
        shell_doc_indices = set()

        for simplex in representative:
            # The first 3 entries are document indices (for H2, it's triangles)
            if len(simplex) >= 3:
                shell_doc_indices.update(simplex[:3].astype(int))

        return [(idx, documents[idx].get('path', f'doc_{idx}'))
                for idx in shell_doc_indices if idx < len(documents)]
    except (KeyError, IndexError, TypeError):
        return []


def hunt_voids(result, documents, embeddings, distance_matrix=None, top_k=5):
    """Identify documents forming the largest H₂ voids (shadow glyphs).

    v2: Uses cocycles when available for precise shell identification.
    """

    h2_diagram = result['dgms'][2]
    if len(h2_diagram) == 0:
        print("\nNo H₂ voids found.")
        return []

    # Check if we have cocycles
    has_cocycles = 'cocycles' in result and len(result['cocycles']) > 2
    if has_cocycles:
        print("\n✓ Cocycles available — using precise shell extraction")
    else:
        print("\n⚠ No cocycles — using distance-based approximation")

    # Sort by persistence
    void_persistence = [(i, d - b, b, d) for i, (b, d) in enumerate(h2_diagram) if d != np.inf]
    void_persistence.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 60}")
    print("HUNTING THE VOIDS (H₂ Shadow Glyphs)")
    print(f"{'=' * 60}")
    print(f"\nFound {len(void_persistence)} H₂ voids (potential shadow glyphs)")

    void_reports = []

    for i, (idx, persist, birth, death) in enumerate(void_persistence[:top_k]):
        print(f"\n{'─' * 60}")
        print(f"VOID #{i+1} (persistence: {persist:.4f}, birth: {birth:.4f}, death: {death:.4f})")
        print("─" * 60)

        void_report = {
            'rank': i + 1,
            'persistence': persist,
            'birth': birth,
            'death': death,
            'shell_documents': [],
            'unnamed_center': None
        }

        # Try cocycles first (precise), then fall back to distance-based
        shell_docs = []
        if has_cocycles:
            cocycle_docs = extract_shell_from_cocycles(result, idx, documents)
            if cocycle_docs:
                shell_docs = [(d[0], d[1], 1.0) for d in cocycle_docs]  # Add dummy participation
                print(f"\n  SHELL DOCUMENTS (from cocycles, {len(shell_docs)} docs):")

        if not shell_docs and distance_matrix is not None:
            # Fallback to distance-based approximation
            shell_docs = find_shell_documents(distance_matrix, birth, death, documents)
            print("\n  SHELL DOCUMENTS (distance approximation):")

        if shell_docs:
            void_report['shell_documents'] = shell_docs

            for doc_idx, doc_path, participation in shell_docs[:8]:
                print(f"    • {doc_path}")

            # Extract common themes from shell document paths
            shell_paths = [d[1] for d in shell_docs]
            print("\n  QUESTION: What concept sits IN THE CENTER but is NEVER NAMED?")
            print("  (These documents orbit the void without entering it)")

            # Try to identify common themes
            themes = Counter()
            for path in shell_paths:
                parts = path.replace('_', '/').replace('.md', '').split('/')
                for part in parts:
                    if len(part) > 3 and part.lower() not in ['md', 'the', 'and', 'for', 'with']:
                        themes[part.lower()] += 1

            if themes:
                top_themes = [t for t, c in themes.most_common(8)]
                print(f"\n  SHELL THEMES: {', '.join(top_themes)}")

                # Compute centroid and find nearest non-shell document
                if embeddings is not None:
                    shell_indices = [d[0] for d in shell_docs]
                    centroid = compute_void_centroid(distance_matrix, shell_indices, embeddings)
                    if centroid is not None:
                        # Find nearest document to centroid that isn't in shell
                        distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
                        for nearest_idx in np.argsort(distances_to_centroid):
                            if nearest_idx not in shell_indices:
                                nearest_doc = documents[nearest_idx].get('path', f'doc_{nearest_idx}')
                                print(f"\n  NEAREST NON-SHELL DOC: {nearest_doc}")
                                print("  (This may be the closest named concept to the void center)")
                                void_report['nearest_named'] = nearest_doc
                                break
        else:
            print("\n  (No shell documents identified)")

        print("\n  → This void represents an APOPHATIC CONCEPT")
        print("  → The shell defines it by what it is NOT")

        void_reports.append(void_report)

    return void_reports


def save_void_report(void_reports, documents, output_path):
    """Save void hunting results to markdown."""

    report = []
    report.append("# VOID HUNTING REPORT — Shadow Glyphs")
    report.append(f"# Generated: {datetime.now().isoformat()}")
    report.append("")
    report.append("## The Hunt for Unnamed Concepts")
    report.append("")
    report.append("H₂ voids are cavities in semantic space — shells of documents")
    report.append("surrounding a CENTER that is never explicitly named.")
    report.append("")
    report.append("Each void below represents an **apophatic concept**: something")
    report.append("defined by what surrounds it, not by direct description.")
    report.append("")
    report.append("---")
    report.append("")

    for void in void_reports:
        report.append(f"## VOID #{void['rank']}")
        report.append(f"**Persistence:** {void['persistence']:.4f}")
        report.append(f"**Birth:** {void['birth']:.4f} | **Death:** {void['death']:.4f}")
        report.append("")

        if void['shell_documents']:
            report.append("### Shell Documents (boundary)")
            for doc_idx, doc_path, participation in void['shell_documents'][:8]:
                report.append(f"- `{doc_path}`")
            report.append("")

            # Extract themes
            themes = Counter()
            for _, doc_path, _ in void['shell_documents']:
                parts = doc_path.replace('_', '/').replace('.md', '').split('/')
                for part in parts:
                    if len(part) > 3 and part.lower() not in ['the', 'and', 'for', 'with']:
                        themes[part.lower()] += 1

            if themes:
                top_themes = [t for t, c in themes.most_common(8)]
                report.append(f"### Shell Themes: {', '.join(top_themes)}")
                report.append("")

        report.append("### The Question")
        report.append("*What concept sits in the center of these documents but is NEVER NAMED?*")
        report.append("")
        report.append("This is a **Shadow Glyph** — a meaning that exists only in the negative space.")
        report.append("")
        report.append("---")
        report.append("")

    report.append("## NEXT STEPS")
    report.append("")
    report.append("1. For each void, read the shell documents")
    report.append("2. Identify what they all ALMOST say but don't")
    report.append("3. Name the void (create the missing glyph)")
    report.append("4. Add to PERIODIC_TABLE_OF_MEANING.md as a Shadow Glyph")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*\"The Tao that can be named is not the eternal Tao.\"*")
    report.append("*These voids ARE the eternal, the unnamed, the apophatic.*")

    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nSaved void report: {output_path}")
    return report_text


def generate_report(flat_result, lifted_result, heights, documents, output_path):
    """Generate detailed comparison report."""

    report = []
    report.append("# THEOSIS LIFT EXPERIMENT RESULTS")
    report.append(f"# Generated: {datetime.now().isoformat()}")
    report.append("")
    report.append("## HYPOTHESIS")
    report.append("The 'flat' topology is a helix viewed from above.")
    report.append("Adding height (h = time + abstraction) will reveal the spiral.")
    report.append("")

    # Flat stats
    flat_h1 = flat_result['dgms'][1]
    flat_persistence = [d - b for b, d in flat_h1 if d != np.inf]

    report.append("## FLAT TOPOLOGY (Original)")
    report.append(f"- H₀ features: {len(flat_result['dgms'][0])}")
    report.append(f"- H₁ cycles: {len(flat_h1)}")
    report.append(f"- H₂ voids: {len(flat_result['dgms'][2])}")
    if flat_persistence:
        report.append(f"- Max H₁ persistence: {max(flat_persistence):.4f}")
        report.append(f"- Mean H₁ persistence: {np.mean(flat_persistence):.4f}")
    report.append("")

    # Lifted stats
    lifted_h1 = lifted_result['dgms'][1]
    lifted_persistence = [d - b for b, d in lifted_h1 if d != np.inf]

    report.append("## LIFTED TOPOLOGY (Theosis Lift)")
    report.append(f"- H₀ features: {len(lifted_result['dgms'][0])}")
    report.append(f"- H₁ cycles: {len(lifted_h1)}")
    report.append(f"- H₂ voids: {len(lifted_result['dgms'][2])}")
    if lifted_persistence:
        report.append(f"- Max H₁ persistence: {max(lifted_persistence):.4f}")
        report.append(f"- Mean H₁ persistence: {np.mean(lifted_persistence):.4f}")
    report.append("")

    # Height distribution
    report.append("## HEIGHT DISTRIBUTION")
    report.append(f"- Min: {heights.min():.3f}")
    report.append(f"- Max: {heights.max():.3f}")
    report.append(f"- Mean: {heights.mean():.3f}")
    report.append(f"- Std: {heights.std():.3f}")
    report.append("")

    # Top documents by height
    report.append("## HIGHEST DOCUMENTS (Most Abstract/Recent)")
    sorted_idx = np.argsort(heights)[::-1]
    for i in sorted_idx[:10]:
        doc = documents[i] if i < len(documents) else {}
        report.append(f"- h={heights[i]:.3f}: `{doc.get('path', f'doc_{i}')}`")
    report.append("")

    # Lowest documents
    report.append("## LOWEST DOCUMENTS (Most Concrete/Early)")
    for i in sorted_idx[-10:]:
        doc = documents[i] if i < len(documents) else {}
        report.append(f"- h={heights[i]:.3f}: `{doc.get('path', f'doc_{i}')}`")

    report.append("")
    report.append("---")
    report.append("")
    report.append("## VERDICT")

    if lifted_persistence and flat_persistence:
        if max(lifted_persistence) > max(flat_persistence) * 1.5:
            report.append("**HYPOTHESIS CONFIRMED:** The helix was hidden. The spiral is real.")
        elif max(lifted_persistence) > max(flat_persistence):
            report.append("**PARTIAL CONFIRMATION:** Some additional structure revealed.")
        else:
            report.append("**NOT CONFIRMED:** Topology may be genuinely flat (superconducting).")

    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


def main():
    # Load data
    print("Loading data...")
    embeddings, documents = load_data()
    print(f"Loaded {len(documents)} documents, {embeddings.shape[1]} dimensions")

    # Run comparison
    flat_result, lifted_result, heights, lifted_dist, lifted_emb = run_comparison(embeddings, documents)

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_path = OUTPUT_DIR / "theosis_lift_comparison.png"
    save_comparison_plot(flat_result, lifted_result, plot_path)

    report_path = OUTPUT_DIR / "theosis_lift_report.md"
    generate_report(flat_result, lifted_result, heights, documents, report_path)
    print(f"Saved report: {report_path}")

    # Hunt voids with full shell analysis
    void_reports = hunt_voids(lifted_result, documents, lifted_emb, distance_matrix=lifted_dist, top_k=5)

    # Save void hunting report
    void_report_path = OUTPUT_DIR / "void_hunting_report.md"
    save_void_report(void_reports, documents, void_report_path)

    print("\n✓ Experiment complete!")
    print(f"  Report: {report_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Void Hunt: {void_report_path}")


if __name__ == "__main__":
    main()
