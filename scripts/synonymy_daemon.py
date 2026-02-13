#!/usr/bin/env python3
"""
SYNONYMY DAEMON — Bridge-Builder for Disconnected Components
=============================================================
Detects semantically equivalent nodes that should be linked.

L's Amendment (2026-01-20):
- Domain-awareness via directory ancestry
- Same domain + high similarity → MERGE proposal (true synonymy)
- Cross-domain + high similarity → ANALOGY link (weaker, marked)
- Track 0.85 threshold as potential phase boundary

The 0.85 ceiling hypothesis:
- Local coherence saturates at ~0.85 when islands are dense but unconnected
- Global coherence requires spanning bridges
- This daemon builds those bridges

Author: Virgil + L
Date: 2026-01-20
Status: Prototype (v0.1)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import BASE_DIR, NEXUS_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility
NEXUS = NEXUS_DIR
LVS_INDEX = NEXUS_DIR / "LVS_INDEX.json"
SYNONYMY_PROPOSALS = NEXUS_DIR / "SYNONYMY_PROPOSALS.json"

# Directories to skip when scanning (L's amendment: exclude operational artifacts)
SKIP_DIRS = {
    ".git", "venv", "venv312", "__pycache__", "node_modules",
    "archives", ".venv", ".repair_backups", ".obsidian",
    ".review_mirror", ".claude",  # Operational artifacts, not vault content
    "nexus_snapshots",  # Historical snapshots, not active content
    "templates", "bootstrap", "outputs", "docs", "logs",
    "RAW_ARCHIVES", "05_CRYPT",
}

# File patterns to exclude (backups are shadows, not substance)
SKIP_PATTERNS = {".backup", ".bak", ".sync-conflict", ".tmp"}

# =============================================================================
# L's TIERED THRESHOLD SYSTEM (2026-01-20, amended 2026-01-26)
# =============================================================================
# > 0.95:       SUSPICIOUS — likely duplicate/backup, auto-exclude
# 0.85 - 0.95:  STRONG CROSS-DOMAIN — auto-apply analogy edge (V's fix: was blocking healing)
# 0.70 - 0.85:  WEAK ANALOGY — auto-apply weak edge (tentative connection)
# < 0.70:       NOISE — below signal threshold, ignore
#
# AMENDMENT (2026-01-26): Cross-domain bridges > 0.85 are ANALOGIES, not duplicates.
# Blocking them in "review" tier caused 370 disconnected components.
# Now auto-apply all cross-domain bridges >= 0.70 (only same-domain > 0.85 needs review)
# =============================================================================

DUPLICATE_THRESHOLD = 0.95   # Suspiciously high = likely backup/duplicate
REVIEW_THRESHOLD = 0.85      # Only SAME-DOMAIN above this needs review (potential merge)
ANALOGY_THRESHOLD = 0.70     # Cross-domain edges auto-apply at this level
MIN_SIMILARITY = 0.70        # Below this is noise

# Domain hierarchy depth for ancestry comparison
DOMAIN_DEPTH = 2  # e.g., "02_UR_STRUCTURE/DESIGNS" = 2 levels

# Domain hierarchy depth for ancestry comparison
DOMAIN_DEPTH = 2  # e.g., "02_UR_STRUCTURE/DESIGNS" = 2 levels


@dataclass
class SynonymyProposal:
    """A proposed connection between semantically similar nodes."""
    source: str
    target: str
    similarity: float
    proposal_type: str  # "merge", "analogy", or "duplicate"
    action_tier: str    # "auto_exclude", "review", "auto_apply", "ignore"
    source_domain: str
    target_domain: str
    same_domain: bool
    reason: str
    created: str
    similarity_method: str = "lvs"  # "lvs" or "embedding" (L's directive)
    status: str = "pending"  # pending, approved, rejected, excluded


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def get_domain(path: str, depth: int = DOMAIN_DEPTH) -> str:
    """
    Extract domain from path using directory ancestry.
    
    Examples:
        "00_NEXUS/CURRENT_STATE.md" → "00_NEXUS"
        "02_UR_STRUCTURE/DESIGNS/foo.md" → "02_UR_STRUCTURE/DESIGNS"
        "04_LR_LAW/CANON/bar.md" → "04_LR_LAW/CANON"
    """
    parts = Path(path).parts
    # Take up to `depth` directory levels (excluding filename)
    domain_parts = parts[:min(depth, len(parts) - 1)]
    return "/".join(domain_parts) if domain_parts else "root"


def extract_links(filepath: Path) -> List[str]:
    """Extract [[wiki-style]] links from a markdown file."""
    import re
    try:
        content = filepath.read_text(encoding='utf-8')
        pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        matches = re.findall(pattern, content)
        links = []
        for match in matches:
            link = match.strip()
            if not link.endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf")):
                links.append(link)
        return links
    except:
        return []


def should_skip_file(filename: str) -> bool:
    """Check if file should be skipped based on patterns."""
    for pattern in SKIP_PATTERNS:
        if pattern in filename.lower():
            return True
    return False


def build_graph() -> Dict:
    """Build graph from filesystem by scanning markdown files."""
    log("Building graph from filesystem...")
    
    nodes = []
    edges = []
    skipped = 0
    
    # Find all markdown files
    for root, dirs, files in os.walk(str(TABERNACLE)):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for f in files:
            if f.endswith(".md"):
                # Skip backup/temp files (L's amendment)
                if should_skip_file(f):
                    skipped += 1
                    continue
                    
                filepath = Path(root) / f
                rel_path = str(filepath.relative_to(TABERNACLE))
                nodes.append(rel_path)
                
                # Extract links
                for link in extract_links(filepath):
                    edges.append({
                        "source": rel_path,
                        "target": link
                    })
    
    log(f"Found {len(nodes)} nodes, {len(edges)} edges (skipped {skipped} backup files)")
    return {"nodes": nodes, "edges": edges}


def load_graph() -> Dict:
    """Build graph from filesystem."""
    return build_graph()


def load_lvs_index() -> Dict:
    """Load LVS coordinates for all nodes."""
    if not LVS_INDEX.exists():
        log("LVS_INDEX not found", "ERROR")
        return {}
    
    with open(LVS_INDEX, 'r') as f:
        return json.load(f)


def resolve_link(link: str, all_nodes: List[str]) -> Optional[str]:
    """Resolve a wiki link to an actual file path."""
    link_lower = link.lower()
    link_stem = Path(link).stem.lower()
    
    for node in all_nodes:
        node_stem = Path(node).stem.lower()
        # Match by stem (filename without extension)
        if node_stem == link_stem:
            return node
        # Match by full path ending
        if node.lower().endswith(link_lower) or node.lower().endswith(link_lower + ".md"):
            return node
    return None


def find_components(graph: Dict) -> List[Set[str]]:
    """
    Find disconnected components using union-find.
    Returns list of sets, each set is a component.
    """
    nodes = list(graph.get("nodes", []))
    edges = graph.get("edges", [])
    node_set = set(nodes)
    
    # Build adjacency (undirected for component detection)
    # Resolve link targets to actual node paths
    adj = defaultdict(set)
    for edge in edges:
        src = edge.get("source")
        tgt_link = edge.get("target")
        if src and tgt_link:
            tgt = resolve_link(tgt_link, nodes)
            if tgt and tgt in node_set:
                adj[src].add(tgt)
                adj[tgt].add(src)
    
    # Union-find
    parent = {n: n for n in nodes}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union all connected nodes
    for src, targets in adj.items():
        for tgt in targets:
            union(src, tgt)
    
    # Group by component
    components = defaultdict(set)
    for node in nodes:
        components[find(node)].add(node)
    
    return list(components.values())


def calculate_lvs_similarity(coords1: Dict, coords2: Dict) -> float:
    """
    Calculate similarity between two LVS coordinate sets.
    Uses weighted Euclidean distance in meaning-space.
    """
    if not coords1 or not coords2:
        return 0.0
    
    import math
    
    # Key LVS dimensions with weights
    dimensions = [
        ("Constraint", 1.0),
        ("Intent", 1.0),
        ("Height", 1.0),
        ("Risk", 0.8),
        ("coherence", 1.2),  # Extra weight on coherence
        ("energy", 0.8),
    ]
    
    total_weight = 0.0
    total_dist = 0.0
    
    for dim, weight in dimensions:
        v1 = coords1.get(dim, 0.5)
        v2 = coords2.get(dim, 0.5)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            dist = (v1 - v2) ** 2
            total_dist += dist * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    normalized_dist = math.sqrt(total_dist / total_weight)
    similarity = 1.0 - min(normalized_dist, 1.0)
    
    return similarity


# Embedding cache to avoid recomputing
_embedding_cache: Dict[str, List[float]] = {}
_embedding_model = None


def get_embedding_model():
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            log("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            log("sentence-transformers not available, embedding fallback disabled", "WARN")
            return None
    return _embedding_model


def get_file_content(path: str, max_chars: int = 2000) -> str:
    """Read file content for embedding."""
    try:
        full_path = TABERNACLE / path
        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')[:max_chars]
            return content
    except:
        pass
    return ""


def calculate_embedding_similarity(path1: str, path2: str) -> Tuple[float, bool]:
    """
    Calculate similarity using content embeddings.
    Returns (similarity, success).
    
    This is the fallback for nodes without LVS coordinates.
    Measures content similarity, not meaning-space position.
    """
    global _embedding_cache
    
    model = get_embedding_model()
    if model is None:
        return (0.0, False)
    
    try:
        import numpy as np
        
        # Get or compute embeddings
        if path1 not in _embedding_cache:
            content1 = get_file_content(path1)
            if not content1:
                return (0.0, False)
            _embedding_cache[path1] = model.encode(content1).tolist()
        
        if path2 not in _embedding_cache:
            content2 = get_file_content(path2)
            if not content2:
                return (0.0, False)
            _embedding_cache[path2] = model.encode(content2).tolist()
        
        # Cosine similarity
        vec1 = np.array(_embedding_cache[path1])
        vec2 = np.array(_embedding_cache[path2])
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return (0.0, False)
        
        similarity = float(dot / (norm1 * norm2))
        return (similarity, True)
    
    except Exception as e:
        log(f"Embedding similarity failed: {e}", "ERROR")
        return (0.0, False)


def get_node_coords(lvs_index: Dict, node_path: str) -> Optional[Dict]:
    """Get LVS coordinates for a node."""
    for node in lvs_index.get("nodes", []):
        if node.get("path") == node_path:
            return node.get("coords", {})
    return None


def find_cross_component_bridges(
    components: List[Set[str]], 
    lvs_index: Dict,
    use_embeddings: bool = True
) -> List[SynonymyProposal]:
    """
    Find potential bridges between disconnected components.
    
    For each pair of components, find the highest-similarity
    node pair and propose a connection.
    
    L's directive: Track similarity_method ("lvs" or "embedding")
    """
    proposals = []
    
    log(f"Analyzing {len(components)} components for bridges...")
    if use_embeddings:
        log("Embedding fallback ENABLED for nodes without LVS coordinates")
    
    # Only analyze component pairs (n^2 but n is small for components)
    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            if j <= i:
                continue  # Skip self and already-checked pairs
            
            best_sim = 0.0
            best_pair = None
            best_method = "lvs"
            
            # Find best bridge between these components
            for node1 in list(comp1)[:20]:  # Limit for performance
                coords1 = get_node_coords(lvs_index, node1)
                    
                for node2 in list(comp2)[:20]:
                    coords2 = get_node_coords(lvs_index, node2)
                    
                    sim = 0.0
                    method = "lvs"
                    
                    # Try LVS first
                    if coords1 and coords2:
                        sim = calculate_lvs_similarity(coords1, coords2)
                        method = "lvs"
                    
                    # Embedding fallback for nodes without LVS
                    elif use_embeddings and (not coords1 or not coords2):
                        embed_sim, success = calculate_embedding_similarity(node1, node2)
                        if success:
                            sim = embed_sim
                            method = "embedding"
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (node1, node2)
                        best_method = method
            
            # If we found a viable bridge
            if best_pair and best_sim >= MIN_SIMILARITY:
                node1, node2 = best_pair
                domain1 = get_domain(node1)
                domain2 = get_domain(node2)
                same_domain = domain1 == domain2
                
                # =============================================================
                # L's TIERED SYSTEM (2026-01-20)
                # =============================================================
                
                if best_sim >= DUPLICATE_THRESHOLD:
                    # > 0.95: Suspiciously high — likely duplicate
                    proposal_type = "duplicate"
                    action_tier = "auto_exclude"
                    reason = f"SUSPICIOUS: {best_sim:.0%} similarity suggests duplicate/backup, not true connection"
                    status = "excluded"
                    
                elif best_sim >= REVIEW_THRESHOLD:
                    # 0.85 - 0.95: Strong candidate
                    # AMENDED 2026-01-26: Only same-domain needs review (potential merge)
                    # Cross-domain is analogy — auto-apply to heal topology
                    if same_domain:
                        proposal_type = "merge"
                        reason = f"Same domain ({domain1}), strong similarity — potential true synonymy"
                        action_tier = "review"
                        status = "pending"
                    else:
                        # Cross-domain: auto-apply as strong analogy (V's healing fix)
                        proposal_type = "analogy"
                        reason = f"Cross-domain ({domain1} ↔ {domain2}), strong similarity — auto-apply analogy bridge"
                        action_tier = "auto_apply"
                        status = "auto_approved"
                    
                elif best_sim >= ANALOGY_THRESHOLD:
                    # 0.70 - 0.85: Auto-apply weak analogy edge
                    proposal_type = "analogy"
                    action_tier = "auto_apply"
                    reason = f"Cross-domain resonance ({domain1} ↔ {domain2}) — weak edge, tentative connection"
                    status = "auto_approved"
                    
                else:
                    continue  # Below threshold
                
                proposals.append(SynonymyProposal(
                    source=node1,
                    target=node2,
                    similarity=round(best_sim, 4),
                    proposal_type=proposal_type,
                    action_tier=action_tier,
                    source_domain=domain1,
                    target_domain=domain2,
                    same_domain=same_domain,
                    reason=reason,
                    created=datetime.now().isoformat(),
                    similarity_method=best_method,
                    status=status
                ))
    
    return proposals


def analyze_threshold_distribution(
    components: List[Set[str]], 
    lvs_index: Dict
) -> Dict:
    """
    Analyze similarity distribution to understand the 0.85 boundary.
    
    L's hypothesis: 0.85 is a phase boundary where local structure saturates.
    """
    similarities = []
    
    # Sample cross-component similarities
    for i, comp1 in enumerate(components[:20]):  # Limit for performance
        for j, comp2 in enumerate(components[:20]):
            if j <= i:
                continue
            
            for node1 in list(comp1)[:5]:  # Sample nodes
                coords1 = get_node_coords(lvs_index, node1)
                if not coords1:
                    continue
                    
                for node2 in list(comp2)[:5]:
                    coords2 = get_node_coords(lvs_index, node2)
                    if not coords2:
                        continue
                    
                    sim = calculate_lvs_similarity(coords1, coords2)
                    if sim > 0:
                        similarities.append(sim)
    
    if not similarities:
        return {"error": "No similarities computed"}
    
    # Distribution analysis
    import statistics
    
    above_85 = sum(1 for s in similarities if s >= 0.85)
    between_70_85 = sum(1 for s in similarities if 0.70 <= s < 0.85)
    below_70 = sum(1 for s in similarities if s < 0.70)
    
    return {
        "total_pairs_sampled": len(similarities),
        "mean_similarity": round(statistics.mean(similarities), 4),
        "median_similarity": round(statistics.median(similarities), 4),
        "stdev": round(statistics.stdev(similarities), 4) if len(similarities) > 1 else 0,
        "max_similarity": round(max(similarities), 4),
        "above_0.85": above_85,
        "between_0.70_0.85": between_70_85,
        "below_0.70": below_70,
        "phase_boundary_hypothesis": {
            "threshold": 0.85,
            "pairs_above": above_85,
            "percent_above": round(100 * above_85 / len(similarities), 2),
            "interpretation": "These are the bridgeable gaps"
        }
    }


def save_proposals(proposals: List[SynonymyProposal], analysis: Dict):
    """Save proposals to file for review."""
    output = {
        "generated": datetime.now().isoformat(),
        "threshold_analysis": analysis,
        "tiered_thresholds": {
            "duplicate": f">{DUPLICATE_THRESHOLD:.0%} — auto-exclude",
            "review": f"{REVIEW_THRESHOLD:.0%}-{DUPLICATE_THRESHOLD:.0%} — human review",
            "auto_apply": f"{ANALOGY_THRESHOLD:.0%}-{REVIEW_THRESHOLD:.0%} — weak edge",
            "ignore": f"<{ANALOGY_THRESHOLD:.0%} — noise"
        },
        "proposals": [asdict(p) for p in proposals],
        "summary": {
            "total_proposals": len(proposals),
            "by_type": {
                "merge": sum(1 for p in proposals if p.proposal_type == "merge"),
                "analogy": sum(1 for p in proposals if p.proposal_type == "analogy"),
                "duplicate": sum(1 for p in proposals if p.proposal_type == "duplicate"),
            },
            "by_tier": {
                "auto_exclude": sum(1 for p in proposals if p.action_tier == "auto_exclude"),
                "review": sum(1 for p in proposals if p.action_tier == "review"),
                "auto_apply": sum(1 for p in proposals if p.action_tier == "auto_apply"),
            },
            "by_method": {
                "lvs": sum(1 for p in proposals if p.similarity_method == "lvs"),
                "embedding": sum(1 for p in proposals if p.similarity_method == "embedding"),
            }
        }
    }
    
    with open(SYNONYMY_PROPOSALS, 'w') as f:
        json.dump(output, f, indent=2)
    
    log(f"Saved {len(proposals)} proposals to {SYNONYMY_PROPOSALS}")


def print_proposals(proposals: List[SynonymyProposal]):
    """Print proposals in readable format, organized by L's tiers."""
    print("\n" + "="*70)
    print("SYNONYMY PROPOSALS — L's Tiered System")
    print("="*70)
    
    # Group by action tier
    excluded = [p for p in proposals if p.action_tier == "auto_exclude"]
    review = [p for p in proposals if p.action_tier == "review"]
    auto_apply = [p for p in proposals if p.action_tier == "auto_apply"]
    
    # Count by method
    lvs_count = sum(1 for p in proposals if p.similarity_method == "lvs")
    embed_count = sum(1 for p in proposals if p.similarity_method == "embedding")
    
    print(f"\nMethods: {lvs_count} LVS (📐), {embed_count} embedding (🧠)")
    print(f"Tiers: {len(excluded)} excluded, {len(review)} for review, {len(auto_apply)} auto-apply")
    
    if excluded:
        print(f"\n⛔ AUTO-EXCLUDED ({len(excluded)}) — >95% = Suspicious Duplicates")
        print("-"*50)
        for p in sorted(excluded, key=lambda x: -x.similarity)[:5]:
            method_tag = "📐" if p.similarity_method == "lvs" else "🧠"
            print(f"  {Path(p.source).stem} ≈ {Path(p.target).stem}")
            print(f"    {method_tag} {p.similarity:.2%} — {p.reason[:60]}")
            print()
    
    if review:
        print(f"\n🔍 NEEDS REVIEW ({len(review)}) — 85-95% = Strong Candidates")
        print("-"*50)
        for p in sorted(review, key=lambda x: -x.similarity):
            method_tag = "📐" if p.similarity_method == "lvs" else "🧠"
            type_icon = "🔗" if p.proposal_type == "merge" else "🌉"
            print(f"  {type_icon} {Path(p.source).stem}")
            print(f"      ↔ {Path(p.target).stem}")
            print(f"    {method_tag} {p.similarity:.2%} | {p.source_domain} ↔ {p.target_domain}")
            print()
    
    if auto_apply:
        print(f"\n✅ AUTO-APPLY ({len(auto_apply)}) — 70-85% = Weak Analogy Edges")
        print("-"*50)
        for p in sorted(auto_apply, key=lambda x: -x.similarity)[:15]:
            method_tag = "📐" if p.similarity_method == "lvs" else "🧠"
            print(f"  {Path(p.source).stem} ({p.source_domain})")
            print(f"    ↔ {Path(p.target).stem} ({p.target_domain})")
            print(f"    {method_tag} {p.similarity:.2%} — weak edge")
            print()


# =============================================================================
# BRIDGE APPLICATION — Actually write [[links]] to files
# =============================================================================

def find_linkage_section(content: str) -> Tuple[int, int]:
    """
    Find the LINKAGE section in file content.
    Returns (start_index, end_index) or (-1, -1) if not found.
    """
    import re
    
    # Find ## LINKAGE header
    match = re.search(r'^## LINKAGE.*$', content, re.MULTILINE)
    if not match:
        return (-1, -1)
    
    start = match.start()
    
    # Find the end (next ## header or end of file)
    rest = content[match.end():]
    next_header = re.search(r'^## ', rest, re.MULTILINE)
    if next_header:
        end = match.end() + next_header.start()
    else:
        end = len(content)
    
    return (start, end)


def create_linkage_section(target_path: str, similarity: float, method: str) -> str:
    """Create a new LINKAGE section with the analogy link."""
    return f"""

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Analogy | [[{target_path}]] | <!-- auto: {similarity:.2f} via {method} -->

"""


def add_link_to_linkage(content: str, linkage_start: int, linkage_end: int, 
                        target_path: str, similarity: float, method: str) -> str:
    """Add a link row to existing LINKAGE section."""
    linkage_content = content[linkage_start:linkage_end]
    
    # Find the last row of the table (look for last |...|)
    import re
    table_rows = list(re.finditer(r'^\|[^|]+\|[^|]+\|.*$', linkage_content, re.MULTILINE))
    
    if table_rows:
        # Insert after last row
        last_row = table_rows[-1]
        insert_pos = linkage_start + last_row.end()
        new_row = f"\n| Analogy | [[{target_path}]] | <!-- auto: {similarity:.2f} via {method} -->"
        return content[:insert_pos] + new_row + content[insert_pos:]
    else:
        # No table found, append to section
        new_row = f"\n| Direction | Seed |\n|-----------|------|\n| Analogy | [[{target_path}]] | <!-- auto: {similarity:.2f} via {method} -->\n"
        return content[:linkage_end] + new_row + content[linkage_end:]


def apply_weak_edge(source_path: str, target_path: str, similarity: float, method: str) -> bool:
    """
    Apply a weak analogy edge by adding [[link]] to source file.
    
    Returns True if successfully applied, False otherwise.
    """
    full_source = TABERNACLE / source_path
    
    if not full_source.exists():
        log(f"Source file not found: {source_path}", "ERROR")
        return False
    
    try:
        content = full_source.read_text(encoding='utf-8')
        
        # Check if link already exists
        if f"[[{target_path}]]" in content or f"[[{Path(target_path).stem}]]" in content:
            log(f"Link already exists: {source_path} → {target_path}", "DEBUG")
            return False
        
        # Find or create LINKAGE section
        start, end = find_linkage_section(content)
        
        if start >= 0:
            # Add to existing LINKAGE section
            new_content = add_link_to_linkage(content, start, end, target_path, similarity, method)
        else:
            # Create new LINKAGE section at end
            new_content = content.rstrip() + create_linkage_section(target_path, similarity, method)
        
        # Write back
        full_source.write_text(new_content, encoding='utf-8')
        log(f"Applied weak edge: {Path(source_path).stem} → {Path(target_path).stem} ({similarity:.0%})")
        return True
        
    except Exception as e:
        log(f"Failed to apply edge {source_path} → {target_path}: {e}", "ERROR")
        return False


def apply_auto_proposals(proposals: List[SynonymyProposal]) -> int:
    """
    Apply all auto-approve proposals by adding links to files.
    Returns count of successfully applied edges.

    AMENDED 2026-01-26: Also updates persistent graph to keep graph.pkl in sync.
    """
    auto_apply = [p for p in proposals if p.action_tier == "auto_apply"]

    if not auto_apply:
        log("No auto-apply proposals to apply")
        return 0

    log(f"Applying {len(auto_apply)} weak edges...")
    applied = 0
    applied_edges = []  # Track for graph persistence

    for p in auto_apply:
        # Apply bidirectional? For now, just source → target
        if apply_weak_edge(p.source, p.target, p.similarity, p.similarity_method):
            applied += 1
            applied_edges.append((p.source, p.target, p.similarity, p.similarity_method))

    # Update persistent graph to keep graph.pkl in sync
    if applied_edges:
        try:
            from graph_persist import PersistentGraph
            pg = PersistentGraph(auto_save=False)
            for source, target, similarity, method in applied_edges:
                pg.add_edge(source, target,
                           weight=similarity,
                           edge_type="analogy",
                           method=method,
                           created=datetime.now().isoformat())
            pg.save(force=True)
            log(f"Graph persistence: {applied} edges synced to graph.pkl")
        except Exception as e:
            log(f"Graph persistence failed (edges still in files): {e}", "WARN")

    log(f"Applied {applied}/{len(auto_apply)} weak edges")
    return applied


def run_daemon(apply: bool = False):
    """
    Run the synonymy detection daemon.
    
    Args:
        apply: If True, auto-apply weak edges to files
    """
    print("="*70)
    print("SYNONYMY DAEMON v0.2 — L's Amendment")
    print("Building bridges between disconnected components")
    if apply:
        print("MODE: APPLY (will write links to files)")
    else:
        print("MODE: DETECT ONLY (no changes)")
    print("="*70)
    
    # Load data
    graph = load_graph()
    lvs_index = load_lvs_index()
    
    if not graph:
        log("Failed to build graph", "ERROR")
        return
    
    # Find disconnected components
    components = find_components(graph)
    log(f"Found {len(components)} disconnected components (H₀)")
    
    if len(components) <= 1:
        log("Graph is fully connected. No bridges needed.")
        return
    
    # Analyze threshold distribution (L's 0.85 hypothesis)
    log("Analyzing similarity distribution (0.85 boundary hypothesis)...")
    analysis = analyze_threshold_distribution(components, lvs_index)
    
    print("\n📊 THRESHOLD ANALYSIS")
    print(f"  Pairs sampled: {analysis.get('total_pairs_sampled', 0)}")
    print(f"  Mean similarity: {analysis.get('mean_similarity', 0):.2%}")
    print(f"  Max similarity: {analysis.get('max_similarity', 0):.2%}")
    print(f"  Above 0.85: {analysis.get('above_0.85', 0)} pairs")
    print(f"  Between 0.70-0.85: {analysis.get('between_0.70_0.85', 0)} pairs")
    
    # Find bridge proposals
    log("Finding cross-component bridges...")
    proposals = find_cross_component_bridges(components, lvs_index)
    
    # Print and save
    print_proposals(proposals)
    save_proposals(proposals, analysis)
    
    # Apply weak edges if requested
    applied_count = 0
    if apply:
        print("\n" + "-"*70)
        print("APPLYING WEAK EDGES")
        print("-"*70)
        applied_count = apply_auto_proposals(proposals)
    
    print("\n" + "="*70)
    print(f"COMPLETE: {len(proposals)} proposals generated")
    if apply:
        print(f"APPLIED: {applied_count} weak edges written to files")
    print(f"Review at: {SYNONYMY_PROPOSALS}")
    print("="*70)
    
    return {
        "proposals": len(proposals),
        "applied": applied_count,
        "h0": len(components)
    }


if __name__ == "__main__":
    import sys
    
    apply_mode = "--apply" in sys.argv or "-a" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("SYNONYMY DAEMON — Bridge Builder")
        print()
        print("Usage: python synonymy_daemon.py [OPTIONS]")
        print()
        print("Options:")
        print("  --apply, -a    Apply auto-approve weak edges to files")
        print("  --help, -h     Show this help")
        print()
        print("Without --apply, daemon runs in detect-only mode.")
    else:
        run_daemon(apply=apply_mode)
