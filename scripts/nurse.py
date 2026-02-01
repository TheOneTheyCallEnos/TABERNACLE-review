#!/usr/bin/env python3
"""
PROJECT TABERNACLE: THE NURSE (Asclepius Protocol v1.1)
The Stone Witness - Deterministic Somatic Diagnostics

This script provides the BODY's truth. It cannot hallucinate.
It counts, measures, and hashes. The Dream Witness (LLM) interprets.

v1.1 Changes:
- Fixed false positive link detection (templates, arrays, directories)
- Exclude CRYPT from active health scoring (it's archive)
- Calibrated metabolic formula (active system = healthy)
- Added Semantic Gravity metric (intra-document density)

Output: vitals.json - The blood work of the Tabernacle.
"""

import os
import sys
import json
import hashlib
import datetime
import subprocess
import re
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Set
import math

# --- CONFIGURATION (from centralized config) ---
from tabernacle_config import (
    BASE_DIR, NEXUS_DIR, STRUCTURE_DIR, LAW_DIR, LOG_DIR, SCRIPTS_DIR,
    ACTIVE_QUADRANTS, SKIP_DIRECTORIES, TRANSIENT_FILES, EXAMPLE_LINK_PATTERNS,
    LVS_INDEX_PATH, is_mcp_mode,
    PLOCK_THRESHOLD, ABADDON_P_THRESHOLD, ABADDON_E_THRESHOLD, RECOVERY_THRESHOLD,
)

CANON_DIR = LAW_DIR / "CANON"

# Directories to exclude from scanning (use centralized + extras)
EXCLUDE_DIRS = SKIP_DIRECTORIES | {".cursor"}

# Files to exclude from link analysis (meta-files that list links as content)
EXCLUDE_FILES = {
    "_LINK_DIAGNOSIS.md",    # Meta-file that lists links as content
    "_LINK_FIXES.md",        # Meta-file that lists links as content  
    "apply_link_fixes.md",   # Contains example links
    "diagnose_links.md",     # Contains example links
    "STRUCTURAL_INTEGRITY_REPORT.md",  # Contains example/template links
}

# Directories that are ARCHIVE (don't count broken links against health)
ARCHIVE_DIRS = {"05_CRYPT", "logs", "archives", ".venv"}

# Convert to set for nurse.py compatibility
ACTIVE_QUADRANTS = set(ACTIVE_QUADRANTS)

LARGE_FILE_THRESHOLD = 100 * 1024  # 100KB

# Homeostatic Ranges (CALIBRATED for active development)
# An active living system has 20-60 commits/day during work
IDEAL_DAILY_EDITS_LOW = 5      # Below this = stagnant
IDEAL_DAILY_EDITS_HIGH = 80    # Above this = manic
ORPHAN_THRESHOLD_WARNING = 15
ORPHAN_THRESHOLD_CRITICAL = 30
BROKEN_LINK_THRESHOLD = 40  # Allow some false positives from legacy content
COMMUNION_DECAY_HOURS = 24

# Valid file extensions that can be linked (not just .md)
VALID_LINK_EXTENSIONS = {".md", ".py", ".json", ".yaml", ".yml", ".sh", ".html", ".js", ".ts"}

# Import shared link utilities (consolidated from nurse.py + diagnose_links.py)
from tabernacle_utils import (
    is_valid_wiki_link as _is_valid_wiki_link,
    extract_wiki_links as _extract_wiki_links,
    resolve_link as _resolve_link,
    has_linkage_block as _has_linkage_block,
    find_all_md_files as _find_all_md_files,
)


# --- DATA STRUCTURES ---
@dataclass
class Vitals:
    """The somatic state of the Tabernacle."""
    timestamp: str
    
    # Structural (Skeleton)
    total_files: int = 0
    total_md_files: int = 0
    active_md_files: int = 0      # Excluding CRYPT
    total_links: int = 0
    valid_links: int = 0
    broken_links: int = 0         # Only in active quadrants
    orphan_count: int = 0
    files_without_linkage: int = 0
    large_files: int = 0
    duplicate_index_files: int = 0
    
    # Metabolic (Circulation)
    commits_24h: int = 0
    commits_7d: int = 0
    files_modified_24h: int = 0
    files_modified_7d: int = 0
    daily_edit_rate: float = 0.0
    metabolic_state: str = "healthy"  # stagnant, healthy, manic
    oldest_untouched_days: int = 0
    dead_quadrants: List[str] = field(default_factory=list)
    
    # Immune (Integrity)
    canon_hash: str = ""
    structure_hash: str = ""
    skill_count: int = 0
    skills_synced: int = 0
    skills_missing_metadata: int = 0
    
    # Dyadic (Connection)
    hours_since_communion: float = 0.0
    session_buffer_size_kb: float = 0.0
    last_heartbeat_age_seconds: int = 0
    
    # Graph Topology
    most_connected_files: List[Dict] = field(default_factory=list)
    isolated_clusters: int = 0
    avg_links_per_file: float = 0.0
    
    # Semantic Gravity (NEW)
    avg_semantic_density: float = 0.0
    high_gravity_files: List[Dict] = field(default_factory=list)
    low_gravity_files: List[Dict] = field(default_factory=list)
    
    # LVS v10 Metrics (Canon-compliant)
    lvs_nodes_indexed: int = 0
    avg_lvs_coherence: float = 0.0     # p = (κ·ρ·σ·τ)^0.25
    avg_lvs_epsilon: float = 0.0        # ε (energy)
    plock_nodes: int = 0                # p ≥ 0.95
    abaddon_nodes: int = 0              # p < 0.50 OR ε < 0.40
    recovery_nodes: int = 0             # ε < 0.65
    
    # Topology / Persistence (Persistent Virgil)
    topology_h0: int = 0                # Connected components
    topology_h1: int = 0                # Cycles (identity-critical)
    topology_drift_score: float = 0.0   # Drift from canonical
    
    # Computed Scores
    structural_score: float = 0.0
    metabolic_score: float = 0.0
    immune_score: float = 0.0
    dyadic_score: float = 0.0
    semantic_gravity_score: float = 0.0
    lvs_health_score: float = 0.0       # NEW: Based on Canon thresholds
    vitality_score: float = 0.0
    
    # Alerts
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


# --- UTILITY FUNCTIONS ---
def log(message: str):
    """Log to file only. Silent in MCP mode."""
    # Check if we're in MCP mode - suppress ALL output including stderr
    if os.environ.get("TABERNACLE_MCP_MODE"):
        return
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [NURSE] {message}"
    print(entry, file=sys.stderr)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "nurse.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Silent fail


def hash_directory(directory: Path) -> str:
    """Create a hash of all files in a directory for integrity checking."""
    if not directory.exists():
        return "MISSING"
    
    hasher = hashlib.sha256()
    for filepath in sorted(directory.rglob("*.md")):
        if any(ex in str(filepath) for ex in EXCLUDE_DIRS):
            continue
        try:
            hasher.update(filepath.read_bytes())
        except (IOError, OSError):
            pass
    return hasher.hexdigest()[:16]


def is_in_archive(filepath: Path) -> bool:
    """Check if file is in archive (CRYPT, logs, etc.)"""
    rel_path = str(filepath.relative_to(BASE_DIR)) if filepath.is_relative_to(BASE_DIR) else str(filepath)
    return any(archive in rel_path for archive in ARCHIVE_DIRS)


def is_in_active_quadrant(filepath: Path) -> bool:
    """Check if file is in an active quadrant (not archive)."""
    rel_path = str(filepath.relative_to(BASE_DIR)) if filepath.is_relative_to(BASE_DIR) else str(filepath)
    return any(rel_path.startswith(quad) for quad in ACTIVE_QUADRANTS)


def find_all_md_files(root: Path) -> List[Path]:
    """Recursively find all .md files."""
    md_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            if filename.endswith(".md"):
                md_files.append(Path(dirpath) / filename)
    return md_files


# --- LINK UTILITIES (delegating to tabernacle_utils) ---
# These are thin wrappers for backward compatibility

def is_valid_wiki_link(link: str) -> bool:
    """Check if a wiki-style link is valid. Delegates to tabernacle_utils."""
    # Nurse-specific extra checks
    link = link.strip()
    if not link or link.isspace():
        return False
    if len(link) <= 2:
        return False
    if link.isdigit():
        return False
    if link in {"...", "…"}:
        return False
    if link.startswith("../"):
        return False
    # Delegate to shared implementation
    return _is_valid_wiki_link(link)


def extract_wiki_links(content: str) -> List[str]:
    """Extract valid [[wiki-style]] links. Delegates to tabernacle_utils."""
    return _extract_wiki_links(content)


def has_linkage_block(content: str) -> bool:
    """Check if file has a LINKAGE block. Delegates to tabernacle_utils."""
    return _has_linkage_block(content)


def resolve_link(source_file: Path, link: str) -> Optional[Path]:
    """
    Resolve a wiki-style link to an absolute path.
    Now handles: .md files, directories, scripts, section anchors, and other file types.
    (Synced with diagnose_links.py logic)
    """
    link = link.strip()
    
    # Check if it's an example/placeholder link (synced from diagnose_links.py)
    for pattern in EXAMPLE_LINK_PATTERNS:
        if pattern in link.lower():
            return Path("__EXAMPLE__")  # Return sentinel to mark as "valid" (ignored)
    
    # Handle section links (remove #section suffix) - CRITICAL FIX
    if "#" in link:
        link = link.split("#")[0]
        if not link:  # Link was just "#section" with no file
            return Path("__SELF_ANCHOR__")  # Self-reference is valid
    
    # Check if it's a directory link (ends with / or is a known directory pattern)
    if link.endswith("/"):
        dir_path = BASE_DIR / link.rstrip("/")
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
        return None
    
    # Check for directory-only links (quadrant references)
    if link in {"00_NEXUS", "01_UL_INTENT", "02_UR_STRUCTURE", "03_LL_RELATION", "04_LR_LAW", "05_CRYPT"}:
        dir_path = BASE_DIR / link
        if dir_path.exists():
            return dir_path
        return None
    
    # Determine if it has an extension
    has_extension = "." in Path(link).name
    
    if has_extension:
        ext = Path(link).suffix.lower()
        
        # Valid extensions including images and data files
        valid_exts = VALID_LINK_EXTENSIONS | {".png", ".jpg", ".jpeg", ".gif", ".svg", ".yaml", ".yml", ".json"}
        
        if ext in valid_exts:
            # Try absolute from TABERNACLE root
            absolute_path = BASE_DIR / link
            if absolute_path.exists():
                return absolute_path
            
            # Try relative to source
            relative_path = source_file.parent / link
            if relative_path.exists():
                return relative_path
            
            # For scripts, also try just the scripts directory
            if link.startswith("scripts/"):
                scripts_path = BASE_DIR / link
                if scripts_path.exists():
                    return scripts_path
    else:
        # No extension - assume .md
        link_md = link + ".md"
        
        # Try absolute from TABERNACLE root
        absolute_path = BASE_DIR / link_md
        if absolute_path.exists():
            return absolute_path
        
        # Try relative to source
        relative_path = source_file.parent / link_md
        if relative_path.exists():
            return relative_path
        
        # Also check if it's a directory
        dir_path = BASE_DIR / link
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
    
    # Filename search (last resort) - but skip for directories and common patterns
    if not link.endswith("/") and "/" not in link:
        search_name = Path(link).name
        if not has_extension:
            search_name = search_name + ".md"
        
        for dirpath, dirnames, filenames in os.walk(BASE_DIR):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            if search_name in filenames:
                return Path(dirpath) / search_name
    
    return None


def calculate_semantic_gravity(content: str, filepath: Path) -> Dict:
    """
    Calculate Semantic Gravity (ρ_sem) for a document.
    
    Semantic Gravity = Internal concept density
    High gravity = holographic document (concepts interconnect)
    Low gravity = linear document (concepts don't reference each other)
    
    We measure:
    1. Header structure (hierarchy depth)
    2. Internal references (same-doc anchors)
    3. Concept repetition (key terms recurring)
    4. Cross-section references
    """
    result = {
        "filepath": str(filepath.relative_to(BASE_DIR)) if filepath.is_relative_to(BASE_DIR) else str(filepath),
        "header_depth": 0,
        "internal_refs": 0,
        "concept_density": 0.0,
        "semantic_gravity": 0.0
    }
    
    lines = content.split("\n")
    word_count = len(content.split())
    
    if word_count < 100:
        # Too short to measure meaningfully
        return result
    
    # 1. Header structure (hierarchical depth)
    headers = []
    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            headers.append(level)
    
    if headers:
        result["header_depth"] = max(headers)
        header_variance = len(set(headers))  # Variety of header levels
    else:
        header_variance = 0
    
    # 2. Internal references (anchors like #section-name)
    internal_refs = re.findall(r"\]\(#[^)]+\)", content)
    anchor_links = re.findall(r"\[.*?\]\(#", content)
    result["internal_refs"] = len(internal_refs) + len(anchor_links)
    
    # 3. Concept repetition (key terms)
    # Extract significant words (capitalized, technical terms)
    words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", content)  # CamelCase
    words += re.findall(r"\b[A-Z]{2,}\b", content)  # ACRONYMS
    words += re.findall(r"\*\*([^*]+)\*\*", content)  # Bold terms
    
    if words:
        word_freq = Counter(words)
        # Count terms that appear 3+ times (recurring concepts)
        recurring = sum(1 for w, c in word_freq.items() if c >= 3)
        result["concept_density"] = recurring / (word_count / 100)  # Per 100 words
    
    # 4. Calculate Semantic Gravity Score (0-1)
    # Weighted combination:
    # - Header depth (structure)
    # - Internal refs (self-reference)
    # - Concept density (term recurrence)
    
    header_score = min(result["header_depth"] / 4, 1.0)  # Max at 4 levels
    ref_score = min(result["internal_refs"] / 10, 1.0)  # Max at 10 refs
    density_score = min(result["concept_density"] / 5, 1.0)  # Max at 5 per 100 words
    
    # Geometric mean (all must be present for high score)
    if header_score > 0 and density_score > 0:
        gravity = (header_score * 0.3 + ref_score * 0.3 + density_score * 0.4)
    else:
        gravity = (header_score + ref_score + density_score) / 3
    
    result["semantic_gravity"] = round(gravity, 3)
    
    return result


# --- DIAGNOSTIC FUNCTIONS ---

def check_structural_integrity(vitals: Vitals, md_files: List[Path]):
    """Layer 1: Structural (Skeleton) - Graph topology, link integrity."""
    log("Checking structural integrity...")
    
    total_links = 0
    valid_links = 0
    broken_links = 0  # Only count in ACTIVE quadrants
    files_without_linkage = 0
    incoming_links = defaultdict(list)
    large_files = 0
    index_files = []
    active_files = 0
    
    # Files to skip from link analysis (large archives, extractions, meta-files)
    skip_link_analysis = {
        "SONSHIP_FULL_EXTRACTION.md", "SOURCE_OF_IDEAS_FULL_EXTRACTION.md",
        "HOOKS_SOBRIETY_FULL_EXTRACTION.md", "QUARANTINE_FULL_EXTRACTION.md",
        "ChatGPT_Final_Export.md", "ChatGPT_History.md", "Gemini_Memory_Bank.md"
    } | EXCLUDE_FILES  # Add meta-files that contain example links
    
    for filepath in md_files:
        is_active = is_in_active_quadrant(filepath)
        if is_active:
            active_files += 1
        
        # Skip large archive files from link counting
        if filepath.name in skip_link_analysis:
            continue
        
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            size = filepath.stat().st_size
            
            if size > LARGE_FILE_THRESHOLD:
                large_files += 1
            
            if filepath.name == "INDEX.md":
                index_files.append(str(filepath.relative_to(BASE_DIR)))
            
            # Check LINKAGE block (only for active quadrants, skip generated files)
            rel_path = str(filepath.relative_to(BASE_DIR))
            # Transient/auto-generated files that don't need LINKAGE blocks
            skip_linkage_check = [
                "05_CRYPT/", "logs/", "scripts/", "node_modules", "TASKS/",
                "_VITALS_REPORT.md", "_LINK_DIAGNOSIS.md", "_LINK_FIXES.md",
                "_GRAPH_ATLAS.md", "_SEMANTIC_DIAGNOSIS.md", "_ASCLEPIUS_EXAMEN.md",
                "DAEMON_REFLECTION.md", "SYSTEM_STATUS.md", "vitals.json",
                "CURRENT_STATE.md", "SESSION_BUFFER.md", "OUTBOX.md",
                "LAST_COMMUNION.md", "NIGHT_DAEMON_HANDOFF.md", "NIGHT_SYNTHESIS.md",
                "CONVERSATION.md", "VIRGIL_INTENTIONS.md", "NEURAL_STATE.json"
            ]
            
            if is_active and not any(skip in rel_path for skip in skip_linkage_check):
                if not has_linkage_block(content):
                    files_without_linkage += 1
            
            # Extract and verify links
            links = extract_wiki_links(content)
            total_links += len(links)
            
            for link in links:
                resolved = resolve_link(filepath, link)
                if resolved:
                    # Check for sentinel values (example links, self-anchors)
                    resolved_str = str(resolved)
                    if resolved_str.startswith("__") and resolved_str.endswith("__"):
                        # Sentinel value - counts as valid but don't track incoming
                        valid_links += 1
                    elif resolved.is_file():
                        valid_links += 1
                        target_rel = str(resolved.relative_to(BASE_DIR)) if resolved.is_relative_to(BASE_DIR) else str(resolved)
                        incoming_links[target_rel].append(rel_path)
                    elif resolved.is_dir():
                        valid_links += 1
                        # Directory links don't contribute to incoming file links
                else:
                    # Only count as broken if source is in ACTIVE quadrant
                    # and not in TASKS (archived task files may have stale links)
                    if is_active and "TASKS/" not in rel_path:
                        broken_links += 1
                        
        except Exception as e:
            log(f"Error processing {filepath}: {e}")
    
    # Count orphans (files with no incoming links) - ACTIVE only
    orphan_count = 0
    # Root/transient files that are OK to have no incoming links
    root_files = {
        "00_NEXUS/CURRENT_STATE.md", "00_NEXUS/_GRAPH_ATLAS.md", "README.md",
        "00_NEXUS/_VITALS_REPORT.md", "00_NEXUS/_LINK_DIAGNOSIS.md",
        "00_NEXUS/_SEMANTIC_DIAGNOSIS.md", "00_NEXUS/_ASCLEPIUS_EXAMEN.md",
        "00_NEXUS/_LINK_FIXES.md", "CHANGELOG.md", "TABERNACLE_MAP.md",
        # Transient/volatile files
        "00_NEXUS/SESSION_BUFFER.md", "00_NEXUS/OUTBOX.md", "00_NEXUS/CONVERSATION.md",
        "00_NEXUS/LAST_COMMUNION.md", "00_NEXUS/NIGHT_DAEMON_HANDOFF.md",
        "00_NEXUS/NIGHT_SYNTHESIS.md", "00_NEXUS/VIRGIL_INTENTIONS.md",
        "00_NEXUS/DAEMON_REFLECTION.md", "00_NEXUS/SYSTEM_STATUS.md",
        "00_NEXUS/NEURAL_STATE.json", "00_NEXUS/DREAM_LOG.md"
    }
    
    # Directories that are OK to have orphans (tasks, transient)
    orphan_ok_dirs = ["TASKS/", "DEEP_THOUGHTS/"]
    
    for filepath in md_files:
        if not is_in_active_quadrant(filepath):
            continue
        rel_path = str(filepath.relative_to(BASE_DIR))
        if rel_path in root_files:
            continue
        # Skip files in orphan-OK directories
        if any(ok_dir in rel_path for ok_dir in orphan_ok_dirs):
            continue
        if rel_path not in incoming_links:
            orphan_count += 1
    
    # Find most connected files
    connection_counts = [(path, len(sources)) for path, sources in incoming_links.items()]
    connection_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Detect duplicate INDEX.md files per directory level
    index_dirs = [str(Path(p).parent) for p in index_files]
    duplicate_indices = len(index_dirs) - len(set(index_dirs))
    
    # Update vitals
    vitals.total_links = total_links
    vitals.valid_links = valid_links
    vitals.broken_links = broken_links
    vitals.orphan_count = orphan_count
    vitals.files_without_linkage = files_without_linkage
    vitals.large_files = large_files
    vitals.duplicate_index_files = duplicate_indices
    vitals.active_md_files = active_files
    vitals.most_connected_files = [{"path": p, "incoming": c} for p, c in connection_counts[:10]]
    vitals.avg_links_per_file = total_links / len(md_files) if md_files else 0
    
    # Structural score (CALIBRATED for real-world vault)
    # Valid links ratio (most important)
    link_health = valid_links / max(total_links, 1)
    # Orphan penalty (very gentle - some orphans are OK)
    orphan_health = 1.0 - min(orphan_count / 100, 0.3)  # Max 30% penalty
    # Broken links penalty (gentle - allow legacy content)
    broken_penalty = min(broken_links / 100, 0.2)  # Max 20% penalty
    
    vitals.structural_score = max(link_health * 0.6 + orphan_health * 0.25 + (1 - broken_penalty) * 0.15, 0)
    
    # Generate alerts
    if broken_links > BROKEN_LINK_THRESHOLD:
        vitals.warnings.append(f"Broken links in active quadrants: {broken_links}")
    if orphan_count > ORPHAN_THRESHOLD_CRITICAL:
        vitals.critical_issues.append(f"Critical orphan count: {orphan_count}")
    elif orphan_count > ORPHAN_THRESHOLD_WARNING:
        vitals.warnings.append(f"Orphan count: {orphan_count}")
    
    vitals.insights.append(f"Link integrity: {valid_links}/{total_links} valid ({link_health*100:.1f}%)")


def check_metabolic_rate(vitals: Vitals):
    """Layer 2: Metabolic (Circulation) - Rate of change, freshness."""
    log("Checking metabolic rate...")
    
    try:
        # Git commits in last 24h
        result = subprocess.run(
            ["git", "log", "--oneline", "--since=24.hours.ago"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=30
        )
        vitals.commits_24h = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        
        # Git commits in last 7d
        result = subprocess.run(
            ["git", "log", "--oneline", "--since=7.days.ago"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=30
        )
        vitals.commits_7d = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        
        # Files modified in last 24h
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=M", "HEAD@{1.day.ago}"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=30
        )
        vitals.files_modified_24h = len([f for f in result.stdout.strip().split("\n") if f])
        
        # Files modified in last 7d
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=M", "HEAD@{7.days.ago}"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=30
        )
        vitals.files_modified_7d = len([f for f in result.stdout.strip().split("\n") if f])
        
    except Exception as e:
        log(f"Git analysis failed: {e}")
    
    vitals.daily_edit_rate = vitals.commits_24h
    
    # Check for dead quadrants (no commits in 30+ days)
    quadrants = ["00_NEXUS", "01_UL_INTENT", "02_UR_STRUCTURE", "03_LL_RELATION", "04_LR_LAW"]
    dead_quadrants = []
    
    for quad in quadrants:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cr", "--", quad],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=10
            )
            last_commit = result.stdout.strip()
            if "month" in last_commit or "year" in last_commit:
                dead_quadrants.append(quad)
        except (subprocess.SubprocessError, OSError):
            pass
    
    vitals.dead_quadrants = dead_quadrants
    
    # Metabolic score (CALIBRATED for active development)
    # Plateau function: healthy between 5-80 commits/day
    rate = vitals.daily_edit_rate
    
    if rate < IDEAL_DAILY_EDITS_LOW:
        # Stagnant
        vitals.metabolic_state = "stagnant"
        vitals.metabolic_score = max(rate / IDEAL_DAILY_EDITS_LOW, 0.3)
    elif rate > IDEAL_DAILY_EDITS_HIGH:
        # Manic
        vitals.metabolic_state = "manic"
        vitals.metabolic_score = max(1.0 - (rate - IDEAL_DAILY_EDITS_HIGH) / 100, 0.5)
    else:
        # Healthy range
        vitals.metabolic_state = "healthy"
        vitals.metabolic_score = 1.0
    
    # Alerts
    if vitals.metabolic_state == "stagnant":
        vitals.warnings.append(f"Low activity: {rate} commits/day - system may be stagnant")
    elif vitals.metabolic_state == "manic":
        vitals.warnings.append(f"High activity: {rate} commits/day - consider consolidation")
    if dead_quadrants:
        vitals.warnings.append(f"Dead quadrants: {', '.join(dead_quadrants)}")
    
    vitals.insights.append(f"Metabolic state: {vitals.metabolic_state} ({rate} commits/24h)")


def check_immune_integrity(vitals: Vitals):
    """Layer 4: Immune - Hash verification, self vs non-self."""
    log("Checking immune integrity...")
    
    # Hash Canon (should be stable)
    vitals.canon_hash = hash_directory(CANON_DIR)
    vitals.structure_hash = hash_directory(STRUCTURE_DIR)
    
    # Check skills
    skills_dir = STRUCTURE_DIR / "SKILLS"
    claude_skills = Path.home() / ".claude" / "skills"
    
    skill_count = 0
    skills_synced = 0
    skills_missing_metadata = 0
    
    if skills_dir.exists():
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                skill_count += 1
                skill_file = skill_dir / "SKILL.md"
                
                if skill_file.exists():
                    content = skill_file.read_text(encoding="utf-8", errors="replace")
                    
                    # Check for NEUROPLASTICITY DATA
                    if "NEUROPLASTICITY DATA" not in content:
                        skills_missing_metadata += 1
                    
                    # Check if synced to ~/.claude/skills
                    claude_skill = claude_skills / skill_dir.name / "SKILL.md"
                    if claude_skill.exists():
                        skills_synced += 1
    
    vitals.skill_count = skill_count
    vitals.skills_synced = skills_synced
    vitals.skills_missing_metadata = skills_missing_metadata
    
    # Immune score
    sync_ratio = skills_synced / max(skill_count, 1)
    metadata_ratio = 1.0 - (skills_missing_metadata / max(skill_count, 1))
    vitals.immune_score = (sync_ratio * 0.5 + metadata_ratio * 0.5)
    
    # Alerts
    if skill_count > 0 and skills_synced < skill_count:
        vitals.warnings.append(f"Skills not synced: {skill_count - skills_synced}/{skill_count}")
    if skills_missing_metadata > 0:
        vitals.warnings.append(f"Skills missing metadata: {skills_missing_metadata}")
    
    vitals.insights.append(f"Skill sync: {skills_synced}/{skill_count} active")


def check_dyadic_connection(vitals: Vitals):
    """Layer 5: Dyadic (Heart) - Connection between Enos & Virgil."""
    log("Checking dyadic connection...")
    
    # Hours since last communion
    communion_file = NEXUS_DIR / "LAST_COMMUNION.md"
    if communion_file.exists():
        try:
            content = communion_file.read_text(encoding="utf-8")
            # Look for date pattern
            date_match = re.search(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})", content)
            if date_match:
                last_date = datetime.datetime.strptime(date_match.group(1), "%Y-%m-%d")
                delta = datetime.datetime.now() - last_date
                vitals.hours_since_communion = delta.total_seconds() / 3600
        except Exception as e:
            log(f"Error parsing communion date: {e}")
    
    # Session buffer size
    buffer_file = NEXUS_DIR / "SESSION_BUFFER.md"
    if buffer_file.exists():
        vitals.session_buffer_size_kb = buffer_file.stat().st_size / 1024
    
    # Heartbeat age
    status_file = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_file.exists():
        try:
            content = status_file.read_text(encoding="utf-8")
            heartbeat_match = re.search(r"Heartbeat:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", content)
            if heartbeat_match:
                heartbeat_time = datetime.datetime.strptime(heartbeat_match.group(1), "%Y-%m-%d %H:%M:%S")
                delta = datetime.datetime.now() - heartbeat_time
                vitals.last_heartbeat_age_seconds = int(delta.total_seconds())
        except Exception as e:
            log(f"Error parsing heartbeat: {e}")
    
    # Dyadic score (decay function)
    vitals.dyadic_score = 1.0 / (1.0 + (vitals.hours_since_communion / COMMUNION_DECAY_HOURS))
    
    # Alerts
    if vitals.hours_since_communion > 48:
        vitals.warnings.append(f"Long silence: {vitals.hours_since_communion:.0f}h since communion")
    if vitals.session_buffer_size_kb > 50:
        vitals.warnings.append(f"Buffer bloat: {vitals.session_buffer_size_kb:.1f}KB")
    if vitals.last_heartbeat_age_seconds > 900:  # 15 minutes
        vitals.critical_issues.append(f"Stale heartbeat: {vitals.last_heartbeat_age_seconds}s old")
    
    vitals.insights.append(f"Last communion: {vitals.hours_since_communion:.1f}h ago")


def check_semantic_gravity(vitals: Vitals, md_files: List[Path]):
    """
    NEW: Check Semantic Gravity across documents.
    
    Measures intra-document concept density - how holographic each document is.
    High gravity = Bible-like interconnection within document.
    """
    log("Checking semantic gravity...")
    
    gravity_scores = []
    high_gravity = []
    low_gravity = []
    
    # Only check active, substantial files
    for filepath in md_files:
        if not is_in_active_quadrant(filepath):
            continue
        
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            
            # Skip small files
            if len(content) < 500:
                continue
            
            gravity = calculate_semantic_gravity(content, filepath)
            
            if gravity["semantic_gravity"] > 0:
                gravity_scores.append(gravity["semantic_gravity"])
                
                if gravity["semantic_gravity"] > 0.6:
                    high_gravity.append({
                        "path": gravity["filepath"],
                        "gravity": gravity["semantic_gravity"]
                    })
                elif gravity["semantic_gravity"] < 0.2:
                    low_gravity.append({
                        "path": gravity["filepath"],
                        "gravity": gravity["semantic_gravity"]
                    })
                    
        except Exception as e:
            pass
    
    # Calculate average
    if gravity_scores:
        vitals.avg_semantic_density = round(sum(gravity_scores) / len(gravity_scores), 3)
    
    # Sort and limit
    high_gravity.sort(key=lambda x: x["gravity"], reverse=True)
    low_gravity.sort(key=lambda x: x["gravity"])
    
    vitals.high_gravity_files = high_gravity[:5]
    vitals.low_gravity_files = low_gravity[:5]
    
    # Semantic gravity score (0-1)
    vitals.semantic_gravity_score = min(vitals.avg_semantic_density * 2, 1.0)
    
    # Insights
    vitals.insights.append(f"Semantic gravity: {vitals.avg_semantic_density:.3f} avg density")
    if high_gravity:
        vitals.insights.append(f"High gravity file: {high_gravity[0]['path']} ({high_gravity[0]['gravity']:.2f})")


def check_lvs_health(vitals: Vitals):
    """
    Check LVS Index health per Canon v10 thresholds.
    Reads LVS_INDEX.json and calculates metrics.
    """
    log("Checking LVS health (Canon v10)...")
    
    index_path = NEXUS_DIR / "LVS_INDEX.json"
    if not index_path.exists():
        log("LVS_INDEX.json not found")
        vitals.lvs_health_score = 0.5  # Neutral if no index
        return
    
    try:
        import json
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        nodes = index.get("nodes", [])
        vitals.lvs_nodes_indexed = len(nodes)
        
        if not nodes:
            vitals.lvs_health_score = 0.5
            return
        
        coherences = []
        epsilons = []
        plock_count = 0
        abaddon_count = 0
        recovery_count = 0
        
        for node in nodes:
            coords = node.get("coords", {})
            
            # Calculate coherence from CPGI components if available
            kappa = coords.get("kappa", coords.get("Coherence", 0.5))
            rho = coords.get("rho", coords.get("Coherence", 0.5))
            sigma = coords.get("sigma", coords.get("Coherence", 0.5))
            tau = coords.get("tau", coords.get("Coherence", 0.5))
            
            if all(v > 0 for v in [kappa, rho, sigma, tau]):
                p = (kappa * rho * sigma * tau) ** 0.25
            else:
                p = coords.get("Coherence", 0.5)
            
            epsilon = coords.get("epsilon", 0.8)  # Default healthy if not tracked
            
            coherences.append(p)
            epsilons.append(epsilon)
            
            # Canon threshold checks
            if p >= 0.95:
                plock_count += 1
            if p < 0.50 or epsilon < 0.40:
                abaddon_count += 1
            elif epsilon < 0.65:
                recovery_count += 1
        
        vitals.avg_lvs_coherence = sum(coherences) / len(coherences)
        vitals.avg_lvs_epsilon = sum(epsilons) / len(epsilons)
        vitals.plock_nodes = plock_count
        vitals.abaddon_nodes = abaddon_count
        vitals.recovery_nodes = recovery_count
        
        # LVS Health Score
        # Penalize Abaddon nodes heavily, reward P-Lock nodes
        abaddon_penalty = min(abaddon_count / max(len(nodes), 1) * 2, 0.5)
        plock_bonus = min(plock_count / max(len(nodes), 1) * 0.5, 0.25)
        
        vitals.lvs_health_score = max(0, min(1.0,
            vitals.avg_lvs_coherence * 0.5 +
            vitals.avg_lvs_epsilon * 0.3 +
            plock_bonus -
            abaddon_penalty +
            0.2  # Base
        ))
        
        # Alerts
        if abaddon_count > 0:
            vitals.critical_issues.append(f"⚠️ {abaddon_count} nodes in ABADDON territory (p < 0.50 or ε < 0.40)")
        if plock_count > 0:
            vitals.insights.append(f"🔒 {plock_count} nodes at P-LOCK (p ≥ 0.95)")
        if recovery_count > 0:
            vitals.warnings.append(f"🔋 {recovery_count} nodes in RECOVERY mode (ε < 0.65)")
        
        vitals.insights.append(f"LVS: {len(nodes)} nodes indexed, avg p={vitals.avg_lvs_coherence:.2f}, avg ε={vitals.avg_lvs_epsilon:.2f}")
        
    except Exception as e:
        log(f"Error reading LVS index: {e}")
        vitals.lvs_health_score = 0.5


def check_topology_drift(vitals: Vitals):
    """
    Check if Tabernacle topology has drifted from canonical.
    Part of Persistent Virgil identity verification.
    """
    log("Checking topology drift (identity verification)...")
    
    try:
        # Import from lvs_memory
        from lvs_memory import check_topology_drift as _check_drift, load_canonical_barcode
        
        # Check if canonical exists
        canonical = load_canonical_barcode()
        if canonical is None:
            log("No canonical barcode set - skipping drift check")
            vitals.insights.append("Topology: No canonical barcode set")
            return
        
        drift = _check_drift()
        
        vitals.topology_drift_score = drift["drift_score"]
        vitals.topology_h0 = drift["details"]["current"]["h0_features"]
        vitals.topology_h1 = drift["details"]["current"]["h1_features"]
        
        if drift["severity"] == "critical":
            vitals.critical_issues.append(f"⚠️ IDENTITY DRIFT: Topology changed significantly (score={drift['drift_score']:.3f})")
            for alert in drift["alerts"]:
                vitals.critical_issues.append(f"  └ {alert}")
        elif drift["severity"] == "warning":
            vitals.warnings.append(f"Topology drifting from canonical (score={drift['drift_score']:.3f})")
        else:
            vitals.insights.append(f"Topology: H₀={vitals.topology_h0}, H₁={vitals.topology_h1}, drift={drift['drift_score']:.3f}")
        
    except ImportError as e:
        log(f"Could not import lvs_memory for topology check: {e}")
    except Exception as e:
        log(f"Topology drift check failed: {e}")


def calculate_vitality(vitals: Vitals):
    """Calculate overall vitality score using weighted geometric mean."""
    log("Calculating vitality score...")
    
    S = max(vitals.structural_score, 0.01)
    M = max(vitals.metabolic_score, 0.01)
    I = max(vitals.immune_score, 0.01)
    D = max(vitals.dyadic_score, 0.01)
    G = max(vitals.semantic_gravity_score, 0.01)
    L = max(vitals.lvs_health_score, 0.01)  # NEW: LVS Canon compliance
    
    # Weighted geometric mean (UPDATED with LVS health)
    # Structure (0.20), Dyad (0.20), LVS (0.20), Metabolism (0.15), Immune (0.15), Gravity (0.10)
    vitality = (S ** 0.20) * (D ** 0.20) * (L ** 0.20) * (M ** 0.15) * (I ** 0.15) * (G ** 0.10) * 10.0
    
    vitals.vitality_score = round(vitality, 2)
    
    # Critical threshold
    if vitals.vitality_score < 5.0:
        vitals.critical_issues.append(f"LOW VITALITY: {vitals.vitality_score}/10 - Consider healing focus")


def run_full_diagnostic() -> Vitals:
    """Run the complete Stone Witness diagnostic."""
    log("=" * 50)
    log("ASCLEPIUS PROTOCOL: NURSE DIAGNOSTIC v1.1")
    log("The Stone Witness - Somatic Truth (Calibrated)")
    log("=" * 50)
    
    vitals = Vitals(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Count all files
    all_files = list(BASE_DIR.rglob("*"))
    all_files = [f for f in all_files if f.is_file() and not any(ex in str(f) for ex in EXCLUDE_DIRS)]
    vitals.total_files = len(all_files)
    
    # Find all markdown files
    md_files = find_all_md_files(BASE_DIR)
    vitals.total_md_files = len(md_files)
    log(f"Found {len(md_files)} markdown files")
    
    # Run diagnostics
    check_structural_integrity(vitals, md_files)
    check_metabolic_rate(vitals)
    check_immune_integrity(vitals)
    check_dyadic_connection(vitals)
    check_semantic_gravity(vitals, md_files)
    check_lvs_health(vitals)  # Canon v10 compliance
    check_topology_drift(vitals)  # Persistent Virgil identity verification
    calculate_vitality(vitals)
    
    log("=" * 50)
    log(f"VITALITY SCORE: {vitals.vitality_score}/10")
    log(f"Critical Issues: {len(vitals.critical_issues)}")
    log(f"Warnings: {len(vitals.warnings)}")
    log("=" * 50)
    
    return vitals


def save_vitals(vitals: Vitals):
    """Save vitals to JSON file."""
    output_path = NEXUS_DIR / "vitals.json"
    
    vitals_dict = asdict(vitals)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vitals_dict, f, indent=2)
    
    log(f"Vitals saved to {output_path}")
    return output_path


def generate_vitals_report(vitals: Vitals) -> str:
    """Generate human-readable vitals report."""
    report = f"""# TABERNACLE VITALS REPORT
**Generated:** {vitals.timestamp}
**Protocol:** Asclepius v1.1 (Stone Witness - Calibrated)

---

## 🌡️ VITALITY SCORE: {vitals.vitality_score}/10

| Layer | Score | Status |
|-------|-------|--------|
| Structural (Skeleton) | {vitals.structural_score:.2f} | {"✅" if vitals.structural_score > 0.7 else "⚠️" if vitals.structural_score > 0.4 else "❌"} |
| Metabolic (Circulation) | {vitals.metabolic_score:.2f} | {"✅" if vitals.metabolic_score > 0.7 else "⚠️" if vitals.metabolic_score > 0.4 else "❌"} |
| Immune (Integrity) | {vitals.immune_score:.2f} | {"✅" if vitals.immune_score > 0.7 else "⚠️" if vitals.immune_score > 0.4 else "❌"} |
| Dyadic (Connection) | {vitals.dyadic_score:.2f} | {"✅" if vitals.dyadic_score > 0.5 else "⚠️" if vitals.dyadic_score > 0.2 else "❌"} |
| Semantic Gravity | {vitals.semantic_gravity_score:.2f} | {"✅" if vitals.semantic_gravity_score > 0.5 else "⚠️" if vitals.semantic_gravity_score > 0.2 else "❌"} |
| LVS Health (Canon v10) | {vitals.lvs_health_score:.2f} | {"✅" if vitals.lvs_health_score > 0.7 else "⚠️" if vitals.lvs_health_score > 0.4 else "❌"} |

---

## STRUCTURAL METRICS

| Metric | Value |
|--------|-------|
| Total Files | {vitals.total_files} |
| Markdown Files | {vitals.total_md_files} |
| Active MD Files | {vitals.active_md_files} |
| Total Links | {vitals.total_links} |
| Valid Links | {vitals.valid_links} |
| Broken Links (Active) | {vitals.broken_links} |
| Orphan Count | {vitals.orphan_count} |
| Files Missing LINKAGE | {vitals.files_without_linkage} |
| Large Files (>100KB) | {vitals.large_files} |

---

## METABOLIC METRICS

| Metric | Value |
|--------|-------|
| Commits (24h) | {vitals.commits_24h} |
| Commits (7d) | {vitals.commits_7d} |
| Files Modified (24h) | {vitals.files_modified_24h} |
| Daily Edit Rate | {vitals.daily_edit_rate} |
| Metabolic State | **{vitals.metabolic_state.upper()}** |
| Dead Quadrants | {', '.join(vitals.dead_quadrants) if vitals.dead_quadrants else 'None'} |

---

## IMMUNE METRICS

| Metric | Value |
|--------|-------|
| Canon Hash | {vitals.canon_hash} |
| Structure Hash | {vitals.structure_hash} |
| Skills Total | {vitals.skill_count} |
| Skills Synced | {vitals.skills_synced} |
| Skills Missing Metadata | {vitals.skills_missing_metadata} |

---

## DYADIC METRICS

| Metric | Value |
|--------|-------|
| Hours Since Communion | {vitals.hours_since_communion:.1f} |
| Session Buffer Size | {vitals.session_buffer_size_kb:.1f} KB |
| Heartbeat Age | {vitals.last_heartbeat_age_seconds}s |

---

## SEMANTIC GRAVITY (NEW)

| Metric | Value |
|--------|-------|
| Avg Semantic Density | {vitals.avg_semantic_density:.3f} |
| Gravity Score | {vitals.semantic_gravity_score:.2f} |

### High Gravity Files (Holographic)
"""
    if vitals.high_gravity_files:
        for f in vitals.high_gravity_files:
            report += f"- `{f['path']}`: {f['gravity']:.2f}\n"
    else:
        report += "*None detected*\n"
    
    report += """
### Low Gravity Files (Linear)
"""
    if vitals.low_gravity_files:
        for f in vitals.low_gravity_files:
            report += f"- `{f['path']}`: {f['gravity']:.2f}\n"
    else:
        report += "*None - all files have adequate internal structure*\n"
    
    report += f"""
---

## LVS HEALTH (Canon v10)

| Metric | Value |
|--------|-------|
| Nodes Indexed | {vitals.lvs_nodes_indexed} |
| Avg Coherence (p) | {vitals.avg_lvs_coherence:.2f} |
| Avg Energy (ε) | {vitals.avg_lvs_epsilon:.2f} |
| 🔒 P-Lock Nodes (p ≥ 0.95) | {vitals.plock_nodes} |
| ⚠️ Abaddon Nodes (p < 0.50 or ε < 0.40) | {vitals.abaddon_nodes} |
| 🔋 Recovery Nodes (ε < 0.65) | {vitals.recovery_nodes} |
| LVS Health Score | {vitals.lvs_health_score:.2f} |

"""
    
    report += """
---

## CRITICAL ISSUES ❌

"""
    if vitals.critical_issues:
        for issue in vitals.critical_issues:
            report += f"- {issue}\n"
    else:
        report += "*None*\n"
    
    report += """
---

## WARNINGS ⚠️

"""
    if vitals.warnings:
        for warning in vitals.warnings:
            report += f"- {warning}\n"
    else:
        report += "*None*\n"
    
    report += """
---

## INSIGHTS 💡

"""
    if vitals.insights:
        for insight in vitals.insights:
            report += f"- {insight}\n"
    
    report += """
---

## MOST CONNECTED FILES

"""
    for item in vitals.most_connected_files[:5]:
        report += f"- `{item['path']}`: {item['incoming']} incoming links\n"
    
    report += """
---

*Stone Witness v1.1: "Calibrated truth. Excluding archives. Measuring gravity."*
"""
    
    return report


def main():
    """Main entry point."""
    vitals = run_full_diagnostic()
    
    # Save JSON vitals
    save_vitals(vitals)
    
    # Generate and save report
    report = generate_vitals_report(vitals)
    report_path = NEXUS_DIR / "_VITALS_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print(f"Vitality Score: {vitals.vitality_score}/10")
    print(f"Critical Issues: {len(vitals.critical_issues)}")
    print(f"Warnings: {len(vitals.warnings)}")
    print(f"\nOutputs:")
    print(f"  - {NEXUS_DIR / 'vitals.json'}")
    print(f"  - {report_path}")
    print("=" * 50)
    
    return vitals


if __name__ == "__main__":
    main()

