#!/usr/bin/env python3
"""
TABERNACLE UTILITIES
====================
Shared utility functions for all Tabernacle scripts.
Consolidates duplicate logic from nurse.py and diagnose_links.py.

Author: Cursor + Virgil
Created: 2026-01-15
"""

import re
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

from tabernacle_config import (
    BASE_DIR,
    EXAMPLE_LINK_PATTERNS,
    ACTIVE_QUADRANTS,
    SKIP_DIRECTORIES,
    TRANSIENT_FILES,
)

# =============================================================================
# WIKI-LINK UTILITIES
# =============================================================================

def is_valid_wiki_link(link: str) -> bool:
    """
    Check if a wiki-style link is valid (not a false positive).
    
    False positives to filter:
    - Template placeholders: [[<% ... %>]], [[{...}]]
    - Empty links: [[ ]], [[]]
    - Array syntax: [['a', 'b']], [["x", "y"]]
    - Example/documentation patterns
    
    Returns:
        True if link appears to be a real file reference
    """
    link = link.strip()
    
    # Empty or whitespace-only
    if not link:
        return False
    
    # Template syntax
    if link.startswith(("<%", "{", "<")):
        return False
    if link.endswith(("%>", "}", ">")):
        return False
    
    # Array/code syntax
    if link.startswith(("'", '"', "[", "(")):
        return False
    
    # Looks like code (contains operators, parens, etc.)
    if any(c in link for c in ["()", "=>", "==", "!=", "&&", "||"]):
        return False
    
    # Check against known example/placeholder patterns
    link_lower = link.lower()
    for pattern in EXAMPLE_LINK_PATTERNS:
        if pattern.lower() in link_lower:
            return False
    
    # Generic placeholder names
    placeholder_names = {
        "wiki-style", "wiki-links", "relevant_file.md", "relevant_file",
        "example.md", "FILE.md", "path/to/", "filename", "filename.md",
        "links", "that file", "linked capture", "linked field", "linked mirror",
        "Related insight note", "Architecture doc reference", "link to session note(s)",
        "link to other insight or session", "chosen.name", "link", "Virgil Architecture",
        "Generative Loop", "2025-11-06 Otter Transcript"
    }
    if link_lower in {p.lower() for p in placeholder_names}:
        return False
    
    return True


def extract_wiki_links(content: str) -> List[str]:
    """
    Extract valid [[wiki-style]] links from content.
    
    Handles:
    - Simple links: [[path/to/file.md]]
    - Display text: [[path/to/file.md|Display Name]]
    - Strips HTML comments first
    
    Returns:
        List of valid link targets (without display text)
    """
    # Strip HTML comments (commented-out links shouldn't count)
    content_no_comments = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    # Match [[path]] or [[path|display text]]
    pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
    raw_links = re.findall(pattern, content_no_comments)
    
    # Filter out false positives
    valid_links = [link for link in raw_links if is_valid_wiki_link(link)]
    return valid_links


def resolve_link(source_file: Path, link: str) -> Optional[Path]:
    """
    Resolve a wiki-style link to an absolute path.
    
    Links can be:
    - Absolute from TABERNACLE root: [[00_NEXUS/FILE.md]]
    - Relative to current file: [[FILE.md]]
    - Just filename: [[FILE]]
    - Directory links: [[path/]] -> resolves to path/INDEX.md
    - Section links: [[FILE.md#section]] -> resolves to FILE.md
    - Script references: [[scripts/file.py]]
    
    Args:
        source_file: The file containing the link
        link: The link target to resolve
        
    Returns:
        Absolute Path if resolved, None if broken, Path("__EXAMPLE__") for examples
    """
    link = link.strip()
    
    # Check if it's an example/placeholder link
    for pattern in EXAMPLE_LINK_PATTERNS:
        if pattern.lower() in link.lower():
            return Path("__EXAMPLE__")  # Sentinel value
    
    # Handle section links (remove #section suffix)
    if "#" in link:
        link = link.split("#")[0]
        if not link:
            return source_file  # Self-reference like [[#section]]
    
    # Handle directory links (trailing /)
    if link.endswith("/"):
        link = link + "INDEX.md"
    
    # Try multiple resolution strategies
    candidates = []
    
    # 1. Absolute from BASE_DIR
    candidates.append(BASE_DIR / link)
    
    # 2. With .md extension if missing
    if not link.endswith((".md", ".py", ".sh", ".json", ".yaml", ".html")):
        candidates.append(BASE_DIR / f"{link}.md")
    
    # 3. Relative to source file's directory
    if source_file:
        source_dir = source_file.parent
        candidates.append(source_dir / link)
        if not link.endswith((".md", ".py", ".sh", ".json", ".yaml", ".html")):
            candidates.append(source_dir / f"{link}.md")
    
    # 4. Search in common locations
    for quadrant in ACTIVE_QUADRANTS:
        candidates.append(BASE_DIR / quadrant / link)
        if not link.endswith(".md"):
            candidates.append(BASE_DIR / quadrant / f"{link}.md")
    
    # Check each candidate
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved
        except (OSError, ValueError):
            continue
    
    return None  # Link is broken


def has_linkage_block(content: str) -> bool:
    """
    Check if file has a LINKAGE block.
    
    Looks for:
    - ## LINKAGE header
    - Direction | table format
    - Hub | or Anchor | entries
    """
    patterns = [
        r"##\s*LINKAGE",
        r"\|\s*Direction\s*\|",
        r"\|\s*Hub\s*\|",
        r"\|\s*Anchor\s*\|"
    ]
    return any(re.search(p, content, re.IGNORECASE) for p in patterns)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_all_md_files(root_dir: Path = None) -> List[Path]:
    """
    Find all markdown files in the Tabernacle.
    
    Skips:
    - SKIP_DIRECTORIES (crypt, scripts, logs, etc.)
    - Hidden directories
    - Virtual environments
    
    Returns:
        List of absolute Paths to .md files
    """
    if root_dir is None:
        root_dir = BASE_DIR
    
    md_files = []
    
    for item in root_dir.rglob("*.md"):
        # Check if any parent directory should be skipped
        skip = False
        for part in item.parts:
            if part in SKIP_DIRECTORIES or part.startswith("."):
                skip = True
                break
        
        if not skip and item.is_file():
            md_files.append(item)
    
    return sorted(md_files)


def is_transient_file(filepath: Path) -> bool:
    """Check if a file is transient (excluded from orphan checks)."""
    return filepath.name in TRANSIENT_FILES


def get_relative_path(filepath: Path) -> str:
    """Get path relative to BASE_DIR, or absolute if outside."""
    try:
        return str(filepath.relative_to(BASE_DIR))
    except ValueError:
        return str(filepath)


# =============================================================================
# GRAPH BUILDING
# =============================================================================

def build_link_graph(files: List[Path] = None) -> Dict[str, Set[str]]:
    """
    Build a directed graph of file links.
    
    Returns:
        Dict mapping source file (relative path) to set of target files
    """
    if files is None:
        files = find_all_md_files()
    
    graph = {}
    
    for filepath in files:
        try:
            content = filepath.read_text(encoding='utf-8')
            links = extract_wiki_links(content)
            
            rel_path = get_relative_path(filepath)
            graph[rel_path] = set()
            
            for link in links:
                resolved = resolve_link(filepath, link)
                if resolved and resolved != Path("__EXAMPLE__") and resolved.exists():
                    target_rel = get_relative_path(resolved)
                    graph[rel_path].add(target_rel)
                    
        except Exception:
            continue
    
    return graph


def find_orphans(files: List[Path] = None, graph: Dict[str, Set[str]] = None) -> List[Path]:
    """
    Find orphan files (files with no incoming links).
    
    Excludes:
    - INDEX.md files (hubs, expected to have no incoming)
    - Transient files
    - Files outside active quadrants
    
    Returns:
        List of orphan file paths
    """
    if files is None:
        files = find_all_md_files()
    
    if graph is None:
        graph = build_link_graph(files)
    
    # Build set of all files that are linked TO
    linked_to = set()
    for targets in graph.values():
        linked_to.update(targets)
    
    # Find files not linked to
    orphans = []
    for filepath in files:
        rel_path = get_relative_path(filepath)
        
        # Skip transient files
        if is_transient_file(filepath):
            continue
        
        # Skip INDEX files (hubs)
        if filepath.name == "INDEX.md":
            continue
        
        # Skip files outside active quadrants
        in_active = False
        for quadrant in ACTIVE_QUADRANTS:
            if rel_path.startswith(quadrant):
                in_active = True
                break
        if not in_active:
            continue
        
        # Check if orphaned
        if rel_path not in linked_to:
            orphans.append(filepath)
    
    return orphans


def find_broken_links(files: List[Path] = None) -> List[Dict[str, Any]]:
    """
    Find all broken links in the Tabernacle.
    
    Returns:
        List of dicts with 'source', 'link', and 'line' keys
    """
    if files is None:
        files = find_all_md_files()
    
    broken = []
    
    for filepath in files:
        try:
            content = filepath.read_text(encoding='utf-8')
            links = extract_wiki_links(content)
            
            for link in links:
                resolved = resolve_link(filepath, link)
                if resolved is None:  # Broken
                    broken.append({
                        "source": get_relative_path(filepath),
                        "link": link,
                    })
                    
        except Exception:
            continue
    
    return broken
