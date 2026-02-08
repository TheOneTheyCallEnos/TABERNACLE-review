#!/usr/bin/env python3
"""
BROKEN LINK SCANNER AND FIXER
=============================
Scans TABERNACLE for broken [[wiki-links]] and attempts to fix them.

Usage:
    python fix_broken_links.py scan      # Dry run - show broken links
    python fix_broken_links.py fix       # Apply fixes

Author: Cursor Opus
Date: 2026-02-04
"""

import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration
TABERNACLE = Path(os.path.expanduser("~/TABERNACLE"))
IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', 'venv312', '.obsidian', 'logs', '.review_mirror'}

def get_all_md_files() -> Dict[str, Path]:
    """Get all markdown files indexed by filename (without extension)."""
    files = {}
    for md in TABERNACLE.rglob("*.md"):
        # Skip ignored directories
        if any(ignored in md.parts for ignored in IGNORE_DIRS):
            continue
        
        # Index by filename (case-insensitive)
        name = md.stem.lower()
        if name not in files:
            files[name] = md
    
    return files


def extract_wiki_links(content: str) -> List[str]:
    """Extract all [[wiki-links]] from content."""
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    return re.findall(pattern, content)


def scan_for_broken_links() -> Dict[Path, List[Tuple[str, str]]]:
    """
    Scan all markdown files for broken wiki-links.
    
    Returns: {source_file: [(broken_link, suggested_fix), ...]}
    """
    all_files = get_all_md_files()
    all_names = set(all_files.keys())
    
    # Also index by full relative path
    all_paths = set()
    for name, path in all_files.items():
        rel = path.relative_to(TABERNACLE)
        all_paths.add(str(rel))
        all_paths.add(str(rel.with_suffix('')))
    
    broken = defaultdict(list)
    
    for name, path in all_files.items():
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        
        links = extract_wiki_links(content)
        
        for link in links:
            link_lower = link.lower()
            link_stem = Path(link).stem.lower()
            
            # Check if link resolves
            resolved = (
                link_lower in all_names or
                link_stem in all_names or
                link in all_paths or
                link.lower() in [str(p).lower() for p in all_paths]
            )
            
            if not resolved:
                # Try to find a close match
                suggestions = difflib.get_close_matches(
                    link_stem,
                    list(all_names),
                    n=1,
                    cutoff=0.6
                )
                
                if suggestions:
                    # Get the actual filename with proper case
                    suggested = all_files[suggestions[0]].stem
                    broken[path].append((link, suggested))
                else:
                    broken[path].append((link, None))
    
    return broken


def apply_fixes(broken_links: Dict[Path, List[Tuple[str, str]]], dry_run: bool = True) -> int:
    """Apply fixes to broken links."""
    fixed_count = 0
    
    for source_file, links in broken_links.items():
        if not links:
            continue
        
        try:
            content = source_file.read_text(encoding='utf-8')
            new_content = content
            
            for broken_link, suggestion in links:
                if suggestion:
                    # Replace [[broken]] with [[suggestion]]
                    old_pattern = f'[[{broken_link}]]'
                    new_pattern = f'[[{suggestion}]]'
                    
                    if old_pattern in new_content:
                        new_content = new_content.replace(old_pattern, new_pattern)
                        print(f"  {source_file.name}: [[{broken_link}]] → [[{suggestion}]]")
                        fixed_count += 1
            
            if not dry_run and new_content != content:
                source_file.write_text(new_content, encoding='utf-8')
        
        except Exception as e:
            print(f"  ERROR processing {source_file}: {e}")
    
    return fixed_count


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_broken_links.py [scan|fix]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    print("=" * 60)
    print("BROKEN LINK SCANNER")
    print("=" * 60)
    print(f"\nScanning {TABERNACLE}...\n")
    
    broken = scan_for_broken_links()
    
    # Count
    total_broken = sum(len(links) for links in broken.values())
    fixable = sum(1 for links in broken.values() for _, sug in links if sug)
    unfixable = total_broken - fixable
    
    print(f"Found {total_broken} broken links in {len(broken)} files")
    print(f"  - Fixable (close match found): {fixable}")
    print(f"  - Unfixable (no match): {unfixable}")
    print()
    
    if command == "scan":
        print("BROKEN LINKS (DRY RUN):")
        print("-" * 60)
        for source, links in sorted(broken.items()):
            rel_path = source.relative_to(TABERNACLE)
            for broken_link, suggestion in links:
                if suggestion:
                    print(f"  {rel_path}: [[{broken_link}]] → [[{suggestion}]]")
                else:
                    print(f"  {rel_path}: [[{broken_link}]] → [NO MATCH]")
        print()
        print("Run with 'fix' to apply changes.")
    
    elif command == "fix":
        print("APPLYING FIXES:")
        print("-" * 60)
        fixed = apply_fixes(broken, dry_run=False)
        print()
        print(f"Fixed {fixed} links.")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
