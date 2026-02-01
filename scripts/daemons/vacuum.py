#!/usr/bin/env python3
"""
VACUUM — Desktop Quarantine Daemon
==================================
Moves files from ~/Desktop to ~/TABERNACLE/09_INBOX/ for enzyme processing.

Part of the Tabernacle metabolic pipeline:
  Desktop → [VACUUM] → 09_INBOX/ → [ENZYME] → 00_NEXUS/ or 05_CRYPT/

Safety:
  - NEVER deletes files, only moves
  - Only moves files older than age threshold (default: 1 hour)
  - Skips: .DS_Store, dotfiles, folders

Usage:
  python vacuum.py [--dry-run] [--age-minutes 60]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

# CONFIG
DESKTOP = Path.home() / "Desktop"
INBOX = Path.home() / "TABERNACLE" / "09_INBOX"

# Files to always skip
SKIP_PATTERNS = {
    ".DS_Store",
    ".localized",
    "Icon\r",  # macOS folder icons
}


def get_file_age_minutes(path: Path) -> float:
    """Get file age in minutes from modification time."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() / 60


def should_skip(path: Path) -> Tuple[bool, str]:
    """
    Determine if a file should be skipped.
    Returns (should_skip, reason).
    """
    name = path.name
    
    # Skip directories
    if path.is_dir():
        return True, "directory"
    
    # Skip dotfiles
    if name.startswith("."):
        return True, "dotfile"
    
    # Skip known system files
    if name in SKIP_PATTERNS:
        return True, "system file"
    
    return False, ""


def vacuum_desktop(age_minutes: int = 60, dry_run: bool = False) -> List[Tuple[Path, Path]]:
    """
    Move files from Desktop to INBOX.
    
    Args:
        age_minutes: Only move files older than this many minutes
        dry_run: If True, only report what would be moved
        
    Returns:
        List of (source, destination) tuples for moved files
    """
    moved = []
    
    # Guard: Desktop must exist
    if not DESKTOP.exists():
        print(f"[VACUUM] Desktop not found: {DESKTOP}")
        return moved
    
    # Create INBOX if needed
    if not dry_run:
        INBOX.mkdir(parents=True, exist_ok=True)
    
    # Scan desktop
    for item in DESKTOP.iterdir():
        # Skip check
        skip, reason = should_skip(item)
        if skip:
            continue
        
        # Age check
        age = get_file_age_minutes(item)
        if age < age_minutes:
            print(f"[VACUUM] SKIP (too young: {age:.0f}m < {age_minutes}m): {item.name}")
            continue
        
        # Destination path (handle collisions)
        dest = INBOX / item.name
        if dest.exists():
            # Add timestamp suffix to avoid overwrite
            stem = item.stem
            suffix = item.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = INBOX / f"{stem}_{timestamp}{suffix}"
        
        # Move or report
        if dry_run:
            print(f"[VACUUM] DRY-RUN would move: {item.name} → 09_INBOX/")
        else:
            try:
                shutil.move(str(item), str(dest))
                print(f"[VACUUM] MOVED: {item.name} → 09_INBOX/{dest.name}")
                moved.append((item, dest))
            except Exception as e:
                print(f"[VACUUM] ERROR moving {item.name}: {e}")
    
    return moved


def main():
    parser = argparse.ArgumentParser(
        description="Vacuum desktop files to TABERNACLE inbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be moved without moving"
    )
    parser.add_argument(
        "--age-minutes", "-a",
        type=int,
        default=60,
        help="Only move files older than N minutes (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[VACUUM] === Desktop Vacuum Run: {timestamp} ===")
    print(f"[VACUUM] Source: {DESKTOP}")
    print(f"[VACUUM] Destination: {INBOX}")
    print(f"[VACUUM] Age threshold: {args.age_minutes} minutes")
    if args.dry_run:
        print("[VACUUM] Mode: DRY-RUN (no changes)")
    print()
    
    # Run vacuum
    moved = vacuum_desktop(
        age_minutes=args.age_minutes,
        dry_run=args.dry_run
    )
    
    # Summary
    print()
    if args.dry_run:
        print(f"[VACUUM] Dry run complete. Would move {len(moved)} files.")
    else:
        print(f"[VACUUM] Complete. Moved {len(moved)} files.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
