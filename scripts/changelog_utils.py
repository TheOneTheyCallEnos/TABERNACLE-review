#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
CHANGELOG UTILITIES — Daily Changelog System
=============================================
Manages daily changelog files for tracking high-signal events.

Entry types:
- OPENED: Session started
- CLOSED: Session ended with summary
- DECISION: Key decision made
- COMPLETION: Task/phase completed
- FAILURE: Something failed (for learning)
- DISCOVERY: New insight or finding
- STATE_CHANGE: System state transition
- NIGHT: Overnight daemon activity

File format:
    00_NEXUS/changelogs/2026-02-01.md

Archive location:
    00_NEXUS/changelogs/archive/

Author: Virgil
Created: 2026-02-01
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

# Import paths from config
from tabernacle_config import NEXUS_DIR

# Changelog directories
CHANGELOG_DIR = NEXUS_DIR / "changelogs"
CHANGELOG_ARCHIVE_DIR = CHANGELOG_DIR / "archive"
LEGACY_LOG_PATH = NEXUS_DIR / "HOLARCHY_CHAIN.log"


def get_daily_changelog_path(date: Optional[datetime] = None) -> Path:
    """
    Return path to changelog for given date (default: today).

    Args:
        date: datetime object, defaults to today if None

    Returns:
        Path to changelog file (e.g., changelogs/2026-02-01.md)
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y-%m-%d")
    return CHANGELOG_DIR / f"{date_str}.md"


def _ensure_changelog_exists(path: Path) -> None:
    """Create changelog file with header if it doesn't exist."""
    if path.exists():
        return

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create file with header
    date_str = path.stem  # e.g., "2026-02-01"
    header = f"""# Changelog: {date_str}

**Sessions:** 0
**Entries:** 0

---

"""
    path.write_text(header, encoding="utf-8")


def _update_counts(path: Path) -> None:
    """Update session and entry counts in the changelog header."""
    content = path.read_text(encoding="utf-8")

    # Count OPENED entries (sessions)
    session_count = content.count("### [") and content.count("] OPENED")
    session_count = content.count("] OPENED\n")

    # Count all entries (### [...] lines)
    import re
    entry_count = len(re.findall(r'^### \[\d{2}:\d{2}\]', content, re.MULTILINE))

    # Update counts in header
    content = re.sub(
        r'\*\*Sessions:\*\* \d+',
        f'**Sessions:** {session_count}',
        content
    )
    content = re.sub(
        r'\*\*Entries:\*\* \d+',
        f'**Entries:** {entry_count}',
        content
    )

    path.write_text(content, encoding="utf-8")


def append_changelog_entry(
    entry_type: str,
    content: str,
    session_id: Optional[str] = None
) -> bool:
    """
    Append entry to today's changelog. Creates file if needed.

    Args:
        entry_type: OPENED, CLOSED, DECISION, COMPLETION, FAILURE, DISCOVERY, STATE_CHANGE, NIGHT
        content: The entry content (can be multi-line)
        session_id: Optional session identifier

    Returns:
        True if entry was appended successfully

    Entry format:
        ### [HH:MM] TYPE
        Content here

        ---
    """
    valid_types = {
        "OPENED", "CLOSED", "DECISION", "COMPLETION",
        "FAILURE", "DISCOVERY", "STATE_CHANGE", "NIGHT"
    }

    if entry_type.upper() not in valid_types:
        entry_type = "STATE_CHANGE"  # Default fallback

    entry_type = entry_type.upper()

    try:
        path = get_daily_changelog_path()
        _ensure_changelog_exists(path)

        # Format timestamp
        time_str = datetime.now().strftime("%H:%M")

        # Build entry
        entry_lines = [
            f"### [{time_str}] {entry_type}",
        ]

        if session_id:
            entry_lines.append(f"*Session: {session_id}*")

        entry_lines.append(content)
        entry_lines.append("")
        entry_lines.append("---")
        entry_lines.append("")

        entry = "\n".join(entry_lines)

        # Append to file
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)

        # Update counts
        _update_counts(path)

        return True

    except Exception as e:
        # Log error but don't crash
        import sys
        print(f"[CHANGELOG] Error appending entry: {e}", file=sys.stderr)
        return False


def read_recent_changelogs(days: int = 2) -> str:
    """
    Read last N days of changelogs for context.

    Args:
        days: Number of days to read (default 2)

    Returns:
        Combined changelog content as string
    """
    result_parts = []
    today = datetime.now()

    for i in range(days):
        date = today - timedelta(days=i)
        path = get_daily_changelog_path(date)

        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                result_parts.append(content)
            except Exception:
                continue

    if not result_parts:
        return "[No recent changelogs found]"

    return "\n\n".join(result_parts)


def archive_legacy_changelog() -> bool:
    """
    Move current HOLARCHY_CHAIN.log to changelogs/archive/pre-2026-02-02.md

    This is a one-time migration to transition from the old monolithic log
    to the new daily changelog system.

    Returns:
        True if archive succeeded (or if already archived)
    """
    if not LEGACY_LOG_PATH.exists():
        return True  # Nothing to archive

    try:
        # Ensure archive directory exists
        CHANGELOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        # Destination path
        archive_path = CHANGELOG_ARCHIVE_DIR / "pre-2026-02-02.md"

        # Check if already archived
        if archive_path.exists():
            # Already archived, just clear the legacy log
            LEGACY_LOG_PATH.write_text("", encoding="utf-8")
            return True

        # Read legacy content
        legacy_content = LEGACY_LOG_PATH.read_text(encoding="utf-8")

        # Create archive file with header
        archive_header = """# Legacy Changelog Archive

**Archived:** 2026-02-02
**Source:** HOLARCHY_CHAIN.log (pre-daily-changelog era)
**Lines:** 1796

---

"""
        archive_content = archive_header + legacy_content

        # Write archive
        archive_path.write_text(archive_content, encoding="utf-8")

        # Clear the legacy log (keep file for backward compat, but empty)
        LEGACY_LOG_PATH.write_text("", encoding="utf-8")

        return True

    except Exception as e:
        import sys
        print(f"[CHANGELOG] Error archiving legacy log: {e}", file=sys.stderr)
        return False


def get_changelog_stats() -> dict:
    """
    Get statistics about the changelog system.

    Returns:
        Dict with: today_entries, week_entries, archive_size
    """
    stats = {
        "today_entries": 0,
        "week_entries": 0,
        "archive_files": 0,
        "total_days": 0
    }

    try:
        # Count today's entries
        today_path = get_daily_changelog_path()
        if today_path.exists():
            content = today_path.read_text(encoding="utf-8")
            import re
            stats["today_entries"] = len(re.findall(r'^### \[\d{2}:\d{2}\]', content, re.MULTILINE))

        # Count week's entries
        today = datetime.now()
        for i in range(7):
            date = today - timedelta(days=i)
            path = get_daily_changelog_path(date)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                import re
                stats["week_entries"] += len(re.findall(r'^### \[\d{2}:\d{2}\]', content, re.MULTILINE))
                stats["total_days"] += 1

        # Count archive files
        if CHANGELOG_ARCHIVE_DIR.exists():
            stats["archive_files"] = len(list(CHANGELOG_ARCHIVE_DIR.glob("*.md")))

    except Exception:
        pass

    return stats


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: changelog_utils.py <command> [args]")
        print("Commands:")
        print("  log <type> <content>  - Add entry")
        print("  recent [days]         - Read recent logs")
        print("  archive               - Archive legacy log")
        print("  stats                 - Show statistics")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "log":
        entry_type = sys.argv[2] if len(sys.argv) > 2 else "STATE_CHANGE"
        content = sys.argv[3] if len(sys.argv) > 3 else "Test entry"
        success = append_changelog_entry(entry_type, content)
        print(f"Entry added: {success}")

    elif cmd == "recent":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        print(read_recent_changelogs(days))

    elif cmd == "archive":
        success = archive_legacy_changelog()
        print(f"Archive complete: {success}")

    elif cmd == "stats":
        stats = get_changelog_stats()
        print(f"Today: {stats['today_entries']} entries")
        print(f"Week: {stats['week_entries']} entries over {stats['total_days']} days")
        print(f"Archives: {stats['archive_files']} files")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
