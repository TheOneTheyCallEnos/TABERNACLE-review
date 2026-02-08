#!/usr/bin/env python3
"""
PROJECT TABERNACLE: MAINTENANCE AUTOMATION (v1.0)
Proactive Systems for 10/10 Health

This script provides:
1. Auto-fix common link issues
2. Auto-add LINKAGE blocks to orphans
3. Pre-commit health check
4. Scheduled maintenance routines

"Don't be reactive. Be proactive."
"""

import os
import sys
import json
import re
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# --- CONFIGURATION ---
BASE_DIR = Path(os.path.expanduser("~/TABERNACLE"))
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LOG_DIR = BASE_DIR / "logs"

EXCLUDE_DIRS = {".git", "venv", "__pycache__", "node_modules", ".cursor"}
ARCHIVE_DIRS = {"05_CRYPT", "logs"}

# LINKAGE template
LINKAGE_TEMPLATE = """
---

## LINKAGE (The Circuit)

| Direction | Seed |
|-----------|------|
| Hub | [[00_NEXUS/CURRENT_STATE.md]] |
| Anchor | [[{anchor}]] |
"""

# Quadrant to anchor mapping
QUADRANT_ANCHORS = {
    "00_NEXUS": "00_NEXUS/TABERNACLE_MAP.md",
    "01_UL_INTENT": "01_UL_INTENT/INDEX.md",
    "02_UR_STRUCTURE": "02_UR_STRUCTURE/INDEX.md",
    "03_LL_RELATION": "03_LL_RELATION/INDEX.md",
    "04_LR_LAW": "04_LR_LAW/CANON/INDEX.md",
}


def log(message: str):
    """Log to console and file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [MAINT] {message}"
    print(entry)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "maintenance.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except:
        pass


def get_quadrant(filepath: Path) -> Optional[str]:
    """Get the quadrant a file belongs to."""
    rel_path = str(filepath.relative_to(BASE_DIR)) if filepath.is_relative_to(BASE_DIR) else str(filepath)
    for quad in QUADRANT_ANCHORS.keys():
        if rel_path.startswith(quad):
            return quad
    return None


def has_linkage_block(content: str) -> bool:
    """Check if file has a LINKAGE block."""
    patterns = [r"##\s*LINKAGE", r"\|\s*Direction\s*\|", r"\|\s*Hub\s*\|", r"\|\s*Anchor\s*\|"]
    return any(re.search(p, content, re.IGNORECASE) for p in patterns)


def add_linkage_to_file(filepath: Path, dry_run: bool = True) -> bool:
    """Add LINKAGE block to a file that's missing one."""
    try:
        content = filepath.read_text(encoding="utf-8")
        
        if has_linkage_block(content):
            return False  # Already has LINKAGE
        
        # Determine anchor based on quadrant
        quadrant = get_quadrant(filepath)
        anchor = QUADRANT_ANCHORS.get(quadrant, "00_NEXUS/CURRENT_STATE.md")
        
        # Generate LINKAGE block
        linkage = LINKAGE_TEMPLATE.format(anchor=anchor)
        
        # Append to file
        new_content = content.rstrip() + "\n" + linkage
        
        if dry_run:
            log(f"[DRY RUN] Would add LINKAGE to: {filepath.relative_to(BASE_DIR)}")
            return True
        else:
            filepath.write_text(new_content, encoding="utf-8")
            log(f"Added LINKAGE to: {filepath.relative_to(BASE_DIR)}")
            return True
            
    except Exception as e:
        log(f"Error adding LINKAGE to {filepath}: {e}")
        return False


def find_files_without_linkage() -> List[Path]:
    """Find all active .md files without LINKAGE blocks."""
    missing = []
    
    # Skip these files (auto-generated, status files, etc.)
    skip_files = {
        "_VITALS_REPORT.md", "_LINK_DIAGNOSIS.md", "_LINK_FIXES.md",
        "_GRAPH_ATLAS.md", "_SEMANTIC_DIAGNOSIS.md", "_ASCLEPIUS_EXAMEN.md",
        "DAEMON_REFLECTION.md", "SYSTEM_STATUS.md", "LAST_COMMUNION.md",
        "SESSION_BUFFER.md", "OUTBOX.md", "vitals.json", "CHANGELOG.md"
    }
    
    for dirpath, dirnames, filenames in os.walk(BASE_DIR):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and d not in ARCHIVE_DIRS]
        
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            if filename in skip_files:
                continue
                
            filepath = Path(dirpath) / filename
            
            # Skip archive
            rel_path = str(filepath.relative_to(BASE_DIR))
            if any(archive in rel_path for archive in ARCHIVE_DIRS):
                continue
            
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                if not has_linkage_block(content):
                    missing.append(filepath)
            except:
                pass
    
    return missing


def auto_fix_linkage(dry_run: bool = True) -> int:
    """Add LINKAGE blocks to all files missing them."""
    log("=" * 50)
    log("AUTO-FIX: Adding LINKAGE blocks")
    log("=" * 50)
    
    missing = find_files_without_linkage()
    log(f"Found {len(missing)} files without LINKAGE")
    
    fixed = 0
    for filepath in missing:
        if add_linkage_to_file(filepath, dry_run=dry_run):
            fixed += 1
    
    log(f"{'Would fix' if dry_run else 'Fixed'}: {fixed} files")
    return fixed


def create_incoming_link(target_path: str, source_file: Path) -> bool:
    """
    Create a link TO a target file FROM a source file.
    Used to fix orphans by adding them to an index.
    """
    # This would add a reference to the target in the source's LINKAGE
    # For now, we'll just log what needs to be done
    log(f"Orphan fix needed: Link TO {target_path} FROM {source_file}")
    return True


def check_pre_commit() -> Tuple[bool, str]:
    """
    Pre-commit health check.
    Returns (pass, message).
    """
    log("Running pre-commit health check...")
    
    # Load latest vitals
    vitals_path = NEXUS_DIR / "vitals.json"
    if not vitals_path.exists():
        return True, "No vitals.json - run nurse.py first"
    
    with open(vitals_path) as f:
        vitals = json.load(f)
    
    score = vitals.get("vitality_score", 0)
    broken = vitals.get("broken_links", 0)
    orphans = vitals.get("orphan_count", 0)
    
    issues = []
    
    # Check thresholds
    if score < 7.0:
        issues.append(f"Vitality too low: {score}/10 (need >= 7.0)")
    
    if broken > 20:
        issues.append(f"Too many broken links: {broken} (need <= 20)")
    
    if orphans > 15:
        issues.append(f"Too many orphans: {orphans} (need <= 15)")
    
    if issues:
        return False, "Pre-commit FAILED:\n" + "\n".join(f"  - {i}" for i in issues)
    
    return True, f"Pre-commit PASSED (Vitality: {score}/10)"


def run_daily_maintenance():
    """
    Daily maintenance routine.
    Called by watchman or cron.
    """
    log("=" * 50)
    log("DAILY MAINTENANCE ROUTINE")
    log("=" * 50)
    
    # 1. Run nurse diagnostic
    log("Step 1: Running nurse diagnostic...")
    import subprocess
    result = subprocess.run(
        ["python3", str(BASE_DIR / "scripts" / "nurse.py")],
        capture_output=True, text=True, timeout=120
    )
    
    # 2. Check for critical issues
    vitals_path = NEXUS_DIR / "vitals.json"
    if vitals_path.exists():
        with open(vitals_path) as f:
            vitals = json.load(f)
        
        score = vitals.get("vitality_score", 0)
        log(f"Current Vitality: {score}/10")
        
        if score < 7.0:
            log("⚠️ LOW VITALITY - Running auto-fixes...")
            auto_fix_linkage(dry_run=False)
        elif score < 9.0:
            log("📊 Moderate health - Checking for easy fixes...")
            auto_fix_linkage(dry_run=True)  # Just report
        else:
            log("✅ Excellent health!")
    
    # 3. Generate maintenance report
    report = generate_maintenance_report()
    report_path = NEXUS_DIR / "_MAINTENANCE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    log(f"Report saved to {report_path}")
    
    log("Daily maintenance complete")


def generate_maintenance_report() -> str:
    """Generate a maintenance status report."""
    vitals_path = NEXUS_DIR / "vitals.json"
    vitals = {}
    if vitals_path.exists():
        with open(vitals_path) as f:
            vitals = json.load(f)
    
    missing_linkage = find_files_without_linkage()
    
    report = f"""# TABERNACLE MAINTENANCE REPORT
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CURRENT HEALTH

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Vitality | {vitals.get('vitality_score', 'N/A')}/10 | 10/10 | {'✅' if vitals.get('vitality_score', 0) >= 9 else '⚠️'} |
| Broken Links | {vitals.get('broken_links', 'N/A')} | 0 | {'✅' if vitals.get('broken_links', 999) <= 5 else '⚠️'} |
| Orphans | {vitals.get('orphan_count', 'N/A')} | 0 | {'✅' if vitals.get('orphan_count', 999) <= 5 else '⚠️'} |
| Missing LINKAGE | {len(missing_linkage)} | 0 | {'✅' if len(missing_linkage) == 0 else '⚠️'} |

---

## FILES NEEDING LINKAGE

"""
    if missing_linkage:
        for f in missing_linkage[:20]:
            report += f"- `{f.relative_to(BASE_DIR)}`\n"
        if len(missing_linkage) > 20:
            report += f"\n*... and {len(missing_linkage) - 20} more*\n"
    else:
        report += "*None - all files have LINKAGE!*\n"
    
    report += f"""
---

## PROACTIVE SYSTEMS STATUS

| System | Status | Last Run |
|--------|--------|----------|
| Nurse Diagnostic | ✅ Active | {vitals.get('timestamp', 'Unknown')} |
| Auto-LINKAGE | ✅ Ready | On demand |
| Pre-commit Hook | 📋 Available | Manual |
| Scheduled Examen | 📋 Configure cron | Not scheduled |

---

## RECOMMENDED ACTIONS

"""
    
    if vitals.get('vitality_score', 0) < 9:
        report += "1. Run `python3 scripts/tabernacle_maintenance.py fix` to auto-add LINKAGE\n"
    if vitals.get('broken_links', 0) > 0:
        report += "2. Review broken links in `_LINK_DIAGNOSIS.md`\n"
    if vitals.get('orphan_count', 0) > 0:
        report += "3. Add incoming links to orphan files\n"
    
    if vitals.get('vitality_score', 0) >= 9.5:
        report += "*System is healthy! Maintain current practices.*\n"
    
    report += """
---

## AUTOMATION SETUP

### Cron Job (Daily Examen)
```bash
# Add to crontab -e:
0 4 * * * cd ~/TABERNACLE && python3 scripts/tabernacle_maintenance.py daily >> logs/cron.log 2>&1
```

### Pre-commit Hook
```bash
# Create .git/hooks/pre-commit:
#!/bin/bash
python3 ~/TABERNACLE/scripts/tabernacle_maintenance.py precommit
```

---

*"Proactive systems keep vitality high. Reactive fixes drain energy."*
"""
    
    return report


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 tabernacle_maintenance.py [command]")
        print("")
        print("Commands:")
        print("  check     - Show files needing LINKAGE (dry run)")
        print("  fix       - Auto-add LINKAGE blocks")
        print("  precommit - Run pre-commit health check")
        print("  daily     - Run daily maintenance routine")
        print("  report    - Generate maintenance report")
        return
    
    command = sys.argv[1].lower()
    
    if command == "check":
        missing = find_files_without_linkage()
        print(f"\nFiles without LINKAGE: {len(missing)}")
        for f in missing:
            print(f"  - {f.relative_to(BASE_DIR)}")
        print(f"\nRun 'python3 tabernacle_maintenance.py fix' to add LINKAGE blocks.")
        
    elif command == "fix":
        auto_fix_linkage(dry_run=False)
        
    elif command == "precommit":
        passed, message = check_pre_commit()
        print(message)
        sys.exit(0 if passed else 1)
        
    elif command == "daily":
        run_daily_maintenance()
        
    elif command == "report":
        report = generate_maintenance_report()
        print(report)
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
