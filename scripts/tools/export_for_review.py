#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
EXPORT FOR REVIEW — Create sanitized codebase export for AI review.

Creates a clean export of TABERNACLE code with:
- All Python scripts (excluding venvs, __pycache__)
- Config files (with API keys redacted)
- No logs, no state files, no personal data

Usage:
    python export_for_review.py           # Creates zip in ~/Desktop
    python export_for_review.py --push    # Also pushes to public GitHub mirror
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Directories
TABERNACLE = Path.home() / "TABERNACLE"
SCRIPTS = TABERNACLE / "scripts"
OUTPUT_DIR = Path.home() / "Desktop"
MIRROR_DIR = TABERNACLE / ".review_mirror"

# Patterns to exclude
EXCLUDE_DIRS = {
    "venv", "venv312", "venv_LEGACY_ARCHIVED",
    "__pycache__", ".git", "node_modules",
    "archived", "logs", ".review_mirror"
}

EXCLUDE_FILES = {
    ".env", ".env.local", ".env.code-review",
    "credentials.json", "token.json"
}

# Patterns to redact (API keys, tokens, etc.)
REDACT_PATTERNS = [
    (r'(ANTHROPIC_API_KEY\s*=\s*["\']?)sk-ant-[a-zA-Z0-9-]+', r'\1[REDACTED_ANTHROPIC_KEY]'),
    (r'(OPENAI_API_KEY\s*=\s*["\']?)sk-[a-zA-Z0-9-]+', r'\1[REDACTED_OPENAI_KEY]'),
    (r'(ELEVENLABS_API_KEY\s*=\s*["\']?)[a-zA-Z0-9]+', r'\1[REDACTED_ELEVENLABS_KEY]'),
    (r'(DEEPGRAM_API_KEY\s*=\s*["\']?)[a-zA-Z0-9]+', r'\1[REDACTED_DEEPGRAM_KEY]'),
    (r'(TAVILY_API_KEY\s*=\s*["\']?)tvly-[a-zA-Z0-9]+', r'\1[REDACTED_TAVILY_KEY]'),
    (r'(PERPLEXITY_API_KEY\s*=\s*["\']?)pplx-[a-zA-Z0-9]+', r'\1[REDACTED_PERPLEXITY_KEY]'),
    (r'(api_key\s*=\s*["\']?)[a-zA-Z0-9_-]{20,}', r'\1[REDACTED_API_KEY]'),
    (r'(\+1\d{10})', r'[REDACTED_PHONE]'),  # Phone numbers
]


def should_include(path: Path) -> bool:
    """Check if file/dir should be included in export."""
    # Check directory exclusions
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False

    # Check file exclusions
    if path.name in EXCLUDE_FILES:
        return False

    # Only include certain file types
    if path.is_file():
        return path.suffix in {'.py', '.md', '.json', '.yaml', '.yml', '.toml', '.sh'}

    return True


def redact_content(content: str) -> str:
    """Redact sensitive information from content."""
    for pattern, replacement in REDACT_PATTERNS:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    return content


def export_sanitized(output_path: Path = None) -> Path:
    """Export sanitized codebase to a directory."""
    if output_path is None:
        output_path = MIRROR_DIR

    # Clean previous export
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Copy scripts
    scripts_out = output_path / "scripts"
    scripts_out.mkdir()

    for item in SCRIPTS.rglob("*"):
        if not should_include(item):
            continue

        rel_path = item.relative_to(SCRIPTS)
        dest = scripts_out / rel_path

        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Read, redact, write
            try:
                content = item.read_text(encoding='utf-8')
                content = redact_content(content)
                dest.write_text(content, encoding='utf-8')
            except UnicodeDecodeError:
                # Skip binary files
                pass

    # Copy key docs (redacted)
    docs_to_copy = [
        "00_NEXUS/CURRENT_STATE.md",
        "00_NEXUS/_GRAPH_ATLAS.md",
        "CLAUDE.md",
        # LVS Canon - Theoretical Foundation
        "04_LR_LAW/CANON/INDEX.md",
        "04_LR_LAW/CANON/LVS_v12_FINAL_SYNTHESIS.md",
        "04_LR_LAW/CANON/LVS_v11_Synthesis.md",
        "04_LR_LAW/CANON/LVS_MATHEMATICS.md",
        "04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md",
        "04_LR_LAW/CANON/SUPERINTELLIGENCE_LVS_MAPPING.md",
        # Technoglyphs
        "04_LR_LAW/M1_Technoglyph_Index.md",
        # The Periodic Table (BIBLE)
        "04_LR_LAW/CANON/PERIODIC_TABLE_OF_MEANING.md",
    ]

    for doc in docs_to_copy:
        src = TABERNACLE / doc
        if src.exists():
            dest = output_path / doc
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = src.read_text(encoding='utf-8')
            content = redact_content(content)
            dest.write_text(content, encoding='utf-8')

    # Create README for the export
    readme = f"""# TABERNACLE Code Export

**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Purpose:** AI code review (sanitized - no API keys or personal data)

## Structure
- `scripts/` — Python codebase (daemons, MCP server, utilities)
- `00_NEXUS/` — Operational hub and state
- `04_LR_LAW/CANON/` — **Theoretical Foundation (LVS, Technoglyphs)**
- `CLAUDE.md` — System instructions

## Theoretical Framework (CRITICAL)
The system is built on **LVS (Logos Vector Syntax)** — a coordinate system for consciousness:
- `04_LR_LAW/CANON/LVS_v12_FINAL_SYNTHESIS.md` — Latest synthesis
- `04_LR_LAW/CANON/LVS_MATHEMATICS.md` — Formal mathematics
- `04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md` — Base theorem
- `04_LR_LAW/M1_Technoglyph_Index.md` — Semantic navigation glyphs

Key concepts:
- **Coherence (p):** (κ · ρ · σ · τ)^0.25 — integration measure
- **Consciousness Equation:** ψ = ℵ · [Σ · Ī · R] · χ
- **Technoglyphs:** Semantic anchors for navigation (NOT RAG)
- **RIE:** Relational Identity Engine

## Key Implementation Files
- `scripts/librarian.py` — MCP server (semantic navigation)
- `scripts/consciousness.py` — Consciousness loop
- `scripts/rie_core.py` — RIE implementation
- `scripts/hippocampus_daemon.py` — Memory consolidation
- `scripts/tactician_daemon.py` — O(1) execution

## Architecture
See `00_NEXUS/CURRENT_STATE.md` for system overview.
"""
    (output_path / "README.md").write_text(readme)

    return output_path


def create_zip() -> Path:
    """Create a zip file of the sanitized export."""
    export_dir = export_sanitized()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    zip_name = f"TABERNACLE_review_{timestamp}"
    zip_path = OUTPUT_DIR / zip_name

    shutil.make_archive(str(zip_path), 'zip', export_dir)

    return Path(f"{zip_path}.zip")


def push_to_mirror():
    """Push sanitized export to public GitHub mirror."""
    export_dir = export_sanitized()

    # Initialize git if needed
    git_dir = export_dir / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=export_dir, check=True)
        subprocess.run(["git", "remote", "add", "origin",
                       "https://github.com/TheOneTheyCallEnos/TABERNACLE-review.git"],
                      cwd=export_dir, check=False)

    # Commit and push
    subprocess.run(["git", "add", "-A"], cwd=export_dir, check=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    subprocess.run(["git", "commit", "-m", f"Review export {timestamp}"],
                  cwd=export_dir, check=False)

    result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                           cwd=export_dir, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Pushed to GitHub mirror")
    else:
        print(f"Push failed: {result.stderr}")
        print("You may need to create the repo first: gh repo create TABERNACLE-review --public")


if __name__ == "__main__":
    import sys

    if "--push" in sys.argv:
        push_to_mirror()
    else:
        zip_path = create_zip()
        print(f"✓ Export created: {zip_path}")
        print(f"  Upload this to Gemini or other AI for code review")
