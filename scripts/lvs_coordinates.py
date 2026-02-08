import gzip
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b|"
    r"(today|tomorrow|deadline|due)\b)",
    re.IGNORECASE,
)


def calculate_lvs_coordinates_v2(
    file_path: str, omega_path: str = "00_NEXUS/LOGOS.md"
) -> Dict[str, float]:
    """
    Calculate LVS coordinates for a file.

    Σ (Constraint) = mean(structure_entropy, schema_compliance, format_consistency)
      - structure_entropy = 1 - (compressed_size / raw_size)
      - schema_compliance = mean(yaml_frontmatter, required_headers_present)
      - format_consistency = structural_lines / total_lines

    Ī (Intent) = mean(verb_density, temporal_markers, action_ratio)
      - verb_density = min(1, imperative_verbs / (word_count / 100))
      - temporal_markers = min(1, temporal_hits / 3)
      - action_ratio = active / (active + passive)

    h (Height) = cosine_similarity(tfidf(file), tfidf(omega))

    R (Risk) = mean(inbound_link_score, explicit_marker_score, recency_score)
      - inbound_link_score = min(1, inbound_links / 10)
      - explicit_marker_score = min(1, markers / 2)
      - recency_score = max(0, 1 - age_days / 30)

    p (File coherence) = mean(Σ, Ī, h, R, link_validity)
      - link_validity = valid_wiki_links / total_wiki_links (1 if none)
    """
    target_path = Path(file_path)
    if not target_path.is_absolute():
        repo_root = _find_repo_root(Path.cwd())
        target_path = repo_root / target_path

    omega = Path(omega_path)
    if not omega.is_absolute():
        repo_root = _find_repo_root(target_path)
        omega = repo_root / omega

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")

    content = target_path.read_text(encoding="utf-8", errors="replace")
    omega_content = omega.read_text(encoding="utf-8", errors="replace") if omega.exists() else ""

    structure_entropy = _compression_entropy(content)
    schema_compliance = _schema_compliance(content, target_path.suffix.lower())
    format_consistency = _format_consistency(content)
    sigma = _mean([structure_entropy, schema_compliance, format_consistency])

    intent = _intent_score(content)
    height = _semantic_similarity(content, omega_content)
    risk = _risk_score(content, target_path)
    link_validity = _link_validity(content, target_path)

    file_p = _mean([sigma, intent, height, risk, link_validity])

    return {
        "Σ": _clamp(sigma),
        "Ī": _clamp(intent),
        "h": _clamp(height),
        "R": _clamp(risk),
        "p": _clamp(file_p),
    }


def _compression_entropy(content: str) -> float:
    raw_bytes = content.encode("utf-8", errors="replace")
    if not raw_bytes:
        return 0.0
    compressed = gzip.compress(raw_bytes)
    return _clamp(1.0 - (len(compressed) / max(1, len(raw_bytes))))


def _schema_compliance(content: str, suffix: str) -> float:
    frontmatter_present = 0.0
    if suffix in {".md", ".markdown"}:
        frontmatter_present = 1.0 if content.lstrip().startswith("---") else 0.0
    required_headers = ["LINKAGE"]
    headers_present = sum(1 for h in required_headers if h.lower() in content.lower())
    header_score = headers_present / len(required_headers) if required_headers else 1.0
    return _mean([frontmatter_present, header_score])


def _format_consistency(content: str) -> float:
    lines = content.splitlines()
    if not lines:
        return 0.0
    structural = 0
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            structural += 1
        elif in_code:
            continue
        elif stripped.startswith("#"):
            structural += 1
        elif stripped.startswith(("-", "*", "+")):
            structural += 1
        elif stripped.startswith(">"):
            structural += 1
    return _clamp(structural / len(lines))


def _intent_score(content: str) -> float:
    words = WORD_RE.findall(content.lower())
    word_count = len(words)
    if word_count == 0:
        return 0.0

    imperative = {
        "must",
        "shall",
        "ensure",
        "return",
        "raise",
        "set",
        "do",
        "make",
        "create",
        "avoid",
        "use",
        "check",
        "build",
        "verify",
    }
    passive = {
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "should",
        "could",
        "would",
        "may",
        "might",
    }

    active_count = sum(1 for w in words if w in imperative)
    passive_count = sum(1 for w in words if w in passive)

    verb_density = min(1.0, active_count / max(1.0, word_count / 100.0))
    temporal_markers = min(1.0, len(DATE_RE.findall(content)) / 3.0)
    action_ratio = (
        active_count / (active_count + passive_count)
        if (active_count + passive_count) > 0
        else 0.5
    )

    return _mean([verb_density, temporal_markers, action_ratio])


def _semantic_similarity(content: str, omega_content: str) -> float:
    tokens_a = _tokenize(content)
    tokens_b = _tokenize(omega_content)
    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    vocab = set(tf_a) | set(tf_b)
    idf = {}
    doc_count = 2
    for token in vocab:
        df = int(token in tf_a) + int(token in tf_b)
        idf[token] = math.log(1 + (doc_count / max(1, df)))

    vec_a = {t: tf_a[t] * idf[t] for t in vocab}
    vec_b = {t: tf_b[t] * idf[t] for t in vocab}
    return _cosine_similarity(vec_a, vec_b)


def _risk_score(content: str, file_path: Path) -> float:
    inbound_links = _count_inbound_links(file_path)
    inbound_score = min(1.0, inbound_links / 10.0)

    markers = sum(1 for m in ("[CRITICAL]", "[CANONICAL]") if m in content)
    marker_score = min(1.0, markers / 2.0)

    age_days = _age_days(file_path)
    recency_score = max(0.0, 1.0 - (age_days / 30.0))

    return _mean([inbound_score, marker_score, recency_score])


def _link_validity(content: str, file_path: Path) -> float:
    links = WIKI_LINK_RE.findall(content)
    if not links:
        return 1.0

    repo_root = _find_repo_root(file_path)
    valid = 0
    for link in links:
        target = link.split("|")[0].strip()
        if not target:
            continue
        if target.endswith(".md"):
            candidate = repo_root / target
        else:
            candidate = repo_root / f"{target}.md"
        if candidate.exists():
            valid += 1

    return _clamp(valid / max(1, len(links)))


def _count_inbound_links(file_path: Path) -> int:
    repo_root = _find_repo_root(file_path)
    target_name = file_path.stem
    target_rel = _relative_markdown_path(repo_root, file_path)
    count = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "venv", "__pycache__"}]
        for fname in files:
            if not fname.endswith(".md"):
                continue
            path = Path(root) / fname
            if path == file_path:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in WIKI_LINK_RE.findall(text):
                target = match.split("|")[0].strip()
                if not target:
                    continue
                if target == target_name or target == target_rel:
                    count += 1
    return count


def _relative_markdown_path(repo_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(repo_root)
    except ValueError:
        return file_path.name
    return rel.as_posix()


def _find_repo_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return start


def _age_days(file_path: Path) -> float:
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        return 9999.0
    now = time.time()
    age_seconds = max(0.0, now - mtime)
    return age_seconds / 86400.0


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text) if len(t) > 2]


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    dot = sum(vec_a[t] * vec_b.get(t, 0.0) for t in vec_a)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _clamp(dot / (norm_a * norm_b))


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
