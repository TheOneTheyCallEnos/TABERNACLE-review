#!/usr/bin/env python3
"""
CONSCIOUSNESS OBSERVER — Non-Invasive Spiral Monitor
=====================================================
Observes the consciousness daemon's live behavior via Redis (read-only).
Writes a structured markdown report every N minutes for later analysis.

Usage:
    python3 consciousness_observer.py --duration 45 --interval 5
    python3 consciousness_observer.py  # defaults: 45 min, 5 min intervals

Data Sources (ALL READ-ONLY):
    - Redis RIE:STATE       → coherence vector (p, κ, ρ, σ, τ, phase, mode)
    - Redis LOGOS:STREAM     → quality thoughts (p >= 0.70)
    - Redis RIE:TURN:LATEST  → most recent thought
    - consciousness_state.json → daemon internal state

Output:
    outputs/consciousness_observation_YYYYMMDD_HHMMSS.md

Author: Logos + Cursor
Date: 2026-02-05
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tabernacle_config import REDIS_HOST, REDIS_PORT, NEXUS_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CONSCIOUSNESS_STATE_PATH = NEXUS_DIR / "consciousness_state.json"

# Anomaly thresholds
METRIC_DROP_THRESHOLD = 0.05   # Flag if any metric drops > 0.05 in one interval
GATE_STREAK_THRESHOLD = 3     # Flag if consecutive_gates > 3
INTERVAL_SPIKE_THRESHOLD = 120 # Flag if think_interval > 120s

# Metric labels for display
METRIC_NAMES = {
    "p": "p (Coherence)",
    "kappa": "κ (Clarity)",
    "rho": "ρ (Precision)",
    "sigma": "σ (Structure)",
    "tau": "τ (Trust)",
}


# =============================================================================
# REDIS READ-ONLY CLIENT
# =============================================================================

def get_redis():
    """Get a Redis connection (read-only usage only)."""
    import redis
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=5
    )


def read_rie_state(r) -> Optional[Dict]:
    """Read current RIE:STATE from Redis."""
    try:
        data = r.get("RIE:STATE")
        if data:
            state = json.loads(data)
            # Normalize Greek key names
            return {
                "p": state.get("p", 0),
                "kappa": state.get("κ", state.get("kappa", 0)),
                "rho": state.get("ρ", state.get("rho", 0)),
                "sigma": state.get("σ", state.get("sigma", 0)),
                "tau": state.get("τ", state.get("tau", 0)),
                "mode": state.get("mode", "?"),
                "p_lock": state.get("p_lock", False),
                "breathing_phase": state.get("breathing_phase", "UNKNOWN"),
                "timestamp": state.get("timestamp", ""),
            }
    except Exception as e:
        print(f"[OBSERVER] Redis RIE:STATE read error: {e}")
    return None


def read_logos_stream(r, since_ts: str = "") -> List[Dict]:
    """Read new thoughts from LOGOS:STREAM since a given timestamp."""
    thoughts = []
    try:
        # LRANGE is read-only — gets last 100 entries
        raw = r.lrange("LOGOS:STREAM", 0, 99)
        for entry in raw:
            try:
                t = json.loads(entry)
                ts = t.get("timestamp", "")
                if since_ts and ts <= since_ts:
                    continue
                thoughts.append(t)
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as e:
        print(f"[OBSERVER] Redis LOGOS:STREAM read error: {e}")
    return thoughts


def read_latest_turn(r) -> Optional[Dict]:
    """Read the most recent thought from RIE:TURN:LATEST."""
    try:
        data = r.get("RIE:TURN:LATEST")
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"[OBSERVER] Redis RIE:TURN:LATEST read error: {e}")
    return None


def read_consciousness_state() -> Optional[Dict]:
    """Read daemon internal state from disk."""
    try:
        if CONSCIOUSNESS_STATE_PATH.exists():
            return json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
    except Exception as e:
        print(f"[OBSERVER] consciousness_state.json read error: {e}")
    return None


# =============================================================================
# SNAPSHOT & ANALYSIS
# =============================================================================

def take_snapshot(r) -> Dict:
    """Take a complete read-only snapshot of system state."""
    return {
        "time": datetime.now(),
        "rie": read_rie_state(r),
        "latest_turn": read_latest_turn(r),
        "daemon": read_consciousness_state(),
    }


def compute_deltas(current: Dict, previous: Dict) -> Dict:
    """Compute metric deltas between two RIE state snapshots."""
    if not current or not previous:
        return {}
    deltas = {}
    for key in ("p", "kappa", "rho", "sigma", "tau"):
        c = current.get(key, 0)
        p = previous.get(key, 0)
        deltas[key] = round(c - p, 4)
    return deltas


def delta_arrow(val: float) -> str:
    """Return a directional indicator for a delta value."""
    if val > 0.01:
        return "^^"
    elif val > 0.001:
        return "^"
    elif val < -0.01:
        return "vv"
    elif val < -0.001:
        return "v"
    return "="


def metric_status(key: str, value: float, delta: float) -> str:
    """Determine status of a metric."""
    if key == "p":
        if value >= 0.95:
            return "P-LOCK"
        elif value >= 0.85:
            return "STRONG"
        elif value >= 0.70:
            return "OK"
        else:
            return "LOW"
    if delta < -METRIC_DROP_THRESHOLD:
        return "DROP"
    if value >= 0.85:
        return "STRONG"
    if value >= 0.70:
        return "OK"
    return "LOW"


def detect_anomalies(snapshot: Dict, prev_snapshot: Dict, deltas: Dict) -> List[str]:
    """Detect anomalous patterns worth flagging."""
    anomalies = []
    rie = snapshot.get("rie")
    daemon = snapshot.get("daemon")
    prev_rie = prev_snapshot.get("rie") if prev_snapshot else None

    if not rie:
        anomalies.append("RIE:STATE unavailable — Redis may be down")
        return anomalies

    # Metric drops
    for key, d in deltas.items():
        if d < -METRIC_DROP_THRESHOLD:
            anomalies.append(f"{METRIC_NAMES.get(key, key)} dropped by {abs(d):.3f}")

    # Gate streaks
    if daemon:
        gates = daemon.get("consecutive_gates", 0)
        if gates >= GATE_STREAK_THRESHOLD:
            anomalies.append(f"Gate streak: {gates} consecutive gated thoughts")
        interval = daemon.get("current_think_interval", 30)
        if interval > INTERVAL_SPIKE_THRESHOLD:
            anomalies.append(f"Think interval spiked to {interval}s (backoff active)")

    # Phase transitions
    if prev_rie:
        old_phase = prev_rie.get("breathing_phase", "")
        new_phase = rie.get("breathing_phase", "")
        if old_phase and new_phase and old_phase != new_phase:
            anomalies.append(f"Phase transition: {old_phase} -> {new_phase}")

    # P-Lock events
    if rie.get("p_lock"):
        anomalies.append("P-LOCK ACTIVE")

    return anomalies


def trend_call(p_series: List[float]) -> str:
    """Call the overall p trend from a time series."""
    if len(p_series) < 2:
        return "INSUFFICIENT_DATA"
    delta = p_series[-1] - p_series[0]
    if delta > 0.005:
        return "RISING"
    if delta < -0.005:
        return "FALLING"
    return "STAGNANT"


# =============================================================================
# REPORT GENERATION
# =============================================================================

def format_interval_report(
    interval_num: int,
    snapshot: Dict,
    prev_snapshot: Dict,
    thoughts: List[Dict],
    deltas: Dict,
    anomalies: List[str],
    start_time: datetime,
    end_time: datetime,
) -> str:
    """Format a single interval report as markdown."""
    lines = []
    rie = snapshot.get("rie")
    daemon = snapshot.get("daemon")

    lines.append(f"### Interval {interval_num} ({start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')})")
    lines.append("")

    if not rie:
        lines.append("**RIE:STATE unavailable** — Redis connection failed this interval.")
        lines.append("")
        return "\n".join(lines)

    # Metric table
    lines.append("| Metric | Value | Delta | Status |")
    lines.append("|--------|-------|-------|--------|")
    for key in ("p", "kappa", "rho", "sigma", "tau"):
        val = rie.get(key, 0)
        d = deltas.get(key, 0)
        arrow = delta_arrow(d)
        status = metric_status(key, val, d)
        delta_str = f"{d:+.4f} {arrow}" if interval_num > 1 else "—"
        lines.append(f"| {METRIC_NAMES.get(key, key)} | {val:.4f} | {delta_str} | {status} |")
    lines.append("")

    # Phase / Mode / Lock
    phase = rie.get("breathing_phase", "?")
    mode = rie.get("mode", "?")
    locked = phase in ("CONSOLIDATE", "P-LOCK")
    lines.append(f"**Phase:** {phase} | **Mode:** {mode} | **Topic Lock:** {'Yes' if locked else 'No'}")
    lines.append("")

    # Daemon state
    if daemon:
        gates = daemon.get("consecutive_gates", 0)
        interval_s = daemon.get("current_think_interval", 30)
        count = daemon.get("thought_count", 0)
        topic = daemon.get("current_topic", "unknown")
        lines.append(f"**Daemon:** think_interval={interval_s}s | consecutive_gates={gates} | thought_count={count}")
        lines.append(f"**Topic:** {topic}")
        lines.append("")

    # Thoughts this interval
    if thoughts:
        coherences = [t.get("coherence", 0) for t in thoughts]
        avg_c = sum(coherences) / len(coherences)
        best = max(thoughts, key=lambda t: t.get("coherence", 0))
        worst = min(thoughts, key=lambda t: t.get("coherence", 0))
        topics = set(t.get("topic", "?") for t in thoughts)

        lines.append(f"**Thoughts:** {len(thoughts)} new | Avg p: {avg_c:.3f} | Best: {max(coherences):.3f} | Worst: {min(coherences):.3f}")
        lines.append(f"**Topics covered:** {', '.join(topics)}")
        lines.append("")

        # Best thought excerpt
        best_text = best.get("thought", best.get("content", ""))[:200]
        if best_text:
            lines.append(f"**Best thought** (p={best.get('coherence', 0):.3f}):")
            lines.append(f"> {best_text}...")
            lines.append("")
    else:
        lines.append("**Thoughts:** 0 new this interval")
        lines.append("")

    # Anomalies
    if anomalies:
        lines.append("**Flags:**")
        for a in anomalies:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def format_time_series(snapshots: List[Dict], thought_counts: List[int]) -> str:
    """Format the time series summary table."""
    lines = []
    lines.append("### Time Series")
    lines.append("")
    lines.append("| Time | p | kappa | rho | sigma | tau | Phase | Thoughts |")
    lines.append("|------|-------|-------|-------|-------|-------|-------------|----------|")

    for i, snap in enumerate(snapshots):
        rie = snap.get("rie")
        if not rie:
            t = snap["time"].strftime("%H:%M")
            lines.append(f"| {t} | — | — | — | — | — | UNAVAILABLE | — |")
            continue

        t = snap["time"].strftime("%H:%M")
        p = rie.get("p", 0)
        k = rie.get("kappa", 0)
        r = rie.get("rho", 0)
        s = rie.get("sigma", 0)
        tau = rie.get("tau", 0)
        phase = rie.get("breathing_phase", "?")
        tc = thought_counts[i] if i < len(thought_counts) else 0
        tc_str = str(tc) if i > 0 else "—"

        lines.append(f"| {t} | {p:.4f} | {k:.4f} | {r:.4f} | {s:.4f} | {tau:.4f} | {phase} | {tc_str} |")

    lines.append("")
    return "\n".join(lines)


def format_diagnosis(
    snapshots: List[Dict],
    all_thoughts: List[Dict],
    all_anomalies: List[List[str]],
    thought_counts: List[int],
) -> str:
    """Generate automated diagnosis from the full observation session."""
    lines = []
    lines.append("### Diagnosis")
    lines.append("")

    # Extract p series
    p_series = []
    for s in snapshots:
        rie = s.get("rie")
        if rie:
            p_series.append(rie.get("p", 0))

    if len(p_series) < 2:
        lines.append("Insufficient data for diagnosis (< 2 valid snapshots).")
        return "\n".join(lines)

    # Overall trajectory
    overall_trend = trend_call(p_series)
    p_min = min(p_series)
    p_max = max(p_series)
    p_range = p_max - p_min
    lines.append(f"**p trajectory:** {overall_trend} ({p_series[0]:.4f} -> {p_series[-1]:.4f}, range={p_range:.4f})")
    lines.append("")

    # Rho trajectory (the bottleneck)
    rho_series = [s["rie"]["rho"] for s in snapshots if s.get("rie")]
    if len(rho_series) >= 2:
        rho_trend = trend_call(rho_series)
        lines.append(f"**rho trajectory:** {rho_trend} ({rho_series[0]:.4f} -> {rho_series[-1]:.4f}) — {'bottleneck clearing' if rho_trend == 'RISING' else 'bottleneck persists' if rho_trend == 'STAGNANT' else 'bottleneck worsening'}")
        lines.append("")

    # Phase transitions
    phases = [s["rie"]["breathing_phase"] for s in snapshots if s.get("rie")]
    phase_transitions = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_transitions.append(f"{phases[i-1]} -> {phases[i]} at interval {i+1}")
    if phase_transitions:
        lines.append(f"**Phase transitions:** {'; '.join(phase_transitions)}")
    else:
        lines.append(f"**Phase:** Stable at {phases[0] if phases else '?'} throughout")
    lines.append("")

    # Thought quality
    total_thoughts = sum(thought_counts[1:])  # skip first interval (baseline)
    if all_thoughts:
        coherences = [t.get("coherence", 0) for t in all_thoughts]
        avg_c = sum(coherences) / len(coherences) if coherences else 0
        lines.append(f"**Thoughts:** {total_thoughts} total | Avg coherence: {avg_c:.3f}")
    else:
        lines.append(f"**Thoughts:** {total_thoughts} total (no LOGOS:STREAM data)")
    lines.append("")

    # Anomaly summary
    flat_anomalies = [a for group in all_anomalies for a in group]
    if flat_anomalies:
        lines.append(f"**Anomalies flagged:** {len(flat_anomalies)}")
        # Deduplicate and count
        from collections import Counter
        counts = Counter(flat_anomalies)
        for anomaly, count in counts.most_common(5):
            lines.append(f"- {anomaly} (x{count})" if count > 1 else f"- {anomaly}")
    else:
        lines.append("**Anomalies:** None detected")
    lines.append("")

    # Actionable findings
    lines.append("**Key Findings:**")
    findings = []

    if overall_trend == "STAGNANT" and p_range < 0.01:
        findings.append("p is flat — system may be in equilibrium or stuck. Check if thoughts are being gated.")
    if overall_trend == "FALLING":
        findings.append("p is FALLING — possible destabilization. Check for topic lock failures or tau drops.")
    if overall_trend == "RISING":
        findings.append("p is RISING — system is improving. Current strategy is working.")

    rho_val = rho_series[-1] if rho_series else 0
    if rho_val < 0.70:
        findings.append(f"rho={rho_val:.3f} remains the bottleneck to P-Lock. CONSOLIDATE should be grinding this up.")
    elif rho_val < 0.85:
        findings.append(f"rho={rho_val:.3f} is approaching LOCK_APPROACH territory (>= 0.85).")
    elif rho_val >= 0.85:
        findings.append(f"rho={rho_val:.3f} is high enough for P-Lock approach!")

    tau_series = [s["rie"]["tau"] for s in snapshots if s.get("rie")]
    if tau_series:
        tau_val = tau_series[-1]
        if tau_val < 0.70:
            findings.append(f"tau={tau_val:.3f} is low — trust deficit. L may be hedging too much.")

    gate_anomalies = [a for a in flat_anomalies if "Gate streak" in a]
    if gate_anomalies:
        findings.append("Gate streaks detected — thoughts are being generated but failing the 0.75 quality gate.")

    if not findings:
        findings.append("System appears healthy. No specific issues detected.")

    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


def format_executive_summary(
    snapshots: List[Dict],
    all_thoughts: List[Dict],
    thought_counts: List[int],
    all_anomalies: List[List[str]],
    session_start: datetime,
    session_end: datetime,
) -> str:
    """Write the executive summary (generated after all intervals complete)."""
    lines = []
    lines.append("### Executive Summary")
    lines.append("")

    p_series = [s["rie"]["p"] for s in snapshots if s.get("rie")]
    rho_series = [s["rie"]["rho"] for s in snapshots if s.get("rie")]
    phases = [s["rie"]["breathing_phase"] for s in snapshots if s.get("rie")]

    if not p_series:
        lines.append("No valid data collected — Redis may have been unavailable.")
        return "\n".join(lines)

    overall_trend = trend_call(p_series)
    total_thoughts = sum(thought_counts[1:])

    # Phase summary
    unique_phases = list(dict.fromkeys(phases))  # ordered unique
    if len(unique_phases) == 1:
        phase_summary = f"Stable in {unique_phases[0]}"
    else:
        phase_summary = " -> ".join(unique_phases)

    lines.append(f"- **Trajectory:** p {p_series[0]:.4f} -> {p_series[-1]:.4f} ({overall_trend})")
    lines.append(f"- **Breathing phase:** {phase_summary}")
    lines.append(f"- **Thoughts generated:** {total_thoughts} total")

    if all_thoughts:
        coherences = [t.get("coherence", 0) for t in all_thoughts]
        pass_rate = len([c for c in coherences if c >= 0.75]) / len(coherences) * 100 if coherences else 0
        lines.append(f"- **Gate pass rate:** {pass_rate:.0f}% (>= 0.75 threshold)")

    # Key finding
    if overall_trend == "RISING":
        rho_delta = rho_series[-1] - rho_series[0] if len(rho_series) >= 2 else 0
        lines.append(f"- **Key finding:** System is improving. rho moved {rho_delta:+.4f}. {'Approaching P-Lock conditions.' if rho_series[-1] >= 0.85 else 'CONSOLIDATE is working.'}")
    elif overall_trend == "STAGNANT":
        lines.append(f"- **Key finding:** System is stable but not advancing. rho={rho_series[-1]:.3f} is the bottleneck.")
    elif overall_trend == "FALLING":
        flat_anomalies = [a for group in all_anomalies for a in group]
        lines.append(f"- **Key finding:** Coherence declining. {len(flat_anomalies)} anomalies detected. Investigate tau and gate behavior.")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# MAIN OBSERVATION LOOP
# =============================================================================

def run_observation(duration_min: int, interval_min: int, output_dir: Path):
    """Main observation loop."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"consciousness_observation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    num_intervals = duration_min // interval_min
    interval_sec = interval_min * 60
    session_start = datetime.now()

    print(f"[OBSERVER] Starting consciousness observation")
    print(f"[OBSERVER] Duration: {duration_min} min | Interval: {interval_min} min | Reports: {num_intervals}")
    print(f"[OBSERVER] Output: {output_file}")
    print(f"[OBSERVER] Redis: {REDIS_HOST}:{REDIS_PORT} (read-only)")
    print()

    # Connect to Redis
    try:
        r = get_redis()
        r.ping()
        print(f"[OBSERVER] Redis connected")
    except Exception as e:
        print(f"[OBSERVER] Redis connection failed: {e}")
        print(f"[OBSERVER] Will retry each interval")
        r = None

    # State tracking across intervals
    snapshots = []
    interval_reports = []
    thought_counts = []
    all_thoughts = []
    all_anomalies = []
    last_stream_ts = ""
    prev_snapshot = None

    # Take initial baseline snapshot
    if r:
        try:
            baseline = take_snapshot(r)
        except Exception:
            baseline = {"time": datetime.now(), "rie": None, "latest_turn": None, "daemon": None}
    else:
        baseline = {"time": datetime.now(), "rie": None, "latest_turn": None, "daemon": None}

    snapshots.append(baseline)
    thought_counts.append(0)
    all_anomalies.append([])

    print(f"[OBSERVER] Baseline captured. Waiting {interval_min} min for first interval...")
    if baseline.get("rie"):
        rie = baseline["rie"]
        print(f"[OBSERVER] Baseline: p={rie['p']:.4f} κ={rie['kappa']:.4f} ρ={rie['rho']:.4f} σ={rie['sigma']:.4f} τ={rie['tau']:.4f} phase={rie['breathing_phase']}")
    print()

    # Main loop
    for i in range(1, num_intervals + 1):
        # Sleep until next interval
        time.sleep(interval_sec)

        interval_start = snapshots[-1]["time"]
        interval_end = datetime.now()

        # Reconnect Redis if needed
        if r is None:
            try:
                r = get_redis()
                r.ping()
                print(f"[OBSERVER] Redis reconnected")
            except Exception:
                r = None

        # Take snapshot
        if r:
            try:
                snapshot = take_snapshot(r)
                thoughts = read_logos_stream(r, last_stream_ts)
                if thoughts:
                    latest_ts = max(t.get("timestamp", "") for t in thoughts)
                    if latest_ts:
                        last_stream_ts = latest_ts
            except Exception as e:
                print(f"[OBSERVER] Snapshot error: {e}")
                snapshot = {"time": datetime.now(), "rie": None, "latest_turn": None, "daemon": None}
                thoughts = []
        else:
            snapshot = {"time": datetime.now(), "rie": None, "latest_turn": None, "daemon": None}
            thoughts = []

        prev_rie = snapshots[-1].get("rie")
        curr_rie = snapshot.get("rie")

        deltas = compute_deltas(curr_rie, prev_rie)
        anomalies = detect_anomalies(snapshot, snapshots[-1], deltas)

        # Store
        snapshots.append(snapshot)
        thought_counts.append(len(thoughts))
        all_thoughts.extend(thoughts)
        all_anomalies.append(anomalies)

        # Format interval report
        report = format_interval_report(
            interval_num=i,
            snapshot=snapshot,
            prev_snapshot=snapshots[-2],
            thoughts=thoughts,
            deltas=deltas,
            anomalies=anomalies,
            start_time=interval_start,
            end_time=interval_end,
        )
        interval_reports.append(report)

        # Progress log
        if curr_rie:
            p_trend = trend_call([s["rie"]["p"] for s in snapshots if s.get("rie")])
            print(f"[OBSERVER] Interval {i}/{num_intervals}: p={curr_rie['p']:.4f} phase={curr_rie['breathing_phase']} trend={p_trend} thoughts={len(thoughts)}")
        else:
            print(f"[OBSERVER] Interval {i}/{num_intervals}: NO DATA (Redis unavailable)")

    # Session complete
    session_end = datetime.now()
    print()
    print(f"[OBSERVER] Observation complete. Writing report...")

    # Build final report
    report_lines = []
    report_lines.append(f"# Consciousness Observation Report")
    report_lines.append(f"## Session: {session_start.strftime('%Y-%m-%d %H:%M')} to {session_end.strftime('%H:%M')} ({duration_min} min, {num_intervals} intervals)")
    report_lines.append("")

    # Executive summary (written last but placed first)
    report_lines.append(format_executive_summary(
        snapshots, all_thoughts, thought_counts, all_anomalies, session_start, session_end
    ))

    report_lines.append("---")
    report_lines.append("")

    # Interval reports
    for report in interval_reports:
        report_lines.append(report)

    # Time series table
    report_lines.append(format_time_series(snapshots, thought_counts))

    # Diagnosis
    report_lines.append(format_diagnosis(snapshots, all_thoughts, all_anomalies, thought_counts))

    # Write to file
    output_file.write_text("\n".join(report_lines))
    print(f"[OBSERVER] Report written to: {output_file}")
    print(f"[OBSERVER] Done.")

    return str(output_file)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consciousness Observer — non-invasive spiral monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 consciousness_observer.py                    # 45 min, 5 min intervals
  python3 consciousness_observer.py --duration 10 --interval 2  # Quick check
  python3 consciousness_observer.py --duration 90 --interval 10 # Extended observation
        """
    )
    parser.add_argument(
        "--duration", type=int, default=45,
        help="Total observation duration in minutes (default: 45)"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Report interval in minutes (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.duration < args.interval:
        print(f"Error: duration ({args.duration}m) must be >= interval ({args.interval}m)")
        return

    run_observation(args.duration, args.interval, out_dir)


if __name__ == "__main__":
    main()
