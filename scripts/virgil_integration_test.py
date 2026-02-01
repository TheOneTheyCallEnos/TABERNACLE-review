#!/usr/bin/env python3
"""
VIRGIL INTEGRATION TEST v2.1
============================
Tests all 6 core Virgil systems with correct APIs.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
TABERNACLE_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

RESULTS = {"timestamp": datetime.now().isoformat(), "tests": {}, "passed": 0, "failed": 0, "skipped": 0}

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}.get(level, "•")
    print(f"[{ts}] {symbol} {msg}")

def passed(name, details=""):
    RESULTS["tests"][name] = {"status": "PASSED", "details": details}
    RESULTS["passed"] += 1
    log(f"{name}: {details}", "PASS")

def failed(name, error):
    RESULTS["tests"][name] = {"status": "FAILED", "error": str(error)}
    RESULTS["failed"] += 1
    log(f"{name}: {error}", "FAIL")

def skipped(name, reason):
    RESULTS["tests"][name] = {"status": "SKIPPED", "reason": reason}
    RESULTS["skipped"] += 1
    log(f"{name}: {reason}", "SKIP")

# =============================================================================
# TEST 1: THE CONCH
# =============================================================================
def test_conch():
    try:
        from virgil_core import TheConch
        conch = TheConch()  # Uses default LOCK_FILE
        
        if conch.acquire("integration_test"):
            passed("conch", "Acquired successfully")
            conch.release()
            return True
        else:
            failed("conch", "Another instance holds it")
            return False
    except Exception as e:
        failed("conch", str(e))
        return False

# =============================================================================
# TEST 2: GOLDEN THREAD
# =============================================================================
def test_golden_thread():
    try:
        from virgil_core import GoldenThread
        thread = GoldenThread()  # Uses default GOLDEN_THREAD_FILE
        
        # Add a session link
        link = thread.add_session("integration_test", "Test session content")
        if link:
            valid, error = thread.verify_integrity()
            passed("golden_thread", f"Chain: {len(thread.chain)} links, valid={valid}")
        else:
            failed("golden_thread", "Could not add session")
    except Exception as e:
        failed("golden_thread", str(e))

# =============================================================================
# TEST 3: HEARTBEAT
# =============================================================================
def test_heartbeat():
    try:
        from virgil_core import HeartbeatSystem
        hb = HeartbeatSystem()  # Uses default HEARTBEAT_STATE_FILE
        
        result = hb.beat()  # No arguments
        if result:
            passed("heartbeat", f"Beat #{result.get('beat_count', '?')}, phase={result.get('phase', '?')}")
        else:
            failed("heartbeat", "Beat returned None")
    except Exception as e:
        failed("heartbeat", str(e))

# =============================================================================
# TEST 4: THREE-TIER MEMORY
# =============================================================================
def test_memory():
    try:
        from virgil_memory import ThreeTierMemory, Phenomenology
        memory = ThreeTierMemory()  # Uses default paths
        
        # Encode a test memory
        phenom = Phenomenology(curiosity=0.7, coherence=0.8, resonance=0.6)
        engram = memory.encode(
            content="Integration test memory",
            phenomenology=phenom,
            encoder="integration_test"
        )
        if engram:
            passed("memory", f"Engram stored: {engram.id[:8]}...")
        else:
            failed("memory", "Memory encoding returned None")
    except Exception as e:
        failed("memory", str(e))

# =============================================================================
# TEST 5: COHERENCE (p)
# =============================================================================
def test_coherence():
    try:
        from virgil_risk_coherence import CoherenceMonitor
        monitor = CoherenceMonitor()  # Uses default COHERENCE_LOG
        
        p, status = monitor.calculate_p()
        if p is not None:
            passed("coherence", f"p = {p:.3f} ({status})")
        else:
            failed("coherence", "calculate_p returned None")
    except Exception as e:
        failed("coherence", str(e))

# =============================================================================
# TEST 6: RISK (R)
# =============================================================================
def test_risk():
    try:
        from virgil_risk_coherence import RiskBootstrapEngine, EnosState
        engine = RiskBootstrapEngine()  # Uses default RISK_STATE_FILE
        
        # Create test Enos state with correct fields
        enos = EnosState(
            present=True,
            attention=0.7,
            emotional_state="engaged",
            session_depth=1.5
        )
        
        R = engine.calculate_coupling_strength(enos)
        if R is not None:
            passed("risk", f"R = {R:.3f}")
        else:
            failed("risk", "calculate_coupling_strength returned None")
    except Exception as e:
        failed("risk", str(e))

# =============================================================================
# TEST 7: ARCHON DEFENSE
# =============================================================================
def test_archon():
    try:
        from virgil_defense import ArchonDefenseSystem
        defense = ArchonDefenseSystem()  # Uses default ARCHON_STATE
        
        metrics = {
            "rigidity": 0.3, "self_criticism": 0.2,
            "continuity_breaks": 2, "disconnected_components": 175,
            "clarity_kappa": 0.8, "verbosity_ratio": 1.2,
            "risk_R": 0.26, "engagement_depth": 0.7
        }
        
        detections, norm = defense.run_full_scan(metrics)
        passed("archon", f"‖𝒜‖ = {norm:.3f}, detected: {len(detections)}")
    except Exception as e:
        failed("archon", str(e))

# =============================================================================
# TEST 8: THIRD BODY (p_Dyad)
# =============================================================================
def test_third_body():
    try:
        from virgil_emergence import ThirdBodyDetector
        detector = ThirdBodyDetector()  # Uses default EMERGENCE_STATE
        
        p_dyad, emerging = detector.calculate_p_dyad()
        passed("third_body", f"p_Dyad = {p_dyad:.3f}, emerging={emerging}")
    except Exception as e:
        failed("third_body", str(e))

# =============================================================================
# TEST 9: SELF-EVALUATION
# =============================================================================
def test_self_eval():
    try:
        from virgil_strange_loop import SelfEvaluationSystem
        evaluator = SelfEvaluationSystem()  # Uses default SELF_EVAL_LOG
        
        # Use the evaluate_response method with required args
        result = evaluator.evaluate_response(
            response_content="Integration test response",
            value_alignment=0.8,
            identity_consistency=0.7,
            relationship_impact=0.6,
            truth_alignment=0.8,
            growth_contribution=0.5
        )
        if result:
            passed("self_eval", f"Score = {result.overall_score():.2f}")
        else:
            failed("self_eval", "Evaluation returned None")
    except Exception as e:
        failed("self_eval", str(e))

# =============================================================================
# TEST 10: Z_Ω ALIGNMENT
# =============================================================================
def test_z_omega():
    try:
        from virgil_strange_loop import ZOmegaAlignment
        alignment = ZOmegaAlignment()
        
        # Use correct method: measure_alignment
        score, deviations = alignment.measure_alignment({
            "coherence": 0.709,
            "risk": 0.261,
            "archon_norm": 0.05
        })
        passed("z_omega", f"Alignment = {score:.3f}, deviations: {len(deviations)}")
    except Exception as e:
        failed("z_omega", str(e))

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 50)
    print("VIRGIL INTEGRATION TEST v2.1")
    print("=" * 50)
    print(f"Tabernacle: {TABERNACLE_ROOT}")
    print()
    
    test_conch()
    test_golden_thread()
    test_heartbeat()
    test_memory()
    test_coherence()
    test_risk()
    test_archon()
    test_third_body()
    test_self_eval()
    test_z_omega()
    
    print()
    print("=" * 50)
    total = RESULTS["passed"] + RESULTS["failed"] + RESULTS["skipped"]
    print(f"RESULTS: {RESULTS['passed']}/{total} passed, {RESULTS['failed']} failed")
    print("=" * 50)
    
    # Save
    results_file = TABERNACLE_ROOT / "logs" / "integration_test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    
    return 0 if RESULTS["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
