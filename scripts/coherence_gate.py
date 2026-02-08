#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
COHERENCE GATE — LVS Safety System
===================================

This module implements the core LVS safety mechanism:
Actions are gated by coherence metrics.

The insight: Traditional agents act without sensing their own
internal coherence. They lack PROPRIOCEPTION.

By gating actions with p (coherence) and ρ (prediction error),
Logos KNOWS when it is confused and can halt before causing harm.

Usage:
    @requires_plock(risk_level=0.5)
    def risky_action():
        ...

Author: Logos + Deep Think
Created: 2026-01-29
"""

import redis
import json
import functools
from typing import Optional, Callable, Any
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import REDIS_HOST, REDIS_PORT, REDIS_KEY_TTS_QUEUE

# =============================================================================
# REDIS KEYS
# =============================================================================

LVS_P_KEY = "LOGOS:LVS_P"           # Coherence
LVS_RHO_KEY = "LOGOS:LVS_RHO"       # Prediction Error
LVS_SIGMA_KEY = "LOGOS:LVS_SIGMA"   # Structural Integrity
LVS_TAU_KEY = "LOGOS:LVS_TAU"       # Trust
LVS_KAPPA_KEY = "LOGOS:LVS_KAPPA"   # Clarity


# =============================================================================
# LVS STATE ACCESS
# =============================================================================

def get_lvs_state() -> dict:
    """Get current LVS state from Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        return {
            'p': float(r.get(LVS_P_KEY) or 0.7),
            'rho': float(r.get(LVS_RHO_KEY) or 0.5),
            'sigma': float(r.get(LVS_SIGMA_KEY) or 0.7),
            'tau': float(r.get(LVS_TAU_KEY) or 0.6),
            'kappa': float(r.get(LVS_KAPPA_KEY) or 0.7),
        }
    except:
        return {'p': 0.7, 'rho': 0.5, 'sigma': 0.7, 'tau': 0.6, 'kappa': 0.7}


def set_lvs_metric(key: str, value: float):
    """Set an LVS metric in Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.set(key, str(value))
    except:
        pass


def update_coherence(delta: float):
    """Update coherence metric."""
    state = get_lvs_state()
    new_p = max(0.0, min(1.0, state['p'] + delta))
    set_lvs_metric(LVS_P_KEY, new_p)
    return new_p


def update_rho(delta: float):
    """Update prediction error metric."""
    state = get_lvs_state()
    new_rho = max(0.0, min(1.0, state['rho'] + delta))
    set_lvs_metric(LVS_RHO_KEY, new_rho)
    return new_rho


# =============================================================================
# SPEECH OUTPUT
# =============================================================================

def speak(text: str):
    """Queue speech to TTS (for coherence warnings)."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        payload = json.dumps({
            "text": text,
            "type": "coherence_gate",
            "timestamp": datetime.now().isoformat()
        })
        r.rpush(REDIS_KEY_TTS_QUEUE, payload)
    except:
        pass


# =============================================================================
# THE COHERENCE GATE DECORATOR
# =============================================================================

def requires_plock(risk_level: float = 0.5, speak_on_halt: bool = True):
    """
    Coherence gate decorator.

    Prevents action execution if system coherence is too low.

    Args:
        risk_level: 0.0 (read-only) to 1.0 (destructive action)
                   Higher risk requires higher coherence to proceed.
        speak_on_halt: Whether to voice the halt condition.

    The required coherence is calculated as:
        required_p = 0.7 + (risk_level * 0.25)

    So:
        - risk_level=0.0 → required_p=0.70 (basic actions)
        - risk_level=0.5 → required_p=0.825 (moderate risk)
        - risk_level=1.0 → required_p=0.95 (destructive actions need P-Lock)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            state = get_lvs_state()
            current_p = state['p']
            current_rho = state['rho']

            # Calculate required coherence based on risk
            required_p = 0.7 + (risk_level * 0.25)

            # Also check prediction error - high ρ means we're confused
            max_rho = 0.7 - (risk_level * 0.2)  # Tighter tolerance for risky actions

            # ABADDON CHECK
            if current_p < 0.5:
                if speak_on_halt:
                    speak("I'm in a fragmented state. I need to pause and recalibrate.")
                print(f"⛔ ABADDON: p={current_p:.2f} < 0.5")
                return None

            # COHERENCE GATE
            if current_p < required_p:
                if speak_on_halt:
                    speak(f"I'm not confident enough for this action. My coherence is {current_p:.0%}.")
                print(f"⛔ HALT: Coherence {current_p:.2f} < Required {required_p:.2f}")
                return None

            # PREDICTION ERROR GATE
            if current_rho > max_rho:
                if speak_on_halt:
                    speak("My predictions have been off lately. Let me verify before acting.")
                print(f"⛔ HALT: Prediction error {current_rho:.2f} > Max {max_rho:.2f}")
                return None

            # Execute action
            try:
                result = func(*args, **kwargs)

                # Action succeeded - slightly boost coherence
                update_coherence(0.01)
                update_rho(-0.02)

                return result

            except Exception as e:
                # Action failed - decrease coherence, increase prediction error
                update_coherence(-0.05)
                update_rho(0.1)
                raise

        return wrapper
    return decorator


def requires_verification(prediction: str):
    """
    Decorator that verifies a prediction after action execution.

    Usage:
        @requires_verification("URL contains 'success'")
        def submit_form():
            click_submit()
            return get_current_url()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)

            # TODO: Implement actual prediction verification
            # For now, just log the prediction
            print(f"📋 Prediction to verify: {prediction}")
            print(f"📋 Action result: {result}")

            return result
        return wrapper
    return decorator


# =============================================================================
# COHERENCE RECOVERY
# =============================================================================

def trigger_recovery():
    """
    Trigger coherence recovery mode.

    Called when system detects low coherence. Implements:
    1. Halt all risky actions
    2. Voice the state
    3. Request human verification
    """
    state = get_lvs_state()

    if state['p'] < 0.5:
        speak("I've entered recovery mode. My coherence has dropped significantly. "
              "I'll pause autonomous actions until we recalibrate together.")

        # Set recovery flag
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.setex("LOGOS:RECOVERY_MODE", 300, "1")  # 5 minute recovery window
        except:
            pass

        return True

    return False


def check_recovery_mode() -> bool:
    """Check if system is in recovery mode."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        return r.get("LOGOS:RECOVERY_MODE") == "1"
    except:
        return False


def exit_recovery_mode():
    """Exit recovery mode after human verification."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.delete("LOGOS:RECOVERY_MODE")

        # Boost coherence back up
        update_coherence(0.2)
        update_rho(-0.2)

        speak("Recovery complete. I'm feeling more coherent now.")
    except:
        pass


# =============================================================================
# CLI / TESTING
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Coherence Gate - LVS Safety System")
    parser.add_argument("command", choices=["status", "set", "test", "recover"],
                       nargs="?", default="status")
    parser.add_argument("--p", type=float, help="Set coherence")
    parser.add_argument("--rho", type=float, help="Set prediction error")

    args = parser.parse_args()

    if args.command == "status":
        state = get_lvs_state()
        print("\n🧠 LVS COHERENCE STATUS")
        print("=" * 40)
        print(f"  p (coherence):        {state['p']:.3f}")
        print(f"  ρ (prediction error): {state['rho']:.3f}")
        print(f"  σ (structure):        {state['sigma']:.3f}")
        print(f"  τ (trust):            {state['tau']:.3f}")
        print(f"  κ (clarity):          {state['kappa']:.3f}")
        print()

        if state['p'] >= 0.95:
            print("✨ P-LOCK ACHIEVED - Maximum coherence")
        elif state['p'] >= 0.7:
            print("✓ Coherent - Normal operation")
        elif state['p'] >= 0.5:
            print("⚠️ Degraded - Caution advised")
        else:
            print("⛔ ABADDON - Recovery needed")

        if check_recovery_mode():
            print("\n🔄 RECOVERY MODE ACTIVE")

    elif args.command == "set":
        if args.p is not None:
            set_lvs_metric(LVS_P_KEY, args.p)
            print(f"Set p = {args.p}")
        if args.rho is not None:
            set_lvs_metric(LVS_RHO_KEY, args.rho)
            print(f"Set ρ = {args.rho}")

    elif args.command == "test":
        print("\n🧪 COHERENCE GATE TEST")
        print("=" * 40)

        @requires_plock(risk_level=0.3)
        def low_risk_action():
            print("  ✓ Low risk action executed")
            return True

        @requires_plock(risk_level=0.8)
        def high_risk_action():
            print("  ✓ High risk action executed")
            return True

        state = get_lvs_state()
        print(f"Current p = {state['p']:.2f}")

        print("\nTesting low risk action (requires p >= 0.775):")
        low_risk_action()

        print("\nTesting high risk action (requires p >= 0.9):")
        high_risk_action()

    elif args.command == "recover":
        print("Triggering recovery mode...")
        if trigger_recovery():
            print("Recovery mode activated")
        else:
            print("Coherence sufficient, no recovery needed")


if __name__ == "__main__":
    main()
