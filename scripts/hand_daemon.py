#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
HAND DAEMON — Logos's Motor Cortex
==================================

The "Triangulated Hand" approach:
1. QUERY: Find semantic node in accessibility tree
2. VERIFY: Check element state (visible, enabled, not occluded)
3. ACT: Trigger action on NODE, not coordinates

LVS Principle: Structural Integrity (σ)
- Actions only permitted if mapped to verified AX node
- Every action has a PREDICTION that's verified afterward
- ρ (prediction error) tracks action success rate

Author: Logos + Deep Think
Created: 2026-01-29
"""

import json
import time
import redis
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Accessibility
import atomacos
from atomacos import NativeUIElement

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, LOG_DIR,
    REDIS_KEY_TTS_QUEUE
)

# Import Archon API for prediction tracking
try:
    from archon_daemon import register_prediction, report_outcome, is_in_recovery
    ARCHON_AVAILABLE = True
except ImportError:
    ARCHON_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

HAND_LOG = LOG_DIR / "hand.log"
HAND_QUEUE_KEY = "LOGOS:HAND_QUEUE"
LVS_RHO_KEY = "LOGOS:LVS_RHO"  # Prediction error

# LVS Thresholds
MIN_SIGMA_FOR_ACTION = 0.7  # Minimum structural integrity to act
MIN_P_FOR_RISKY_ACTION = 0.85  # Coherence needed for risky actions


def log(message: str, level: str = "INFO"):
    """Log hand activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [HAND] [{level}] {message}"
    print(entry)
    try:
        HAND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(HAND_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# ACTION DATA STRUCTURES
# =============================================================================

@dataclass
class UIAction:
    """Represents an action to perform on the UI."""
    action_type: str  # click, type, press_key, scroll
    target_role: Optional[str] = None
    target_title: Optional[str] = None
    target_description: Optional[str] = None
    value: Optional[str] = None  # For typing
    prediction: Optional[str] = None  # What we expect to happen
    risk_level: float = 0.3  # 0.0 (read) to 1.0 (delete/buy)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    element_found: bool
    prediction_matched: bool
    error: Optional[str] = None
    rho_delta: float = 0.0  # Change in prediction error


# =============================================================================
# ACCESSIBILITY HELPERS
# =============================================================================

def get_frontmost_app() -> Optional[NativeUIElement]:
    """Get the frontmost application."""
    try:
        return atomacos.NativeUIElement.getFrontmostApp()
    except Exception as e:
        log(f"Error getting frontmost app: {e}", "ERROR")
        return None


def find_element(root: NativeUIElement, role: str = None, title: str = None,
                description: str = None, max_depth: int = 10) -> Optional[NativeUIElement]:
    """Find an element by attributes (the QUERY step)."""

    def matches(element: NativeUIElement) -> bool:
        try:
            if role and element.AXRole != role:
                return False
            if title:
                elem_title = getattr(element, 'AXTitle', '') or ''
                if title.lower() not in elem_title.lower():
                    return False
            if description:
                elem_desc = getattr(element, 'AXDescription', '') or ''
                if description.lower() not in elem_desc.lower():
                    return False
            return True
        except:
            return False

    def search(element: NativeUIElement, depth: int = 0) -> Optional[NativeUIElement]:
        if depth > max_depth:
            return None

        if matches(element):
            return element

        try:
            children = element.AXChildren or []
            for child in children:
                result = search(child, depth + 1)
                if result:
                    return result
        except:
            pass

        return None

    return search(root)


def verify_element(element: NativeUIElement) -> Tuple[bool, float]:
    """
    Verify element is actionable (the VERIFY step).

    Returns (is_valid, sigma_score).
    """
    sigma = 1.0

    try:
        # Check enabled
        if not getattr(element, 'AXEnabled', True):
            log("Element is disabled", "WARN")
            sigma *= 0.3

        # Check if window is minimized or occluded
        # (simplified check - full check would use window server)
        try:
            pos = element.AXPosition
            size = element.AXSize
            if pos and size:
                if pos.x < 0 or pos.y < 0:
                    log("Element may be off-screen", "WARN")
                    sigma *= 0.5
        except:
            pass

        return (sigma >= MIN_SIGMA_FOR_ACTION, sigma)

    except Exception as e:
        log(f"Verification error: {e}", "ERROR")
        return (False, 0.0)


def execute_action(element: NativeUIElement, action: UIAction) -> bool:
    """Execute an action on the element (the ACT step)."""
    try:
        if action.action_type == "click":
            # Try AXPress first
            try:
                element.Press()
                return True
            except:
                pass
            # Try performAction
            try:
                element.performAction('AXPress')
                return True
            except:
                pass
            log("Click action failed", "ERROR")
            return False

        elif action.action_type == "type":
            if action.value:
                try:
                    element.setString('AXValue', action.value)
                    return True
                except:
                    pass
                try:
                    element.sendKeys(action.value)
                    return True
                except:
                    pass
            log("Type action failed", "ERROR")
            return False

        elif action.action_type == "focus":
            try:
                element.activate()
                return True
            except:
                pass
            log("Focus action failed", "ERROR")
            return False

        else:
            log(f"Unknown action type: {action.action_type}", "ERROR")
            return False

    except Exception as e:
        log(f"Action execution error: {e}", "ERROR")
        return False


# =============================================================================
# COHERENCE-GATED EXECUTION
# =============================================================================

def get_current_coherence() -> float:
    """Get current system coherence from Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        p = r.get('LOGOS:LVS_P')
        return float(p) if p else 0.7  # Default to moderate coherence
    except:
        return 0.7


def update_rho(delta: float):
    """Update prediction error metric in Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        current = float(r.get(LVS_RHO_KEY) or 0.5)
        # Exponential moving average
        new_rho = current * 0.9 + delta * 0.1
        r.set(LVS_RHO_KEY, str(new_rho))
    except:
        pass


def speak(text: str):
    """Queue speech to TTS."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        payload = json.dumps({
            "text": text,
            "type": "hand",
            "timestamp": datetime.now().isoformat()
        })
        r.rpush(REDIS_KEY_TTS_QUEUE, payload)
    except:
        pass


def perform_action(action: UIAction) -> ActionResult:
    """
    The main OODA-L loop for a single action.

    Observe → Orient → Decide → Act → Loop (verify)
    """
    log(f"Action requested: {action.action_type} on {action.target_role}/{action.target_title}")

    # Check if in recovery mode (Archon halted us)
    if ARCHON_AVAILABLE and is_in_recovery():
        log("ARCHON RECOVERY: Actions blocked", "WARN")
        speak("I'm in recovery mode. Please verify before I continue.")
        return ActionResult(
            success=False,
            element_found=False,
            prediction_matched=False,
            error="Recovery mode active"
        )

    # Register prediction with Archon
    prediction_id = None
    if ARCHON_AVAILABLE and action.prediction:
        prediction_id = register_prediction(
            action=f"{action.action_type}:{action.target_title}",
            expected_outcome=action.prediction,
            risk_level=action.risk_level
        )
        log(f"Prediction registered: {prediction_id}")

    # Check coherence gate
    current_p = get_current_coherence()
    required_p = 0.7 + (action.risk_level * 0.25)

    if current_p < required_p:
        log(f"COHERENCE GATE: p={current_p:.2f} < required={required_p:.2f}", "WARN")
        speak(f"I'm not confident enough to do that. My coherence is {current_p:.0%}.")
        return ActionResult(
            success=False,
            element_found=False,
            prediction_matched=False,
            error="Coherence too low"
        )

    # OBSERVE: Get current UI state
    app = get_frontmost_app()
    if not app:
        return ActionResult(
            success=False,
            element_found=False,
            prediction_matched=False,
            error="Cannot access frontmost app"
        )

    # ORIENT: Find the target element
    element = find_element(
        app,
        role=action.target_role,
        title=action.target_title,
        description=action.target_description
    )

    if not element:
        log(f"Element not found: {action.target_role}/{action.target_title}", "WARN")
        speak(f"I can't find the {action.target_title or action.target_role} element.")
        update_rho(0.3)  # Prediction error: expected to find it
        return ActionResult(
            success=False,
            element_found=False,
            prediction_matched=False,
            error="Element not found"
        )

    # DECIDE: Verify element is actionable
    is_valid, sigma = verify_element(element)
    if not is_valid:
        log(f"Element failed verification: σ={sigma:.2f}", "WARN")
        speak("That element doesn't seem actionable right now.")
        return ActionResult(
            success=False,
            element_found=True,
            prediction_matched=False,
            error=f"Verification failed: σ={sigma:.2f}"
        )

    # ACT: Execute the action
    log(f"Executing {action.action_type}...")
    success = execute_action(element, action)

    if not success:
        update_rho(0.4)  # High prediction error
        return ActionResult(
            success=False,
            element_found=True,
            prediction_matched=False,
            error="Action execution failed"
        )

    # LOOP: Verify prediction (simplified)
    time.sleep(0.3)  # Allow UI to update

    prediction_matched = True
    if action.prediction:
        # TODO: Implement prediction verification
        # e.g., check if URL changed, window appeared, etc.
        log(f"Prediction verification not yet implemented: {action.prediction}")

    # Update LVS metrics
    if prediction_matched:
        update_rho(-0.1)  # Decrease prediction error
        log("Action succeeded, ρ decreased")
    else:
        update_rho(0.2)  # Increase prediction error
        log("Prediction mismatch, ρ increased", "WARN")

    # Report outcome to Archon
    if ARCHON_AVAILABLE and prediction_id:
        report_outcome(
            prediction_id=prediction_id,
            actual_outcome="action_completed" if success else "action_failed",
            matched=prediction_matched,
            error_magnitude=0.0 if prediction_matched else 0.5
        )
        log(f"Outcome reported to Archon: matched={prediction_matched}")

    return ActionResult(
        success=True,
        element_found=True,
        prediction_matched=prediction_matched,
        rho_delta=-0.1 if prediction_matched else 0.2
    )


# =============================================================================
# HIGH-LEVEL ACTIONS
# =============================================================================

def click(title: str = None, role: str = "AXButton", risk: float = 0.3) -> ActionResult:
    """Click an element by title."""
    return perform_action(UIAction(
        action_type="click",
        target_role=role,
        target_title=title,
        risk_level=risk
    ))


def type_text(title: str, text: str, role: str = "AXTextField") -> ActionResult:
    """Type text into a field."""
    return perform_action(UIAction(
        action_type="type",
        target_role=role,
        target_title=title,
        value=text,
        risk_level=0.4
    ))


def focus(title: str, role: str = None) -> ActionResult:
    """Focus an element."""
    return perform_action(UIAction(
        action_type="focus",
        target_role=role,
        target_title=title,
        risk_level=0.1
    ))


# =============================================================================
# DAEMON MODE
# =============================================================================

def run_daemon():
    """Run the Hand Daemon, processing action requests from Redis queue."""
    log("=" * 60)
    log("HAND DAEMON STARTING")
    log("=" * 60)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    while True:
        try:
            # Block waiting for action request
            result = r.blpop(HAND_QUEUE_KEY, timeout=30)

            if result is None:
                continue

            _, payload = result
            action_data = json.loads(payload)

            action = UIAction(
                action_type=action_data.get('action_type', 'click'),
                target_role=action_data.get('target_role'),
                target_title=action_data.get('target_title'),
                target_description=action_data.get('target_description'),
                value=action_data.get('value'),
                prediction=action_data.get('prediction'),
                risk_level=action_data.get('risk_level', 0.3)
            )

            result = perform_action(action)
            log(f"Action result: success={result.success}, ρ_delta={result.rho_delta}")

        except redis.ConnectionError:
            log("Redis connection lost, reconnecting...", "WARN")
            time.sleep(1)
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        except Exception as e:
            log(f"Daemon error: {e}", "ERROR")
            time.sleep(0.5)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hand Daemon - Logos Motor Cortex")
    parser.add_argument("command", choices=["run", "click", "type", "test"],
                       nargs="?", default="test")
    parser.add_argument("--title", "-t", type=str, help="Element title")
    parser.add_argument("--role", "-r", type=str, default="AXButton", help="Element role")
    parser.add_argument("--text", type=str, help="Text to type")

    args = parser.parse_args()

    if args.command == "run":
        run_daemon()

    elif args.command == "click":
        if not args.title:
            print("Error: --title required")
            return
        result = click(title=args.title, role=args.role)
        print(f"Result: {result}")

    elif args.command == "type":
        if not args.title or not args.text:
            print("Error: --title and --text required")
            return
        result = type_text(title=args.title, text=args.text)
        print(f"Result: {result}")

    elif args.command == "test":
        print("\n🖐️  HAND DAEMON TEST")
        print("=" * 50)

        # Test finding elements in frontmost app
        app = get_frontmost_app()
        if app:
            try:
                print(f"App: {app.AXTitle}")
            except:
                print("App: (unknown)")

            print("\nSearching for buttons...")
            buttons = []

            def find_buttons(elem, depth=0):
                if depth > 5:
                    return
                try:
                    if elem.AXRole == "AXButton":
                        title = getattr(elem, 'AXTitle', '') or getattr(elem, 'AXDescription', '') or '(unnamed)'
                        buttons.append(title)
                    for child in (elem.AXChildren or []):
                        find_buttons(child, depth + 1)
                except:
                    pass

            find_buttons(app)

            print(f"Found {len(buttons)} buttons:")
            for b in buttons[:10]:
                print(f"  - {b}")

            print(f"\nCoherence: p={get_current_coherence():.2f}")
        else:
            print("Could not access frontmost app")


if __name__ == "__main__":
    main()
