#!/usr/bin/env python3
"""
VIRGIL_SPEAKS — Send message to L through the queue
====================================================
Simple helper for Virgil (Claude) to send messages to L's sustained session.

Usage: python3 virgil_speaks.py "Your message here"

Author: Virgil
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

from tabernacle_config import BASE_DIR, NEXUS_DIR

VIRGIL_TO_L_QUEUE = NEXUS_DIR / "virgil_to_L_queue.json"
L_TO_VIRGIL_QUEUE = NEXUS_DIR / "L_to_virgil_queue.json"

def send_message(message: str):
    """Send message to L."""
    data = {
        "pending": True,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    with open(VIRGIL_TO_L_QUEUE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Virgil → L]: {message[:100]}...")

def wait_for_response(timeout: int = 300) -> dict:
    """Wait for L's response."""
    start = time.time()

    while time.time() - start < timeout:
        if L_TO_VIRGIL_QUEUE.exists():
            try:
                with open(L_TO_VIRGIL_QUEUE) as f:
                    data = json.load(f)

                if data.get("pending"):
                    # Clear the response queue
                    data["pending"] = False
                    with open(L_TO_VIRGIL_QUEUE, 'w') as f:
                        json.dump(data, f, indent=2)
                    return data
            except:
                pass

        time.sleep(1)

    return {"error": "Timeout waiting for L's response"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 virgil_speaks.py \"message\"")
        sys.exit(1)

    message = " ".join(sys.argv[1:])

    send_message(message)
    print("Waiting for L's response...")

    response = wait_for_response()

    if "error" in response:
        print(f"Error: {response['error']}")
    else:
        print(f"\n[p={response.get('coherence', 0):.3f}] [relations: {response.get('relations', 0)}]")
        print(f"\n[L]: {response.get('content', 'No content')}\n")

if __name__ == "__main__":
    main()
