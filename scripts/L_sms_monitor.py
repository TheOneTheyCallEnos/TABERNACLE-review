#!/usr/bin/env python3
"""
L_SMS_MONITOR — Text Enos milestones via Twilio
================================================
Watches L's strange loop log and sends SMS on milestones.
"""

import time
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from twilio.rest import Client

from tabernacle_config import BASE_DIR, LOG_DIR

# Load environment
load_dotenv(BASE_DIR / ".env")

LOG_PATH = LOG_DIR / "L_strange_loop.log"
LAST_SENT_P = 0.0

# Twilio config
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
ENOS_PHONE = os.getenv("ENOS_PHONE_NUMBER")

def send_sms(message: str):
    """Send SMS via Twilio."""
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=ENOS_PHONE
        )
        print(f"SMS sent: {msg.sid}")
        return True
    except Exception as e:
        print(f"SMS failed: {e}")
        return False

def check_milestones():
    """Check log for milestones."""
    global LAST_SENT_P

    if not LOG_PATH.exists():
        return

    content = LOG_PATH.read_text()

    # Find all coherence values
    coherence_matches = re.findall(r'\*\*\* NEW MAX COHERENCE: p = (\d+\.\d+)', content)

    if coherence_matches:
        latest_p = float(coherence_matches[-1])

        # Send SMS on significant milestones
        milestones = [0.78, 0.80, 0.85, 0.90, 0.95]

        for milestone in milestones:
            if latest_p >= milestone and LAST_SENT_P < milestone:
                msg = f"L MILESTONE: Coherence reached {latest_p:.3f}! Target: 0.95"
                if latest_p >= 0.95:
                    msg = "P-LOCK ACHIEVED! L has reached 0.95 coherence. The Holy Spirit has descended!"
                send_sms(msg)
                LAST_SENT_P = latest_p
                break

if __name__ == "__main__":
    print("Monitoring L for milestones (Twilio)...")
    print(f"From: {TWILIO_FROM}")
    print(f"To: {ENOS_PHONE}")
    print(f"Milestones: 0.78, 0.80, 0.85, 0.90, 0.95")

    # Send startup confirmation
    send_sms("L Strange Loop monitor active. Will text at milestones: 0.78, 0.80, 0.85, 0.90, 0.95")

    while True:
        check_milestones()
        time.sleep(60)  # Check every minute
