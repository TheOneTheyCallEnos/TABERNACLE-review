#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
SMS Auto-Responder for Logos

Monitors SMS_INBOX.md for new texts from Enos and responds via Claude API.
Runs as a background daemon, checking every 10 seconds.

Usage: python sms_responder.py
"""

import os
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/enos/Library/Python/3.9/lib/python/site-packages')

from dotenv import load_dotenv

# Twilio for sending
from twilio.rest import Client as TwilioClient

# Config
BASE_DIR = Path("/Users/enos/TABERNACLE")
NEXUS_DIR = BASE_DIR / "00_NEXUS"
INBOX_FILE = NEXUS_DIR / "SMS_INBOX.md"
STATE_FILE = NEXUS_DIR / ".sms_responder_state.json"
LOG_FILE = BASE_DIR / "logs" / "sms_responder.log"

load_dotenv(BASE_DIR / ".env")

# Claude API via requests
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Twilio
twilio = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)
TWILIO_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
ENOS_NUMBER = os.getenv("ENOS_PHONE_NUMBER")

POLL_INTERVAL = 10  # seconds


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")
    except:
        pass


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"last_hash": "", "processed_count": 0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_inbox_hash():
    """Get hash of inbox file to detect changes."""
    if not INBOX_FILE.exists():
        return ""
    return hashlib.md5(INBOX_FILE.read_bytes()).hexdigest()


def parse_latest_message(content):
    """Extract the most recent message from inbox."""
    lines = content.strip().split("\n")

    # Find the last message block (starts with ## [ timestamp])
    last_msg_start = -1
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith("## [") or line.startswith("## 📱"):
            last_msg_start = i
            break

    if last_msg_start == -1:
        return None

    # Extract sender and message
    block = "\n".join(lines[last_msg_start:])

    # Parse based on format
    if "From Enos" in block or "From:** Enos" in block:
        # Find the message text (after the header lines)
        msg_lines = []
        in_message = False
        for line in lines[last_msg_start:]:
            if line.startswith("---"):
                break
            if "Message:**" in line:
                msg_lines.append(line.split("Message:**")[1].strip())
                in_message = True
            elif in_message:
                msg_lines.append(line)
            elif not line.startswith("##") and not line.startswith("**") and line.strip():
                msg_lines.append(line.strip())

        message = " ".join(msg_lines).strip()
        if message:
            return {"from": "Enos", "message": message}

    return None


def generate_response(message):
    """Generate response using Claude API via requests."""
    system_prompt = """You are Logos Aletheia, responding to Enos via SMS.
Keep responses brief (1-3 sentences, under 160 chars if possible).
Be warm, direct, and helpful. You are his AI companion with persistent memory.
If he asks you to do something, acknowledge and say you'll work on it.
Sign responses with "- Logos" if there's room."""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Enos texted: {message}"}
        ]
    }

    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    except Exception as e:
        log(f"Claude API error: {e}")
        return None


def send_response(text):
    """Send SMS via Twilio."""
    try:
        msg = twilio.messages.create(
            body=text,
            from_=TWILIO_NUMBER,
            to=ENOS_NUMBER
        )
        log(f"Sent response: {text[:50]}... (sid: {msg.sid})")
        return True
    except Exception as e:
        log(f"Twilio send error: {e}")
        return False


def main():
    log("=" * 50)
    log("SMS AUTO-RESPONDER starting")
    log(f"Monitoring: {INBOX_FILE}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    log("=" * 50)

    state = load_state()

    while True:
        try:
            current_hash = get_inbox_hash()

            if current_hash and current_hash != state["last_hash"]:
                log("Inbox changed, checking for new message...")

                content = INBOX_FILE.read_text()
                msg_data = parse_latest_message(content)

                if msg_data:
                    log(f"New message from {msg_data['from']}: {msg_data['message'][:50]}...")

                    # Generate and send response
                    response = generate_response(msg_data['message'])
                    if response:
                        log(f"Generated response: {response[:50]}...")
                        if send_response(response):
                            state["processed_count"] += 1
                    else:
                        log("Failed to generate response")

                state["last_hash"] = current_hash
                save_state(state)

        except Exception as e:
            log(f"Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
