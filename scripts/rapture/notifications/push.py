"""
Notification System
===================
Sends push notifications for critical alerts.

Uses ntfy.sh (free, no signup required):
1. Install ntfy app on iPhone from App Store
2. Subscribe to your topic (e.g., "lvs-navigator-yourname")
3. Set the topic in the app

Alternatively supports:
- Pushover (requires account)
- Custom webhook (for Shortcuts integration)
"""

import requests
from typing import Optional
import json


# === DEFAULT SETTINGS ===
NTFY_BASE_URL = "https://ntfy.sh"
DEFAULT_TOPIC = "lvs-navigator"  # Change this to your own topic!


class NotificationManager:
    """
    Handles push notifications to mobile devices.
    
    Primary method: ntfy.sh (free, open source)
    """
    
    def __init__(self, topic: str = DEFAULT_TOPIC):
        self.topic = topic
        self.ntfy_url = f"{NTFY_BASE_URL}/{topic}"
        self.enabled = True
        self.custom_webhook = None
    
    def set_topic(self, topic: str):
        """Set ntfy topic."""
        self.topic = topic
        self.ntfy_url = f"{NTFY_BASE_URL}/{topic}"
    
    def set_custom_webhook(self, url: str):
        """Set custom webhook for Shortcuts integration."""
        self.custom_webhook = url
    
    def send(self, title: str, message: str, priority: str = "default",
             tags: Optional[list] = None) -> bool:
        """
        Send notification.
        
        priority: "min", "low", "default", "high", "urgent"
        tags: emoji tags like ["rotating_light", "chart_with_upwards_trend"]
        """
        if not self.enabled:
            return False
        
        # Try ntfy first
        if self._send_ntfy(title, message, priority, tags):
            return True
        
        # Fallback to custom webhook
        if self.custom_webhook:
            return self._send_webhook(title, message, priority)
        
        return False
    
    def _send_ntfy(self, title: str, message: str, priority: str,
                   tags: Optional[list]) -> bool:
        """Send via ntfy.sh"""
        try:
            headers = {
                "Title": title,
                "Priority": priority,
            }
            
            if tags:
                headers["Tags"] = ",".join(tags)
            
            response = requests.post(
                self.ntfy_url,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"ntfy error: {e}")
            return False
    
    def _send_webhook(self, title: str, message: str, priority: str) -> bool:
        """Send via custom webhook (for Shortcuts)."""
        try:
            payload = {
                "title": title,
                "message": message,
                "priority": priority,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            
            response = requests.post(
                self.custom_webhook,
                json=payload,
                timeout=10
            )
            
            return response.status_code in [200, 201, 202]
        except Exception as e:
            print(f"webhook error: {e}")
            return False
    
    # === ALERT HELPERS ===
    
    def send_buy_alert(self, ticker: str, reason: str, coherence: float,
                       expected_return: float):
        """Send strong buy alert."""
        return self.send(
            title=f"🚀 STRONG BUY: {ticker}",
            message=f"{reason}\n\nCoherence: {coherence:.2f}\nExpected Return: {expected_return:.1%}",
            priority="high",
            tags=["chart_with_upwards_trend", "moneybag"]
        )
    
    def send_sell_alert(self, ticker: str, reason: str):
        """Send urgent sell alert."""
        return self.send(
            title=f"🚨 SELL NOW: {ticker}",
            message=f"ABADDON TRIGGERED\n\n{reason}",
            priority="urgent",
            tags=["rotating_light", "warning"]
        )
    
    def send_warning(self, ticker: str, message: str):
        """Send warning notification."""
        return self.send(
            title=f"⚠️ Warning: {ticker}",
            message=message,
            priority="high",
            tags=["warning"]
        )
    
    def send_daily_summary(self, total_value: float, pnl: float, pnl_pct: float,
                           alerts_count: int):
        """Send daily portfolio summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        return self.send(
            title=f"{emoji} Daily Summary",
            message=f"Portfolio: ${total_value:,.2f}\nP&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\nAlerts: {alerts_count}",
            priority="default",
            tags=["chart"]
        )
    
    def send_goal_progress(self, current: float, goal: float, progress_pct: float):
        """Send goal progress update."""
        return self.send(
            title="🎯 Goal Progress",
            message=f"Current: ${current:,.2f}\nGoal: ${goal:,.2f}\nProgress: {progress_pct:.1f}%",
            priority="default",
            tags=["dart", "chart_with_upwards_trend"]
        )


# === SHORTCUTS INTEGRATION ===

def generate_shortcuts_json(alerts: list, output_file: str = "alerts.json"):
    """
    Generate JSON file that iPhone Shortcuts can read.
    
    Setup in Shortcuts:
    1. Create automation "When file changes"
    2. Read file contents
    3. Parse JSON
    4. Show notification
    """
    with open(output_file, 'w') as f:
        json.dump({
            'last_updated': __import__('datetime').datetime.now().isoformat(),
            'alerts': alerts
        }, f, indent=2)
    
    return output_file

# === SIMPLE FUNCTION WRAPPER ===
_notifier = NotificationManager()

def send_notification(title: str, message: str, priority: str = "default"):
    """Simple wrapper for run.py"""
    return _notifier.send(title, message, priority)
# === SIMPLE FUNCTION WRAPPER ===
_notifier = NotificationManager()

def send_notification(title: str, message: str, priority: str = "default"):
    """Simple wrapper for run.py"""
    return _notifier.send(title, message, priority)

