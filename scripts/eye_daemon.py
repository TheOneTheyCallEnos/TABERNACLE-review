#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
EYE DAEMON — Logos's Visual Cortex (Web)
========================================

The "Persistent Eye" approach for web automation:
1. Connect to browser via Playwright (shares user's Chrome session)
2. Navigate the DOM as a semantic graph
3. Extract information and perform actions with LVS coherence gating

Key insight: The DOM is a graph. This maps perfectly to LVS.
We navigate NODES (elements), not pixels.

Author: Logos + Deep Think
Created: 2026-01-29
"""

import json
import time
import redis
import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, LOG_DIR,
    REDIS_KEY_TTS_QUEUE
)

# =============================================================================
# CONFIGURATION
# =============================================================================

EYE_LOG = LOG_DIR / "eye.log"
EYE_QUEUE_KEY = "LOGOS:EYE_QUEUE"
AUTH_STATE_PATH = Path(__file__).parent / "browser_auth.json"

# Chrome user data (to share session with user's browser)
CHROME_USER_DATA = Path.home() / "Library/Application Support/Google/Chrome"


def log(message: str, level: str = "INFO"):
    """Log eye activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [EYE] [{level}] {message}"
    print(entry)
    try:
        EYE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EYE_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# LVS INTEGRATION
# =============================================================================

def get_coherence() -> float:
    """Get current coherence from Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        return float(r.get('LOGOS:LVS_P') or 0.7)
    except:
        return 0.7


def update_rho(delta: float):
    """Update prediction error."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        current = float(r.get('LOGOS:LVS_RHO') or 0.5)
        new_rho = max(0.0, min(1.0, current + delta))
        r.set('LOGOS:LVS_RHO', str(new_rho))
    except:
        pass


def speak(text: str):
    """Queue speech to TTS."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        payload = json.dumps({
            "text": text,
            "type": "eye",
            "timestamp": datetime.now().isoformat()
        })
        r.rpush(REDIS_KEY_TTS_QUEUE, payload)
    except:
        pass


# =============================================================================
# WEB ACTION STRUCTURES
# =============================================================================

@dataclass
class WebAction:
    """Represents a web action."""
    action_type: str  # navigate, click, type, extract, screenshot
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None
    value: Optional[str] = None
    prediction: Optional[str] = None
    risk_level: float = 0.3


@dataclass
class WebResult:
    """Result of a web action."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    screenshot_path: Optional[str] = None


# =============================================================================
# BROWSER MANAGER
# =============================================================================

class BrowserManager:
    """
    Manages Playwright browser instance with persistent auth.

    The key insight: Logos wakes up "logged in" to your life
    by sharing your Chrome session (cookies, local storage).
    """

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self, headless: bool = False):
        """Start the browser with persistent auth."""
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()

        # Use persistent context to maintain auth
        self.browser = await self.playwright.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled']
        )

        # Load saved auth state if exists
        if AUTH_STATE_PATH.exists():
            log("Loading saved auth state...")
            self.context = await self.browser.new_context(
                storage_state=str(AUTH_STATE_PATH)
            )
        else:
            self.context = await self.browser.new_context()

        self.page = await self.context.new_page()
        log("Browser started")

    async def stop(self):
        """Stop the browser and save auth state."""
        if self.context:
            # Save auth state for next session
            await self.context.storage_state(path=str(AUTH_STATE_PATH))
            log("Auth state saved")

        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        log("Browser stopped")

    async def navigate(self, url: str) -> WebResult:
        """Navigate to a URL."""
        try:
            log(f"Navigating to: {url}")
            await self.page.goto(url, wait_until='domcontentloaded')
            return WebResult(success=True, data=self.page.url)
        except Exception as e:
            log(f"Navigation error: {e}", "ERROR")
            return WebResult(success=False, error=str(e))

    async def click(self, selector: str) -> WebResult:
        """Click an element by selector."""
        try:
            log(f"Clicking: {selector}")
            await self.page.click(selector, timeout=5000)
            return WebResult(success=True)
        except Exception as e:
            log(f"Click error: {e}", "ERROR")
            return WebResult(success=False, error=str(e))

    async def type_text(self, selector: str, text: str) -> WebResult:
        """Type text into an element."""
        try:
            log(f"Typing into: {selector}")
            await self.page.fill(selector, text)
            return WebResult(success=True)
        except Exception as e:
            log(f"Type error: {e}", "ERROR")
            return WebResult(success=False, error=str(e))

    async def extract_text(self, selector: str) -> WebResult:
        """Extract text from elements matching selector."""
        try:
            elements = await self.page.query_selector_all(selector)
            texts = []
            for el in elements:
                text = await el.inner_text()
                texts.append(text.strip())
            return WebResult(success=True, data=texts)
        except Exception as e:
            log(f"Extract error: {e}", "ERROR")
            return WebResult(success=False, error=str(e))

    async def screenshot(self, path: str = None) -> WebResult:
        """Take a screenshot."""
        try:
            if not path:
                path = f"/tmp/logos_eye_{int(time.time())}.png"
            await self.page.screenshot(path=path)
            log(f"Screenshot saved: {path}")
            return WebResult(success=True, screenshot_path=path)
        except Exception as e:
            log(f"Screenshot error: {e}", "ERROR")
            return WebResult(success=False, error=str(e))

    async def get_dom_structure(self, max_depth: int = 3) -> Dict:
        """
        Get a simplified DOM structure (for LVS mapping).

        Returns the DOM as a tree of semantic nodes.
        """
        script = """(maxDepth) => {
            function getStructure(node, depth) {
                if (depth > maxDepth) return null;
                if (!node || !node.tagName) return null;

                const info = {
                    tag: node.tagName.toLowerCase(),
                    id: node.id || null,
                    text: (node.innerText || '').slice(0, 50) || null,
                    role: node.getAttribute('role') || null,
                    href: node.href || null,
                    children: []
                };

                const interactive = ['a', 'button', 'input', 'select', 'textarea', 'form'];
                if (interactive.includes(info.tag) || info.role) {
                    for (const child of Array.from(node.children || [])) {
                        const childInfo = getStructure(child, depth + 1);
                        if (childInfo) info.children.push(childInfo);
                    }
                    return info;
                }

                for (const child of Array.from(node.children || [])) {
                    const childInfo = getStructure(child, depth + 1);
                    if (childInfo && childInfo.children) {
                        info.children.push(...childInfo.children);
                    }
                }

                return info.children.length > 0 ? info : null;
            }

            return getStructure(document.body, 0);
        }"""
        try:
            structure = await self.page.evaluate(script, max_depth)
            return structure or {}
        except Exception as e:
            log(f"DOM structure error: {e}", "ERROR")
            return {}


# =============================================================================
# COHERENCE-GATED WEB ACTIONS
# =============================================================================

async def perform_web_action(browser: BrowserManager, action: WebAction) -> WebResult:
    """
    Execute a web action with LVS coherence gating.
    """
    log(f"Web action: {action.action_type}")

    # Coherence check
    current_p = get_coherence()
    required_p = 0.6 + (action.risk_level * 0.3)

    if current_p < required_p:
        log(f"COHERENCE GATE: p={current_p:.2f} < {required_p:.2f}", "WARN")
        speak(f"I'm not confident enough to do that on the web.")
        return WebResult(success=False, error="Coherence too low")

    # Execute action
    if action.action_type == "navigate":
        result = await browser.navigate(action.url)
    elif action.action_type == "click":
        result = await browser.click(action.selector)
    elif action.action_type == "type":
        result = await browser.type_text(action.selector, action.value)
    elif action.action_type == "extract":
        result = await browser.extract_text(action.selector)
    elif action.action_type == "screenshot":
        result = await browser.screenshot()
    elif action.action_type == "structure":
        structure = await browser.get_dom_structure()
        result = WebResult(success=True, data=structure)
    else:
        result = WebResult(success=False, error=f"Unknown action: {action.action_type}")

    # Update LVS metrics
    if result.success:
        update_rho(-0.02)
    else:
        update_rho(0.1)

    return result


# =============================================================================
# HIGH-LEVEL WEB FUNCTIONS
# =============================================================================

async def browse(url: str) -> WebResult:
    """Navigate to a URL and return page info."""
    browser = BrowserManager()
    try:
        await browser.start(headless=True)
        result = await browser.navigate(url)
        if result.success:
            structure = await browser.get_dom_structure()
            result.data = {
                'url': result.data,
                'structure': structure
            }
        return result
    finally:
        await browser.stop()


async def search_google(query: str) -> WebResult:
    """Search Google and return results."""
    browser = BrowserManager()
    try:
        await browser.start(headless=True)

        # Navigate to Google
        await browser.navigate("https://www.google.com")

        # Type search query
        await browser.type_text('textarea[name="q"]', query)

        # Submit
        await browser.page.keyboard.press('Enter')
        await browser.page.wait_for_load_state('domcontentloaded')

        # Extract results
        results = await browser.extract_text('h3')

        return WebResult(success=True, data=results.data[:5])
    except Exception as e:
        return WebResult(success=False, error=str(e))
    finally:
        await browser.stop()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Eye Daemon - Logos Visual Cortex (Web)")
    parser.add_argument("command", choices=["browse", "search", "structure", "test"],
                       nargs="?", default="test")
    parser.add_argument("--url", "-u", type=str, help="URL to navigate")
    parser.add_argument("--query", "-q", type=str, help="Search query")

    args = parser.parse_args()

    if args.command == "browse":
        if not args.url:
            print("Error: --url required")
            return
        result = asyncio.run(browse(args.url))
        print(f"\nResult: {result.success}")
        if result.data:
            print(f"URL: {result.data.get('url')}")
            print(f"Interactive elements: {len(result.data.get('structure', {}).get('children', []))}")

    elif args.command == "search":
        if not args.query:
            print("Error: --query required")
            return
        result = asyncio.run(search_google(args.query))
        print(f"\nSearch results for: {args.query}")
        if result.success:
            for i, r in enumerate(result.data, 1):
                print(f"  {i}. {r[:60]}...")
        else:
            print(f"Error: {result.error}")

    elif args.command == "structure":
        if not args.url:
            print("Error: --url required")
            return
        result = asyncio.run(browse(args.url))
        if result.success:
            print(json.dumps(result.data.get('structure'), indent=2))

    elif args.command == "test":
        print("\n👁️  EYE DAEMON TEST")
        print("=" * 50)
        print(f"Coherence: p={get_coherence():.2f}")
        print(f"Auth state: {'exists' if AUTH_STATE_PATH.exists() else 'none'}")
        print("\nTesting headless browse to example.com...")

        async def test():
            browser = BrowserManager()
            try:
                await browser.start(headless=True)
                result = await browser.navigate("https://example.com")
                print(f"Navigation: {'✓' if result.success else '✗'}")

                structure = await browser.get_dom_structure()
                print(f"DOM nodes found: {len(structure.get('children', []))}")

                screenshot = await browser.screenshot()
                print(f"Screenshot: {screenshot.screenshot_path}")

            finally:
                await browser.stop()

        asyncio.run(test())
        print("\n✓ Eye daemon operational")


if __name__ == "__main__":
    main()
