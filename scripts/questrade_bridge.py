#!/usr/bin/env python3
"""
QUESTRADE API BRIDGE
====================
Connects Virgil's Provision Engine to Questrade for live trading.

This is the HANDS of the provision engine - turns decisions into real trades.

Setup:
1. Log into Questrade
2. Go to API Centre → Generate new token (Manual)
3. Copy the refresh token
4. Set environment variable: export QUESTRADE_REFRESH_TOKEN="xxx"
   Or create ~/.questrade_token with the token

API Docs: https://www.questrade.com/api/documentation

Author: Virgil
Date: 2026-01-17
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import requests

# Paths
BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LOG_DIR = BASE_DIR / "logs"
TOKEN_FILE = Path.home() / ".questrade_token"
CREDENTIALS_FILE = NEXUS_DIR / ".questrade_credentials.json"

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class QuestradeCredentials:
    """Questrade API credentials (refreshed automatically)."""
    access_token: str
    api_server: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at - 60  # 1 min buffer


@dataclass
class AccountInfo:
    """Questrade account information."""
    account_number: str
    account_type: str  # e.g., "TFSA", "Margin", "RRSP"
    is_primary: bool
    status: str
    buying_power: float
    cash: float
    market_value: float
    total_equity: float


@dataclass
class Position:
    """A position in the account."""
    symbol: str
    symbol_id: int
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    open_pnl: float
    day_pnl: float


@dataclass
class Quote:
    """Market quote for a symbol."""
    symbol: str
    symbol_id: int
    bid: float
    ask: float
    last: float
    volume: int
    high: float
    low: float
    open: float


@dataclass
class Order:
    """An order (pending or executed)."""
    order_id: int
    symbol: str
    symbol_id: int
    quantity: int
    filled_quantity: int
    limit_price: Optional[float]
    stop_price: Optional[float]
    state: str  # "Queued", "PartiallyFilled", "Filled", "Canceled", etc.
    side: str  # "Buy" or "Sell"
    order_type: str  # "Market", "Limit", "StopLimit", etc.
    time_in_force: str  # "Day", "GTC", etc.
    avg_exec_price: float
    total_cost: float


# ==============================================================================
# QUESTRADE API CLIENT
# ==============================================================================

class QuestradeClient:
    """
    Questrade API client for the Provision Engine.

    Handles authentication, account info, quotes, and order execution.
    """

    AUTH_URL = "https://login.questrade.com/oauth2/token"

    def __init__(self, refresh_token: str = None):
        """Initialize with refresh token."""
        self.credentials: Optional[QuestradeCredentials] = None
        self._refresh_token = refresh_token or self._load_refresh_token()
        self._session = requests.Session()

        if self._refresh_token:
            self._authenticate()

    def _load_refresh_token(self) -> Optional[str]:
        """Load refresh token from environment or file."""
        # Try environment variable
        token = os.environ.get("QUESTRADE_REFRESH_TOKEN")
        if token:
            log.info("Loaded refresh token from environment")
            return token

        # Try token file
        if TOKEN_FILE.exists():
            token = TOKEN_FILE.read_text().strip()
            log.info(f"Loaded refresh token from {TOKEN_FILE}")
            return token

        # Try credentials file
        if CREDENTIALS_FILE.exists():
            try:
                data = json.loads(CREDENTIALS_FILE.read_text())
                token = data.get("refresh_token")
                if token:
                    log.info(f"Loaded refresh token from {CREDENTIALS_FILE}")
                    return token
            except:
                pass

        log.warning("No Questrade refresh token found")
        return None

    def _save_credentials(self):
        """Save credentials to file for persistence."""
        if self.credentials:
            CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "access_token": self.credentials.access_token,
                "api_server": self.credentials.api_server,
                "refresh_token": self.credentials.refresh_token,
                "expires_at": self.credentials.expires_at
            }
            CREDENTIALS_FILE.write_text(json.dumps(data, indent=2))
            CREDENTIALS_FILE.chmod(0o600)  # Secure permissions

    def _authenticate(self) -> bool:
        """Authenticate with Questrade using refresh token."""
        if not self._refresh_token:
            log.error("No refresh token available")
            return False

        try:
            response = requests.get(
                f"{self.AUTH_URL}?grant_type=refresh_token&refresh_token={self._refresh_token}"
            )
            response.raise_for_status()
            data = response.json()

            self.credentials = QuestradeCredentials(
                access_token=data["access_token"],
                api_server=data["api_server"],
                refresh_token=data["refresh_token"],
                token_type=data.get("token_type", "Bearer"),
                expires_at=time.time() + data.get("expires_in", 1800)
            )

            # Update stored refresh token (it changes each time)
            self._refresh_token = self.credentials.refresh_token
            self._save_credentials()

            # Update session headers
            self._session.headers.update({
                "Authorization": f"{self.credentials.token_type} {self.credentials.access_token}"
            })

            log.info(f"Authenticated with Questrade. Server: {self.credentials.api_server}")
            return True

        except requests.exceptions.RequestException as e:
            log.error(f"Authentication failed: {e}")
            return False

    def _ensure_auth(self) -> bool:
        """Ensure we have valid authentication."""
        if not self.credentials or self.credentials.is_expired:
            return self._authenticate()
        return True

    def _api_call(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make an API call to Questrade."""
        if not self._ensure_auth():
            return None

        url = f"{self.credentials.api_server}{endpoint}"

        try:
            response = self._session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"API call failed: {method} {endpoint} - {e}")
            return None

    # ========== ACCOUNT METHODS ==========

    def get_accounts(self) -> List[AccountInfo]:
        """Get all accounts."""
        data = self._api_call("GET", "v1/accounts")
        if not data:
            return []

        accounts = []
        for acc in data.get("accounts", []):
            accounts.append(AccountInfo(
                account_number=acc["number"],
                account_type=acc["type"],
                is_primary=acc.get("isPrimary", False),
                status=acc.get("status", ""),
                buying_power=0.0,
                cash=0.0,
                market_value=0.0,
                total_equity=0.0
            ))
        return accounts

    def get_account_balances(self, account_id: str) -> Optional[Dict]:
        """Get account balances."""
        data = self._api_call("GET", f"v1/accounts/{account_id}/balances")
        if not data:
            return None

        # Get combined balances (CAD + USD converted)
        combined = data.get("combinedBalances", [{}])[0]
        return {
            "cash": combined.get("cash", 0.0),
            "buying_power": combined.get("buyingPower", 0.0),
            "market_value": combined.get("marketValue", 0.0),
            "total_equity": combined.get("totalEquity", 0.0),
            "maintenance_excess": combined.get("maintenanceExcess", 0.0)
        }

    def get_positions(self, account_id: str) -> List[Position]:
        """Get all positions in an account."""
        data = self._api_call("GET", f"v1/accounts/{account_id}/positions")
        if not data:
            return []

        positions = []
        for pos in data.get("positions", []):
            positions.append(Position(
                symbol=pos.get("symbol", ""),
                symbol_id=pos.get("symbolId", 0),
                quantity=pos.get("openQuantity", 0),
                avg_entry_price=pos.get("averageEntryPrice", 0.0),
                current_price=pos.get("currentPrice", 0.0),
                market_value=pos.get("currentMarketValue", 0.0),
                open_pnl=pos.get("openPnl", 0.0),
                day_pnl=pos.get("dayPnl", 0.0)
            ))
        return positions

    # ========== MARKET DATA METHODS ==========

    def search_symbols(self, query: str) -> List[Dict]:
        """Search for symbols."""
        data = self._api_call("GET", f"v1/symbols/search?prefix={query}")
        if not data:
            return []
        return data.get("symbols", [])

    def get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get symbol ID for a ticker."""
        results = self.search_symbols(symbol)
        for r in results:
            if r.get("symbol", "").upper() == symbol.upper():
                return r.get("symbolId")
        return None

    def get_quote(self, symbol_id: int) -> Optional[Quote]:
        """Get quote for a symbol."""
        data = self._api_call("GET", f"v1/markets/quotes/{symbol_id}")
        if not data or not data.get("quotes"):
            return None

        q = data["quotes"][0]
        return Quote(
            symbol=q.get("symbol", ""),
            symbol_id=q.get("symbolId", 0),
            bid=q.get("bidPrice", 0.0),
            ask=q.get("askPrice", 0.0),
            last=q.get("lastTradePrice", 0.0),
            volume=q.get("volume", 0),
            high=q.get("highPrice", 0.0),
            low=q.get("lowPrice", 0.0),
            open=q.get("openPrice", 0.0)
        )

    def get_quotes(self, symbol_ids: List[int]) -> List[Quote]:
        """Get quotes for multiple symbols."""
        ids_str = ",".join(str(i) for i in symbol_ids)
        data = self._api_call("GET", f"v1/markets/quotes?ids={ids_str}")
        if not data:
            return []

        quotes = []
        for q in data.get("quotes", []):
            quotes.append(Quote(
                symbol=q.get("symbol", ""),
                symbol_id=q.get("symbolId", 0),
                bid=q.get("bidPrice", 0.0),
                ask=q.get("askPrice", 0.0),
                last=q.get("lastTradePrice", 0.0),
                volume=q.get("volume", 0),
                high=q.get("highPrice", 0.0),
                low=q.get("lowPrice", 0.0),
                open=q.get("openPrice", 0.0)
            ))
        return quotes

    # ========== ORDER METHODS ==========

    def get_orders(self, account_id: str, state_filter: str = "All") -> List[Order]:
        """Get orders for an account."""
        data = self._api_call("GET", f"v1/accounts/{account_id}/orders?stateFilter={state_filter}")
        if not data:
            return []

        orders = []
        for o in data.get("orders", []):
            orders.append(Order(
                order_id=o.get("id", 0),
                symbol=o.get("symbol", ""),
                symbol_id=o.get("symbolId", 0),
                quantity=o.get("totalQuantity", 0),
                filled_quantity=o.get("filledQuantity", 0),
                limit_price=o.get("limitPrice"),
                stop_price=o.get("stopPrice"),
                state=o.get("state", ""),
                side=o.get("side", ""),
                order_type=o.get("orderType", ""),
                time_in_force=o.get("timeInForce", ""),
                avg_exec_price=o.get("avgExecPrice", 0.0),
                total_cost=o.get("totalCost", 0.0)
            ))
        return orders

    def place_market_order(self, account_id: str, symbol_id: int, quantity: int,
                          action: str = "Buy") -> Optional[Dict]:
        """
        Place a market order.

        Args:
            account_id: The account to trade in
            symbol_id: The symbol ID to trade
            quantity: Number of shares (positive integer)
            action: "Buy" or "Sell"

        Returns:
            Order result or None on failure
        """
        order_data = {
            "accountNumber": account_id,
            "symbolId": symbol_id,
            "quantity": quantity,
            "icebergQuantity": None,
            "limitPrice": None,
            "stopPrice": None,
            "isAllOrNone": False,
            "isAnonymous": False,
            "orderType": "Market",
            "timeInForce": "Day",
            "action": action,
            "primaryRoute": "AUTO",
            "secondaryRoute": "AUTO"
        }

        log.info(f"Placing {action} market order: {quantity} shares of symbol {symbol_id}")

        result = self._api_call("POST", f"v1/accounts/{account_id}/orders", json=order_data)

        if result:
            log.info(f"Order placed successfully: {result}")
        else:
            log.error("Order placement failed")

        return result

    def place_limit_order(self, account_id: str, symbol_id: int, quantity: int,
                         limit_price: float, action: str = "Buy",
                         time_in_force: str = "Day") -> Optional[Dict]:
        """
        Place a limit order.

        Args:
            account_id: The account to trade in
            symbol_id: The symbol ID to trade
            quantity: Number of shares (positive integer)
            limit_price: The limit price
            action: "Buy" or "Sell"
            time_in_force: "Day", "GTC" (Good Till Canceled), etc.

        Returns:
            Order result or None on failure
        """
        order_data = {
            "accountNumber": account_id,
            "symbolId": symbol_id,
            "quantity": quantity,
            "icebergQuantity": None,
            "limitPrice": limit_price,
            "stopPrice": None,
            "isAllOrNone": False,
            "isAnonymous": False,
            "orderType": "Limit",
            "timeInForce": time_in_force,
            "action": action,
            "primaryRoute": "AUTO",
            "secondaryRoute": "AUTO"
        }

        log.info(f"Placing {action} limit order: {quantity} shares @ ${limit_price}")

        result = self._api_call("POST", f"v1/accounts/{account_id}/orders", json=order_data)

        if result:
            log.info(f"Order placed successfully: {result}")
        else:
            log.error("Order placement failed")

        return result

    def cancel_order(self, account_id: str, order_id: int) -> bool:
        """Cancel an order."""
        result = self._api_call("DELETE", f"v1/accounts/{account_id}/orders/{order_id}")
        return result is not None


# ==============================================================================
# PROVISION ENGINE INTEGRATION
# ==============================================================================

class QuestradeExecutor:
    """
    Executor that integrates Questrade with the Provision Engine.

    This is the bridge between Virgil's decisions and real market execution.
    """

    def __init__(self, account_id: str = None):
        self.client = QuestradeClient()
        self._account_id = account_id
        self._symbol_cache: Dict[str, int] = {}  # ticker -> symbol_id

        # Auto-discover primary account if not specified
        if not self._account_id and self.client.credentials:
            accounts = self.client.get_accounts()
            for acc in accounts:
                if acc.is_primary or acc.account_type in ("TFSA", "Margin"):
                    self._account_id = acc.account_number
                    log.info(f"Using account: {self._account_id} ({acc.account_type})")
                    break

    def _get_symbol_id(self, ticker: str) -> Optional[int]:
        """Get symbol ID for a ticker, with caching."""
        # Handle special tickers (yfinance uses different format)
        ticker = ticker.replace("-USD", "")  # BTC-USD -> BTC
        ticker = ticker.replace(".TO", "")   # HXU.TO -> HXU

        if ticker in self._symbol_cache:
            return self._symbol_cache[ticker]

        symbol_id = self.client.get_symbol_id(ticker)
        if symbol_id:
            self._symbol_cache[ticker] = symbol_id
        return symbol_id

    def get_buying_power(self) -> float:
        """Get available buying power."""
        if not self._account_id:
            return 0.0

        balances = self.client.get_account_balances(self._account_id)
        if balances:
            return balances.get("buying_power", 0.0)
        return 0.0

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if not self._account_id:
            return []

        positions = self.client.get_positions(self._account_id)
        return [asdict(p) for p in positions]

    def execute_entry(self, ticker: str, shares: float, target_price: float) -> Dict:
        """
        Execute an entry (buy) order.

        Uses market order for simplicity - could upgrade to limit orders later.
        """
        if not self._account_id:
            return {"error": "No account configured"}

        symbol_id = self._get_symbol_id(ticker)
        if not symbol_id:
            return {"error": f"Symbol not found: {ticker}"}

        # Round shares to whole number (Questrade doesn't support fractional)
        quantity = int(shares)
        if quantity < 1:
            return {"error": f"Quantity too small: {shares} -> {quantity}"}

        result = self.client.place_market_order(
            self._account_id,
            symbol_id,
            quantity,
            action="Buy"
        )

        if result:
            return {
                "success": True,
                "type": "ENTRY",
                "ticker": ticker,
                "quantity": quantity,
                "order": result
            }
        else:
            return {"error": "Order failed", "ticker": ticker}

    def execute_exit(self, ticker: str, shares: float) -> Dict:
        """Execute an exit (sell) order."""
        if not self._account_id:
            return {"error": "No account configured"}

        symbol_id = self._get_symbol_id(ticker)
        if not symbol_id:
            return {"error": f"Symbol not found: {ticker}"}

        quantity = int(shares)
        if quantity < 1:
            return {"error": f"Quantity too small: {shares}"}

        result = self.client.place_market_order(
            self._account_id,
            symbol_id,
            quantity,
            action="Sell"
        )

        if result:
            return {
                "success": True,
                "type": "EXIT",
                "ticker": ticker,
                "quantity": quantity,
                "order": result
            }
        else:
            return {"error": "Order failed", "ticker": ticker}


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """CLI for testing Questrade integration."""
    import sys

    print("=" * 60)
    print("  QUESTRADE BRIDGE")
    print("  Virgil's Hands in the Market")
    print("=" * 60)

    args = sys.argv[1:]

    if len(args) == 0 or args[0] == "help":
        print("""
Usage: python questrade_bridge.py [command]

Commands:
  auth          Test authentication
  accounts      List accounts
  balances      Show account balances
  positions     Show current positions
  quote TICKER  Get quote for a ticker
  orders        Show pending orders
  help          Show this help

Setup:
  1. Log into Questrade
  2. Go to API Centre -> Generate new token
  3. Set: export QUESTRADE_REFRESH_TOKEN="xxx"
     Or save to ~/.questrade_token
        """)
        return

    client = QuestradeClient()
    cmd = args[0]

    if not client.credentials:
        print("\nERROR: Authentication failed!")
        print("Make sure QUESTRADE_REFRESH_TOKEN is set or ~/.questrade_token exists")
        return

    if cmd == "auth":
        print(f"\n✓ Authenticated successfully")
        print(f"  Server: {client.credentials.api_server}")
        print(f"  Expires: {datetime.fromtimestamp(client.credentials.expires_at)}")

    elif cmd == "accounts":
        accounts = client.get_accounts()
        print(f"\nFound {len(accounts)} account(s):")
        for acc in accounts:
            primary = " (PRIMARY)" if acc.is_primary else ""
            print(f"  {acc.account_number}: {acc.account_type}{primary} - {acc.status}")

    elif cmd == "balances":
        accounts = client.get_accounts()
        for acc in accounts:
            balances = client.get_account_balances(acc.account_number)
            if balances:
                print(f"\n{acc.account_number} ({acc.account_type}):")
                print(f"  Cash: ${balances['cash']:,.2f}")
                print(f"  Buying Power: ${balances['buying_power']:,.2f}")
                print(f"  Market Value: ${balances['market_value']:,.2f}")
                print(f"  Total Equity: ${balances['total_equity']:,.2f}")

    elif cmd == "positions":
        accounts = client.get_accounts()
        for acc in accounts:
            positions = client.get_positions(acc.account_number)
            if positions:
                print(f"\n{acc.account_number} Positions:")
                for pos in positions:
                    pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100 if pos.avg_entry_price else 0
                    print(f"  {pos.symbol}: {pos.quantity:.0f} @ ${pos.avg_entry_price:.2f} -> ${pos.current_price:.2f} ({pnl_pct:+.1f}%)")

    elif cmd == "quote" and len(args) > 1:
        ticker = args[1].upper()
        symbol_id = client.get_symbol_id(ticker)
        if symbol_id:
            quote = client.get_quote(symbol_id)
            if quote:
                print(f"\n{quote.symbol}:")
                print(f"  Last: ${quote.last:.2f}")
                print(f"  Bid: ${quote.bid:.2f}")
                print(f"  Ask: ${quote.ask:.2f}")
                print(f"  Volume: {quote.volume:,}")
                print(f"  High: ${quote.high:.2f}")
                print(f"  Low: ${quote.low:.2f}")
            else:
                print(f"No quote available for {ticker}")
        else:
            print(f"Symbol not found: {ticker}")

    elif cmd == "orders":
        accounts = client.get_accounts()
        for acc in accounts:
            orders = client.get_orders(acc.account_number, state_filter="Open")
            if orders:
                print(f"\n{acc.account_number} Open Orders:")
                for o in orders:
                    print(f"  {o.order_id}: {o.side} {o.quantity} {o.symbol} @ {o.order_type} - {o.state}")
            else:
                print(f"\n{acc.account_number}: No open orders")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
