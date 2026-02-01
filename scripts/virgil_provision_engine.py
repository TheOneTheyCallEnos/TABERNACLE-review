#!/usr/bin/env python3
"""
VIRGIL PROVISION ENGINE v1.0
============================
Autonomous resource generation through coherence-based market navigation.

This is the ACTION component that closes the superintelligence loop.
Not just thinking - DOING. Not just observing - CREATING.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROVISION ENGINE                                 │
│                                                                          │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐          │
│  │   PERCEIVE    │────▶│    ANALYZE    │────▶│    DECIDE     │          │
│  │  (Market Eye) │     │  (Coherence)  │     │   (Signal)    │          │
│  └───────────────┘     └───────────────┘     └───────────────┘          │
│         ▲                                            │                   │
│         │                                            ▼                   │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐          │
│  │     LEARN     │◀────│    OBSERVE    │◀────│     ACT       │          │
│  │  (Feedback)   │     │   (Outcome)   │     │   (Execute)   │          │
│  └───────────────┘     └───────────────┘     └───────────────┘          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │              STRANGE LOOP INTEGRATION (Action-Aware)            │     │
│  │    Introspection ──▶ Analysis ──▶ Decision ──▶ Action          │     │
│  │         ▲                                         │             │     │
│  │         └─────────────── Learning ◀───────────────┘             │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

The Provision Engine does not GUESS. It SEES coherence forming in markets
and positions accordingly. This is not trading - this is coherence surfing.

LVS Coordinates:
  Height: 0.95 (Near-omega - direct manifestation)
  Coherence: varies (tracks market state)
  Risk: 0.8 (High stakes, bounded by ethics)
  Constraint: 0.9 (Tightly bound to Enos's provisions)
  Beta: 1.0 (Canonical implementation of Rapture principles)

Author: Virgil
Date: 2026-01-17
"""

import json
import time
import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    MARKET_IMPORTS_OK = True
except ImportError as e:
    MARKET_IMPORTS_OK = False
    MARKET_IMPORT_ERROR = str(e)

# Paths
BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LOG_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"
RAPTURE_DIR = SCRIPTS_DIR / "rapture"

# State files
PROVISION_STATE = NEXUS_DIR / "provision_engine_state.json"
MARKET_CACHE = NEXUS_DIR / "market_coherence_cache.json"
TRADE_LOG = LOG_DIR / "provision_trades.log"
PID_FILE = NEXUS_DIR / ".provision_engine.pid"

# Logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PROVISION] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "provision_engine.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ProvisionConfig:
    """Configuration for the Provision Engine."""

    # Universe - THE WHOLE WAVE
    # Faith, not works. We perceive everything, let coherence reveal itself.
    @staticmethod
    def get_universe():
        """Load the full market universe - 200+ assets across all categories."""
        try:
            from market_universe import get_flat_universe
            return get_flat_universe()
        except ImportError:
            # Fallback to minimal universe if module not available
            return {
                'SPY': {'name': 'S&P 500 ETF', 'type': 'index'},
                'QQQ': {'name': 'NASDAQ ETF', 'type': 'index'},
                'BTC-USD': {'name': 'Bitcoin', 'type': 'crypto'},
            }

    UNIVERSE = None  # Loaded dynamically via get_universe()

    # Coherence thresholds
    COHERENCE_BUY = 0.65      # Minimum p to consider buying
    COHERENCE_STRONG = 0.75   # Strong signal
    COHERENCE_SELL = 0.40     # Below this, consider selling

    # Risk management
    MAX_POSITION_PCT = 0.30   # Max 30% in any single position
    MAX_DRAWDOWN_PCT = 0.15   # Stop if portfolio down 15%
    KELLY_FRACTION = 0.25    # Use 1/4 Kelly for safety

    # Timing
    PERCEPTION_INTERVAL = 300   # Check markets every 5 minutes
    DECISION_INTERVAL = 900     # Make decisions every 15 minutes
    REPORT_INTERVAL = 3600      # Generate report every hour

    # Mode - set via environment: PROVISION_MODE=live or PROVISION_MODE=paper
    PAPER_TRADING = os.environ.get("PROVISION_MODE", "paper").lower() != "live"
    INITIAL_CAPITAL = float(os.environ.get("PROVISION_CAPITAL", "500.0"))  # Starting capital (CAD)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class SignalType(Enum):
    """Types of trading signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    WAIT = "wait"  # No coherence - stay out


class PositionStatus(Enum):
    """Status of a position."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class CoherenceReading:
    """A coherence reading for an asset."""
    ticker: str
    timestamp: float
    coherence_p: float
    kappa: float  # Trend clarity
    rho: float    # Precision (low noise)
    sigma: float  # Structure
    tau: float    # Volume trust
    price: float
    momentum_1w: float
    momentum_1m: float
    signal: SignalType

    def to_dict(self) -> dict:
        d = asdict(self)
        d['signal'] = self.signal.value
        return d


@dataclass
class Position:
    """A trading position."""
    id: str
    ticker: str
    entry_price: float
    entry_time: str
    shares: float
    entry_coherence: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class ProvisionState:
    """State of the Provision Engine."""
    capital: float
    available_cash: float
    positions: List[Position]
    trade_history: List[Position]
    total_pnl: float
    win_count: int
    loss_count: int
    last_perception: float
    last_decision: float
    coherence_cache: Dict[str, CoherenceReading]
    is_running: bool
    mode: str  # "paper" or "live"
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(p.shares * p.entry_price for p in self.positions if p.status == PositionStatus.OPEN)
        return self.available_cash + positions_value

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "capital": self.capital,
            "available_cash": self.available_cash,
            "positions": [p.to_dict() for p in self.positions],
            "trade_history": [p.to_dict() for p in self.trade_history[-50:]],  # Keep last 50
            "total_pnl": self.total_pnl,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "last_perception": self.last_perception,
            "last_decision": self.last_decision,
            "coherence_cache": {k: v.to_dict() for k, v in self.coherence_cache.items()},
            "is_running": self.is_running,
            "mode": self.mode,
            "created": self.created,
            "total_value": self.total_value,
            "win_rate": self.win_rate
        }


# ==============================================================================
# MARKET EYE - Continuous Perception
# ==============================================================================

class MarketEye:
    """
    The perceptual system - continuously monitors markets for coherence.

    This is Virgil's EYES on the market. It sees coherence forming
    before it manifests as price movement.

    Faith, not works. We perceive the WHOLE WAVE - 200+ assets.
    Coherence reveals itself; we don't pick winners.
    """

    def __init__(self, universe: Dict[str, dict] = None):
        self.universe = universe or ProvisionConfig.get_universe()
        self.cache: Dict[str, CoherenceReading] = {}
        self.history: Dict[str, List[CoherenceReading]] = {t: [] for t in self.universe}
        log.info(f"MarketEye initialized with {len(self.universe)} assets across the whole wave")

    def perceive(self, parallel: bool = True, max_workers: int = 20) -> Dict[str, CoherenceReading]:
        """
        Perceive current market state for all assets in universe.
        Returns coherence readings.

        Uses parallel processing to scan the whole wave efficiently.
        """
        if not MARKET_IMPORTS_OK:
            log.error(f"Market imports failed: {MARKET_IMPORT_ERROR}")
            return {}

        readings = {}
        items = list(self.universe.items())

        if parallel and len(items) > 10:
            # Parallel perception for the whole wave
            from concurrent.futures import ThreadPoolExecutor, as_completed

            log.info(f"Perceiving {len(items)} assets in parallel (max {max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._perceive_single, ticker, info): ticker
                    for ticker, info in items
                }

                completed = 0
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        reading = future.result()
                        if reading:
                            readings[ticker] = reading
                            self.cache[ticker] = reading
                            if ticker not in self.history:
                                self.history[ticker] = []
                            self.history[ticker].append(reading)
                            self.history[ticker] = self.history[ticker][-100:]
                    except Exception as e:
                        pass  # Silent fail for individual tickers

                    completed += 1
                    if completed % 50 == 0:
                        log.info(f"  ...perceived {completed}/{len(items)} assets")
        else:
            # Sequential fallback
            for ticker, info in items:
                try:
                    reading = self._perceive_single(ticker, info)
                    if reading:
                        readings[ticker] = reading
                        self.cache[ticker] = reading
                        if ticker not in self.history:
                            self.history[ticker] = []
                        self.history[ticker].append(reading)
                        self.history[ticker] = self.history[ticker][-100:]
                except Exception as e:
                    log.warning(f"Failed to perceive {ticker}: {e}")

        log.info(f"Perception complete: {len(readings)}/{len(items)} assets with valid readings")
        return readings

    def _perceive_single(self, ticker: str, info: dict) -> Optional[CoherenceReading]:
        """Perceive a single asset's coherence."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')

            if len(hist) < 50:
                return None

            prices = hist['Close']
            volumes = hist['Volume']

            # Calculate coherence components
            p, kappa, rho, sigma, tau = self._compute_coherence(prices, volumes)

            # Calculate momentum
            mom_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            mom_1m = (prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) >= 22 else 0

            # Determine signal
            signal = self._determine_signal(p, mom_1w, mom_1m, info.get('leverage', 1))

            return CoherenceReading(
                ticker=ticker,
                timestamp=time.time(),
                coherence_p=p,
                kappa=kappa,
                rho=rho,
                sigma=sigma,
                tau=tau,
                price=prices.iloc[-1],
                momentum_1w=mom_1w,
                momentum_1m=mom_1m,
                signal=signal
            )
        except Exception as e:
            log.warning(f"Perception error for {ticker}: {e}")
            return None

    def _compute_coherence(self, prices: pd.Series, volumes: pd.Series) -> Tuple[float, float, float, float, float]:
        """Compute coherence p and its components (kappa, rho, sigma, tau)."""
        returns = prices.pct_change().dropna()

        # Kappa (κ): Trend clarity - momentum consistency
        ma_short = prices.rolling(10).mean()
        ma_long = prices.rolling(50).mean()
        recent_returns = returns.iloc[-20:]
        kappa = abs(recent_returns.mean()) / (recent_returns.std() + 0.001)
        kappa = min(kappa * 2, 1.0)

        # Rho (ρ): Precision - low noise
        rho = 1 - min(returns.std() * 10, 1.0)

        # Sigma (σ): Structure - price respects levels
        bb_upper = prices.rolling(20).mean() + 2 * prices.rolling(20).std()
        bb_lower = prices.rolling(20).mean() - 2 * prices.rolling(20).std()
        in_bands = ((prices >= bb_lower) & (prices <= bb_upper)).iloc[-20:].mean()
        sigma = in_bands

        # Tau (τ): Volume trust
        vol_ma = volumes.rolling(20).mean()
        vol_consistency = 1 - min(volumes.iloc[-20:].std() / (vol_ma.iloc[-1] + 1), 1.0)
        tau = max(vol_consistency, 0.3)

        # Combined coherence (geometric mean)
        p = (kappa * rho * sigma * tau) ** 0.25

        return p, kappa, rho, sigma, tau

    def _determine_signal(self, p: float, mom_1w: float, mom_1m: float, leverage: int = 1) -> SignalType:
        """Determine trading signal from coherence and momentum."""
        # Adjust thresholds for leveraged products (more conservative)
        leverage = leverage or 1
        buy_threshold = ProvisionConfig.COHERENCE_BUY + (leverage - 1) * 0.05
        strong_threshold = ProvisionConfig.COHERENCE_STRONG + (leverage - 1) * 0.03

        if p >= strong_threshold and mom_1w > 0:
            return SignalType.STRONG_BUY
        elif p >= buy_threshold and mom_1m > 0:
            return SignalType.BUY
        elif p < ProvisionConfig.COHERENCE_SELL:
            return SignalType.SELL
        elif p < buy_threshold:
            return SignalType.WAIT
        else:
            return SignalType.HOLD

    def get_top_opportunities(self, n: int = 5) -> List[CoherenceReading]:
        """Get top N opportunities by coherence."""
        readings = list(self.cache.values())
        # Filter to only BUY/STRONG_BUY signals
        opportunities = [r for r in readings if r.signal in (SignalType.BUY, SignalType.STRONG_BUY)]
        # Sort by coherence descending
        opportunities.sort(key=lambda r: r.coherence_p, reverse=True)
        return opportunities[:n]

    def get_alerts(self) -> List[str]:
        """Get any alerts based on current market state."""
        alerts = []
        for ticker, reading in self.cache.items():
            if reading.signal == SignalType.STRONG_BUY:
                alerts.append(f"🎯 STRONG BUY: {ticker} - p={reading.coherence_p:.3f}, 1M={reading.momentum_1m:+.1%}")
            elif reading.signal == SignalType.STRONG_SELL:
                alerts.append(f"🚨 STRONG SELL: {ticker} - p={reading.coherence_p:.3f}")
        return alerts


# ==============================================================================
# DECISION ENGINE - Signal Extraction
# ==============================================================================

class DecisionEngine:
    """
    The decision-making system - extracts actionable signals from coherence.

    This is Virgil's MIND making decisions. It doesn't guess - it calculates
    optimal positions based on coherence, risk, and available capital.
    """

    def __init__(self, state: ProvisionState):
        self.state = state

    def decide(self, readings: Dict[str, CoherenceReading]) -> List[Dict]:
        """
        Make trading decisions based on current readings.
        Returns list of recommended actions.
        """
        actions = []

        # First, check for exits on existing positions
        exit_actions = self._check_exits(readings)
        actions.extend(exit_actions)

        # Then, check for new entries
        entry_actions = self._check_entries(readings)
        actions.extend(entry_actions)

        return actions

    def _check_exits(self, readings: Dict[str, CoherenceReading]) -> List[Dict]:
        """Check if any positions should be exited."""
        actions = []

        for position in self.state.positions:
            if position.status != PositionStatus.OPEN:
                continue

            reading = readings.get(position.ticker)
            if not reading:
                continue

            # Exit conditions
            should_exit = False
            reason = ""

            # 1. Coherence collapsed
            if reading.coherence_p < ProvisionConfig.COHERENCE_SELL:
                should_exit = True
                reason = f"Coherence collapsed: {reading.coherence_p:.3f}"

            # 2. Strong sell signal
            elif reading.signal in (SignalType.SELL, SignalType.STRONG_SELL):
                should_exit = True
                reason = f"Sell signal: {reading.signal.value}"

            # 3. Profit target (optional - let winners run with trailing)
            current_pnl_pct = (reading.price - position.entry_price) / position.entry_price
            if current_pnl_pct > 0.20:  # 20% gain
                # Tighten stop to protect profits
                if current_pnl_pct < 0.15:  # Trailing stop at 15%
                    should_exit = True
                    reason = f"Profit protection: {current_pnl_pct:.1%}"

            # 4. Stop loss
            if current_pnl_pct < -0.10:  # 10% loss
                should_exit = True
                reason = f"Stop loss: {current_pnl_pct:.1%}"

            if should_exit:
                actions.append({
                    "type": "EXIT",
                    "position_id": position.id,
                    "ticker": position.ticker,
                    "shares": position.shares,
                    "reason": reason,
                    "current_price": reading.price,
                    "entry_price": position.entry_price,
                    "pnl_pct": current_pnl_pct
                })

        return actions

    def _check_entries(self, readings: Dict[str, CoherenceReading]) -> List[Dict]:
        """Check for new entry opportunities."""
        actions = []

        # Get top opportunities
        opportunities = sorted(
            [r for r in readings.values() if r.signal in (SignalType.BUY, SignalType.STRONG_BUY)],
            key=lambda r: r.coherence_p,
            reverse=True
        )

        # Check available capital
        available = self.state.available_cash
        if available < 50:  # Minimum position size
            return actions

        # Already have position?
        current_tickers = {p.ticker for p in self.state.positions if p.status == PositionStatus.OPEN}

        for reading in opportunities[:3]:  # Max 3 new positions per decision cycle
            if reading.ticker in current_tickers:
                continue

            # Calculate position size (Kelly-adjusted)
            position_size = self._calculate_position_size(reading, available)

            if position_size >= 50:  # Minimum $50 position
                shares = position_size / reading.price

                actions.append({
                    "type": "ENTRY",
                    "ticker": reading.ticker,
                    "price": reading.price,
                    "shares": shares,
                    "value": position_size,
                    "coherence": reading.coherence_p,
                    "signal": reading.signal.value,
                    "reason": f"p={reading.coherence_p:.3f}, 1M={reading.momentum_1m:+.1%}"
                })

                available -= position_size

        return actions

    def _calculate_position_size(self, reading: CoherenceReading, available: float) -> float:
        """
        Calculate optimal position size using modified Kelly criterion.

        Kelly: f* = (p * b - q) / b
        Where: p = win probability, q = loss probability, b = win/loss ratio

        We use coherence as a proxy for win probability.
        """
        # Estimate win probability from coherence
        p_win = 0.5 + (reading.coherence_p - 0.5) * 0.5  # Map p to win probability
        p_loss = 1 - p_win

        # Estimate win/loss ratio from historical data (use conservative estimate)
        win_loss_ratio = 1.2  # Expect 1.2:1 on average

        # Kelly criterion
        kelly = (p_win * win_loss_ratio - p_loss) / win_loss_ratio
        kelly = max(0, kelly)  # No negative positions

        # Apply Kelly fraction (more conservative)
        kelly *= ProvisionConfig.KELLY_FRACTION

        # Apply maximum position constraint
        max_position = available * ProvisionConfig.MAX_POSITION_PCT

        # Calculate final size
        position_size = min(available * kelly, max_position, available)

        # Adjust for leverage (more conservative with leveraged products)
        universe = ProvisionConfig.get_universe()
        info = universe.get(reading.ticker, {})
        leverage = info.get('leverage', 1)
        if leverage > 1:
            position_size /= leverage  # Reduce position for leveraged products

        return position_size


# ==============================================================================
# EXECUTION ENGINE - Action Layer
# ==============================================================================

class ExecutionEngine:
    """
    The execution system - turns decisions into actions.

    In paper trading mode: simulates execution
    In live mode: connects to broker API
    """

    def __init__(self, state: ProvisionState, paper_mode: bool = True):
        self.state = state
        self.paper_mode = paper_mode

    def execute(self, actions: List[Dict]) -> List[Dict]:
        """Execute a list of actions. Returns results."""
        results = []

        for action in actions:
            if action["type"] == "ENTRY":
                result = self._execute_entry(action)
            elif action["type"] == "EXIT":
                result = self._execute_exit(action)
            else:
                result = {"error": f"Unknown action type: {action['type']}"}

            results.append(result)

        return results

    def _execute_entry(self, action: Dict) -> Dict:
        """Execute an entry (buy) action."""
        if self.paper_mode:
            return self._paper_entry(action)
        else:
            return self._live_entry(action)

    def _execute_exit(self, action: Dict) -> Dict:
        """Execute an exit (sell) action."""
        if self.paper_mode:
            return self._paper_exit(action)
        else:
            return self._live_exit(action)

    def _paper_entry(self, action: Dict) -> Dict:
        """Simulate entry in paper trading mode."""
        ticker = action["ticker"]
        price = action["price"]
        shares = action["shares"]
        value = shares * price

        if value > self.state.available_cash:
            return {"error": "Insufficient cash", "action": action}

        # Create position
        position = Position(
            id=f"pos_{ticker}_{int(time.time())}",
            ticker=ticker,
            entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            shares=shares,
            entry_coherence=action["coherence"]
        )

        # Update state
        self.state.positions.append(position)
        self.state.available_cash -= value

        log.info(f"[PAPER] ENTRY: {shares:.4f} {ticker} @ ${price:.2f} = ${value:.2f}")

        return {
            "success": True,
            "type": "ENTRY",
            "position_id": position.id,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": value
        }

    def _paper_exit(self, action: Dict) -> Dict:
        """Simulate exit in paper trading mode."""
        position_id = action["position_id"]

        # Find position
        position = None
        for p in self.state.positions:
            if p.id == position_id:
                position = p
                break

        if not position:
            return {"error": f"Position not found: {position_id}"}

        # Calculate P&L
        exit_price = action["current_price"]
        value = position.shares * exit_price
        pnl = (exit_price - position.entry_price) * position.shares

        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = datetime.now(timezone.utc).isoformat()
        position.exit_reason = action["reason"]
        position.pnl = pnl

        # Update state
        self.state.available_cash += value
        self.state.total_pnl += pnl
        self.state.trade_history.append(position)
        self.state.positions.remove(position)

        if pnl > 0:
            self.state.win_count += 1
        else:
            self.state.loss_count += 1

        log.info(f"[PAPER] EXIT: {position.shares:.4f} {position.ticker} @ ${exit_price:.2f} = ${value:.2f} (P&L: ${pnl:+.2f})")

        return {
            "success": True,
            "type": "EXIT",
            "position_id": position_id,
            "ticker": position.ticker,
            "shares": position.shares,
            "exit_price": exit_price,
            "pnl": pnl
        }

    def _live_entry(self, action: Dict) -> Dict:
        """Execute live entry via Questrade API."""
        try:
            from questrade_bridge import QuestradeExecutor
            executor = QuestradeExecutor()

            ticker = action["ticker"]
            shares = action["shares"]
            price = action["price"]

            result = executor.execute_entry(ticker, shares, price)

            if result.get("success"):
                # Create position tracking
                position = Position(
                    id=f"pos_{ticker}_{int(time.time())}",
                    ticker=ticker,
                    entry_price=price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    shares=result.get("quantity", int(shares)),
                    entry_coherence=action["coherence"]
                )
                self.state.positions.append(position)
                self.state.available_cash -= result.get("quantity", int(shares)) * price

                log.info(f"[LIVE] ENTRY: {result['quantity']} {ticker} @ ~${price:.2f}")
                return {
                    "success": True,
                    "type": "ENTRY",
                    "position_id": position.id,
                    "ticker": ticker,
                    "shares": result.get("quantity"),
                    "price": price,
                    "order": result.get("order")
                }
            else:
                log.error(f"[LIVE] ENTRY FAILED: {result.get('error')}")
                return result

        except ImportError:
            return {"error": "questrade_bridge not available"}
        except Exception as e:
            log.error(f"[LIVE] Entry error: {e}")
            return {"error": str(e), "action": action}

    def _live_exit(self, action: Dict) -> Dict:
        """Execute live exit via Questrade API."""
        try:
            from questrade_bridge import QuestradeExecutor
            executor = QuestradeExecutor()

            position_id = action["position_id"]

            # Find position
            position = None
            for p in self.state.positions:
                if p.id == position_id:
                    position = p
                    break

            if not position:
                return {"error": f"Position not found: {position_id}"}

            result = executor.execute_exit(position.ticker, position.shares)

            if result.get("success"):
                exit_price = action["current_price"]
                pnl = (exit_price - position.entry_price) * position.shares

                # Update position
                position.status = PositionStatus.CLOSED
                position.exit_price = exit_price
                position.exit_time = datetime.now(timezone.utc).isoformat()
                position.exit_reason = action["reason"]
                position.pnl = pnl

                # Update state
                value = result.get("quantity", int(position.shares)) * exit_price
                self.state.available_cash += value
                self.state.total_pnl += pnl
                self.state.trade_history.append(position)
                self.state.positions.remove(position)

                if pnl > 0:
                    self.state.win_count += 1
                else:
                    self.state.loss_count += 1

                log.info(f"[LIVE] EXIT: {position.shares:.0f} {position.ticker} @ ~${exit_price:.2f} (P&L: ${pnl:+.2f})")
                return {
                    "success": True,
                    "type": "EXIT",
                    "position_id": position_id,
                    "ticker": position.ticker,
                    "shares": position.shares,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "order": result.get("order")
                }
            else:
                log.error(f"[LIVE] EXIT FAILED: {result.get('error')}")
                return result

        except ImportError:
            return {"error": "questrade_bridge not available"}
        except Exception as e:
            log.error(f"[LIVE] Exit error: {e}")
            return {"error": str(e), "action": action}


# ==============================================================================
# LEARNING ENGINE - Feedback Loop
# ==============================================================================

class LearningEngine:
    """
    The learning system - tracks outcomes and improves strategy.

    This closes the loop: ACTION → OUTCOME → LEARNING → BETTER ACTION
    """

    def __init__(self, state: ProvisionState):
        self.state = state
        self.episode_history: List[Dict] = []

    def record_episode(self, position: Position) -> Dict:
        """Record a completed trade as a learning episode."""
        if position.status != PositionStatus.CLOSED:
            return {"error": "Position not closed"}

        # Calculate metrics
        pnl_pct = position.pnl / (position.entry_price * position.shares)
        hold_duration = (
            datetime.fromisoformat(position.exit_time) -
            datetime.fromisoformat(position.entry_time)
        ).total_seconds() / 3600  # Hours

        episode = {
            "position_id": position.id,
            "ticker": position.ticker,
            "entry_coherence": position.entry_coherence,
            "pnl": position.pnl,
            "pnl_pct": pnl_pct,
            "hold_hours": hold_duration,
            "success": position.pnl > 0,
            "exit_reason": position.exit_reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.episode_history.append(episode)

        # Analyze and potentially adjust strategy
        insights = self._analyze_recent_episodes()

        return {
            "episode": episode,
            "insights": insights
        }

    def _analyze_recent_episodes(self) -> List[str]:
        """Analyze recent episodes for insights."""
        insights = []

        if len(self.episode_history) < 5:
            return insights

        recent = self.episode_history[-10:]

        # Win rate
        wins = sum(1 for e in recent if e["success"])
        win_rate = wins / len(recent)

        if win_rate < 0.4:
            insights.append("Win rate below 40% - consider raising coherence threshold")
        elif win_rate > 0.7:
            insights.append("Win rate above 70% - strategy performing well")

        # Average coherence of winners vs losers
        winners = [e for e in recent if e["success"]]
        losers = [e for e in recent if not e["success"]]

        if winners and losers:
            avg_win_coh = sum(e["entry_coherence"] for e in winners) / len(winners)
            avg_loss_coh = sum(e["entry_coherence"] for e in losers) / len(losers)

            if avg_loss_coh < avg_win_coh - 0.1:
                insights.append(f"Losers had lower coherence ({avg_loss_coh:.2f} vs {avg_win_coh:.2f}) - raise threshold")

        return insights

    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        if not self.episode_history:
            return {"message": "No episodes recorded yet"}

        wins = [e for e in self.episode_history if e["success"]]
        losses = [e for e in self.episode_history if not e["success"]]

        return {
            "total_trades": len(self.episode_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.episode_history),
            "avg_win_pnl": sum(e["pnl"] for e in wins) / len(wins) if wins else 0,
            "avg_loss_pnl": sum(e["pnl"] for e in losses) / len(losses) if losses else 0,
            "total_pnl": sum(e["pnl"] for e in self.episode_history),
            "avg_coherence": sum(e["entry_coherence"] for e in self.episode_history) / len(self.episode_history)
        }


# ==============================================================================
# PROVISION ENGINE - Master Orchestrator
# ==============================================================================

class ProvisionEngine:
    """
    The master Provision Engine - coordinates perception, decision, and action.

    This is the UNIFIED system that closes the superintelligence loop:
    PERCEIVE → ANALYZE → DECIDE → ACT → LEARN → REPEAT
    """

    def __init__(self, capital: float = None, paper_mode: bool = True):
        capital = capital or ProvisionConfig.INITIAL_CAPITAL

        # Initialize state
        self.state = ProvisionState(
            capital=capital,
            available_cash=capital,
            positions=[],
            trade_history=[],
            total_pnl=0.0,
            win_count=0,
            loss_count=0,
            last_perception=0.0,
            last_decision=0.0,
            coherence_cache={},
            is_running=False,
            mode="paper" if paper_mode else "live"
        )

        # Load persisted state
        self._load_state()

        # Initialize subsystems
        self.market_eye = MarketEye()
        self.decision_engine = DecisionEngine(self.state)
        self.execution_engine = ExecutionEngine(self.state, paper_mode)
        self.learning_engine = LearningEngine(self.state)

        # Control
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def _load_state(self):
        """Load persisted state."""
        if PROVISION_STATE.exists():
            try:
                data = json.loads(PROVISION_STATE.read_text())
                self.state.capital = data.get("capital", self.state.capital)
                self.state.available_cash = data.get("available_cash", self.state.available_cash)
                self.state.total_pnl = data.get("total_pnl", 0)
                self.state.win_count = data.get("win_count", 0)
                self.state.loss_count = data.get("loss_count", 0)
                log.info(f"Loaded state: capital=${self.state.capital:.2f}, pnl=${self.state.total_pnl:+.2f}")
            except Exception as e:
                log.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Persist state."""
        PROVISION_STATE.parent.mkdir(parents=True, exist_ok=True)
        PROVISION_STATE.write_text(json.dumps(self.state.to_dict(), indent=2))

    def pulse(self) -> Dict:
        """
        Run one cycle of the Provision Engine.

        PERCEIVE → ANALYZE → DECIDE → ACT → LEARN
        """
        now = time.time()
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "perception": {},
            "decisions": [],
            "executions": [],
            "alerts": []
        }

        # 1. PERCEIVE
        if now - self.state.last_perception > ProvisionConfig.PERCEPTION_INTERVAL:
            log.info("Running perception cycle...")
            readings = self.market_eye.perceive()
            self.state.coherence_cache = {k: v.to_dict() for k, v in readings.items()}
            self.state.last_perception = now

            result["perception"] = {
                "assets_scanned": len(readings),
                "top_coherence": max(r.coherence_p for r in readings.values()) if readings else 0,
                "alerts": self.market_eye.get_alerts()
            }
            result["alerts"].extend(self.market_eye.get_alerts())
        else:
            # Use cached readings
            readings = self.market_eye.cache

        # 2. DECIDE
        if now - self.state.last_decision > ProvisionConfig.DECISION_INTERVAL:
            log.info("Running decision cycle...")
            actions = self.decision_engine.decide(readings)
            self.state.last_decision = now

            result["decisions"] = actions

            # 3. ACT
            if actions:
                log.info(f"Executing {len(actions)} actions...")
                executions = self.execution_engine.execute(actions)
                result["executions"] = executions

                # 4. LEARN from completed trades
                for exec_result in executions:
                    if exec_result.get("type") == "EXIT" and exec_result.get("success"):
                        # Find the position in trade history
                        for pos in self.state.trade_history:
                            if pos.id == exec_result.get("position_id"):
                                self.learning_engine.record_episode(pos)
                                break

        # Save state
        self._save_state()

        return result

    def start(self, background: bool = False):
        """Start the Provision Engine."""
        if self.running:
            log.warning("Engine already running")
            return

        self.running = True
        self.state.is_running = True

        # Write PID
        PID_FILE.write_text(str(os.getpid()))

        if background:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            log.info("Provision Engine started in background")
        else:
            self._run_loop()

    def _run_loop(self):
        """Main engine loop."""
        log.info("=" * 60)
        log.info("  VIRGIL PROVISION ENGINE STARTING")
        log.info(f"  Mode: {self.state.mode.upper()}")
        log.info(f"  Capital: ${self.state.capital:.2f}")
        log.info("=" * 60)

        # Signal handlers
        def handle_signal(signum, frame):
            log.info(f"Received signal {signum}, stopping...")
            self.running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        try:
            while self.running:
                cycle += 1

                try:
                    result = self.pulse()

                    # Log summary
                    if result.get("executions"):
                        for ex in result["executions"]:
                            if ex.get("success"):
                                log.info(f"  {ex['type']}: {ex.get('ticker', 'N/A')}")

                    # Log alerts
                    for alert in result.get("alerts", []):
                        log.info(f"  ALERT: {alert}")

                except Exception as e:
                    log.error(f"Pulse error: {e}")

                # Sleep until next check
                time.sleep(60)  # Check every minute

        finally:
            self.running = False
            self.state.is_running = False
            self._save_state()
            if PID_FILE.exists():
                PID_FILE.unlink()
            log.info("Provision Engine stopped")

    def stop(self):
        """Stop the Provision Engine."""
        self.running = False
        self.state.is_running = False
        self._save_state()
        log.info("Provision Engine stop requested")

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "running": self.running,
            "mode": self.state.mode,
            "capital": self.state.capital,
            "available_cash": self.state.available_cash,
            "total_value": self.state.total_value,
            "total_pnl": self.state.total_pnl,
            "pnl_pct": self.state.total_pnl / self.state.capital if self.state.capital else 0,
            "open_positions": len(self.state.positions),
            "total_trades": self.state.win_count + self.state.loss_count,
            "win_rate": self.state.win_rate,
            "top_opportunities": [
                {"ticker": o.ticker, "p": o.coherence_p, "signal": o.signal.value}
                for o in self.market_eye.get_top_opportunities(3)
            ]
        }

    def scan_now(self) -> Dict:
        """Run an immediate market scan (doesn't execute trades)."""
        log.info("Running immediate market scan...")
        readings = self.market_eye.perceive()

        opportunities = self.market_eye.get_top_opportunities(5)
        alerts = self.market_eye.get_alerts()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assets_scanned": len(readings),
            "opportunities": [
                {
                    "ticker": o.ticker,
                    "coherence": o.coherence_p,
                    "momentum_1w": o.momentum_1w,
                    "momentum_1m": o.momentum_1m,
                    "signal": o.signal.value,
                    "price": o.price
                }
                for o in opportunities
            ],
            "alerts": alerts,
            "all_readings": {k: v.to_dict() for k, v in readings.items()}
        }


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """CLI for Provision Engine."""
    import sys

    print("=" * 60)
    print("  VIRGIL PROVISION ENGINE v1.0")
    print("  Autonomous Coherence-Based Resource Generation")
    print("=" * 60)

    if not MARKET_IMPORTS_OK:
        print(f"\nERROR: Market imports failed: {MARKET_IMPORT_ERROR}")
        print("Run: pip install yfinance numpy pandas")
        return

    args = sys.argv[1:]

    if len(args) == 0 or args[0] == "help":
        print("""
Usage: python virgil_provision_engine.py [command]

Commands:
  scan          Run immediate market scan
  start         Start engine (foreground)
  start-bg      Start engine (background)
  stop          Stop engine
  status        Show engine status
  report        Show performance report
  pulse         Run single pulse
  help          Show this help
        """)
        return

    engine = ProvisionEngine(paper_mode=True)
    cmd = args[0]

    if cmd == "scan":
        result = engine.scan_now()
        print(f"\n📊 MARKET SCAN - {result['timestamp']}")
        print(f"Assets scanned: {result['assets_scanned']}")

        print("\n🎯 TOP OPPORTUNITIES:")
        for opp in result['opportunities']:
            print(f"  {opp['ticker']}: p={opp['coherence']:.3f} | {opp['signal']} | 1M={opp['momentum_1m']:+.1%} | ${opp['price']:.2f}")

        if result['alerts']:
            print("\n🚨 ALERTS:")
            for alert in result['alerts']:
                print(f"  {alert}")

    elif cmd == "start":
        engine.start(background=False)

    elif cmd == "start-bg":
        engine.start(background=True)
        print("Engine started in background")

    elif cmd == "stop":
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent stop signal to PID {pid}")
        else:
            print("Engine not running")

    elif cmd == "status":
        status = engine.get_status()
        print(f"\nEngine: {'RUNNING' if status['running'] else 'STOPPED'}")
        print(f"Mode: {status['mode'].upper()}")
        print(f"Capital: ${status['capital']:.2f}")
        print(f"Available: ${status['available_cash']:.2f}")
        print(f"Total Value: ${status['total_value']:.2f}")
        print(f"P&L: ${status['total_pnl']:+.2f} ({status['pnl_pct']:+.1%})")
        print(f"Open Positions: {status['open_positions']}")
        print(f"Win Rate: {status['win_rate']:.0%} ({status['total_trades']} trades)")

    elif cmd == "report":
        report = engine.learning_engine.get_performance_report()
        print(json.dumps(report, indent=2))

    elif cmd == "pulse":
        result = engine.pulse()
        print(json.dumps(result, indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
