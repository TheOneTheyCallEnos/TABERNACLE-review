"""
LVS Paper Trader
================
Forward simulation with fake money, real prices.

Runs continuously, making decisions based on live data.
Tracks all trades in a log for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import time
from pathlib import Path

from core.coherence import compute_coherence
from core.returns import compute_expected_return
from core.risk import compute_archon_deviation, check_abaddon_triggers


DATA_FILE = Path("data/paper_portfolio.json")


class PaperTrader:
    """
    Simulated trading with real market data.
    """
    
    def __init__(self, capital: float = 1000,
                 p_threshold: float = 0.50,
                 max_positions: int = 3,
                 hold_days: int = 5):
        
        self.initial_capital = capital
        self.p_threshold = p_threshold
        self.max_positions = max_positions
        self.hold_days = hold_days
        
        # Load or initialize state
        self.state = self._load_state()
        
        if not self.state:
            self.state = {
                'capital': capital,
                'initial_capital': capital,
                'positions': {},
                'trades': [],
                'daily_snapshots': [],
                'created': datetime.now().isoformat(),
                'last_run': None
            }
            self._save_state()
    
    def _load_state(self) -> dict:
        """Load state from file."""
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Save state to file."""
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def fetch_current_data(self, ticker: str) -> tuple:
        """Fetch latest price and volume data."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            if len(hist) < 50:
                return None, None, None
            current_price = hist['Close'].iloc[-1]
            return hist['Close'], hist['Volume'], current_price
        except:
            return None, None, None
    
    def analyze(self, ticker: str) -> dict:
        """Analyze a single ticker."""
        prices, volumes, current_price = self.fetch_current_data(ticker)
        
        if prices is None:
            return None
        
        coherence = compute_coherence(prices, volumes)
        archon = compute_archon_deviation(prices, volumes)
        exp_ret = compute_expected_return(prices, volumes, coherence['p'], archon['deviation'])
        
        ma200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
        path = 'ASCENDING' if current_price >= ma200 else 'DESCENDING'
        
        signal = 'BUY' if coherence['p'] >= self.p_threshold and exp_ret['expected_return'] > 0 else 'AVOID'
        
        return {
            'ticker': ticker,
            'price': current_price,
            'p': coherence['p'],
            'expected_return': exp_ret['expected_return'],
            'path': path,
            'archon': archon['deviation'],
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_cycle(self, tickers: list) -> dict:
        """
        Run one trading cycle:
        1. Check for exits (hold period or ABADDON)
        2. Analyze all tickers
        3. Enter new positions
        4. Record snapshot
        """
        print(f"\n🔄 PAPER TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("-" * 50)
        
        # 1. Check exits
        self._check_exits(tickers)
        
        # 2. Analyze all tickers
        print("\nAnalyzing...")
        signals = []
        for ticker in tickers:
            analysis = self.analyze(ticker)
            if analysis:
                signals.append(analysis)
                print(f"   {ticker}: p={analysis['p']:.2f}, E[r]={analysis['expected_return']:.1%}, {analysis['signal']}")
        
        # 3. Enter new positions
        buy_signals = [s for s in signals if s['signal'] == 'BUY' and s['ticker'] not in self.state['positions']]
        buy_signals.sort(key=lambda x: x['expected_return'], reverse=True)
        
        for sig in buy_signals:
            if len(self.state['positions']) >= self.max_positions:
                break
            if self.state['capital'] < 50:
                break
            
            # Allocate
            allocation = self.state['capital'] / (self.max_positions - len(self.state['positions']))
            allocation = min(allocation, self.state['capital'])
            
            shares = allocation / sig['price']
            
            self.state['positions'][sig['ticker']] = {
                'shares': shares,
                'entry_price': sig['price'],
                'entry_date': datetime.now().isoformat(),
                'entry_p': sig['p'],
                'entry_er': sig['expected_return']
            }
            
            self.state['capital'] -= allocation
            
            print(f"\n   📈 BOUGHT {sig['ticker']}: {shares:.2f} shares @ ${sig['price']:.2f}")
        
        # 4. Record snapshot
        portfolio_value = self._calculate_value(signals)
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'value': portfolio_value,
            'cash': self.state['capital'],
            'positions': len(self.state['positions']),
            'return_pct': (portfolio_value / self.state['initial_capital'] - 1) * 100
        }
        
        self.state['daily_snapshots'].append(snapshot)
        self.state['last_run'] = datetime.now().isoformat()
        
        self._save_state()
        
        # Print summary
        print(f"\n💼 PORTFOLIO STATUS")
        print(f"   Value: ${portfolio_value:,.2f}")
        print(f"   Cash: ${self.state['capital']:,.2f}")
        print(f"   Return: {snapshot['return_pct']:+.1f}%")
        print(f"   Positions: {len(self.state['positions'])}")
        
        if self.state['positions']:
            print(f"\n   Holdings:")
            for ticker, pos in self.state['positions'].items():
                current = next((s['price'] for s in signals if s['ticker'] == ticker), pos['entry_price'])
                pnl_pct = (current / pos['entry_price'] - 1) * 100
                print(f"      {ticker}: {pos['shares']:.2f} @ ${pos['entry_price']:.2f} → ${current:.2f} ({pnl_pct:+.1f}%)")
        
        return snapshot
    
    def _check_exits(self, tickers: list):
        """Check if any positions should be exited."""
        to_exit = []
        
        for ticker, pos in self.state['positions'].items():
            entry_date = datetime.fromisoformat(pos['entry_date'])
            days_held = (datetime.now() - entry_date).days
            
            # Exit if hold period reached
            if days_held >= self.hold_days:
                to_exit.append((ticker, 'hold_period'))
                continue
            
            # Check ABADDON
            prices, volumes, current_price = self.fetch_current_data(ticker)
            if prices is not None:
                coherence = compute_coherence(prices, volumes)
                archon = compute_archon_deviation(prices, volumes)
                
                if coherence['p'] < 0.50 or archon['deviation'] >= 0.15:
                    to_exit.append((ticker, 'abaddon'))
        
        for ticker, reason in to_exit:
            self._exit_position(ticker, reason)
    
    def _exit_position(self, ticker: str, reason: str):
        """Exit a position."""
        if ticker not in self.state['positions']:
            return
        
        pos = self.state['positions'][ticker]
        _, _, current_price = self.fetch_current_data(ticker)
        
        if current_price is None:
            current_price = pos['entry_price']
        
        proceeds = pos['shares'] * current_price
        pnl = proceeds - (pos['shares'] * pos['entry_price'])
        pnl_pct = (current_price / pos['entry_price'] - 1) * 100
        
        trade = {
            'ticker': ticker,
            'entry_date': pos['entry_date'],
            'exit_date': datetime.now().isoformat(),
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_p': pos['entry_p'],
            'reason': reason,
            'win': pnl > 0
        }
        
        self.state['trades'].append(trade)
        self.state['capital'] += proceeds
        del self.state['positions'][ticker]
        
        icon = "✅" if pnl > 0 else "❌"
        print(f"\n   {icon} SOLD {ticker}: ${pnl:+.2f} ({pnl_pct:+.1f}%) - {reason}")
    
    def _calculate_value(self, signals: list) -> float:
        """Calculate total portfolio value."""
        value = self.state['capital']
        
        for ticker, pos in self.state['positions'].items():
            current = next((s['price'] for s in signals if s['ticker'] == ticker), pos['entry_price'])
            value += pos['shares'] * current
        
        return value
    
    def get_performance(self) -> dict:
        """Get performance statistics."""
        trades = self.state['trades']
        
        if not trades:
            return {'message': 'No trades yet'}
        
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100,
            'total_pnl': sum(t['pnl'] for t in trades),
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
            'current_value': self.state['daily_snapshots'][-1]['value'] if self.state['daily_snapshots'] else self.state['capital'],
            'return_pct': (self.state['daily_snapshots'][-1]['value'] / self.state['initial_capital'] - 1) * 100 if self.state['daily_snapshots'] else 0
        }
    
    def reset(self, capital: float = 1000):
        """Reset paper trading."""
        self.state = {
            'capital': capital,
            'initial_capital': capital,
            'positions': {},
            'trades': [],
            'daily_snapshots': [],
            'created': datetime.now().isoformat(),
            'last_run': None
        }
        self._save_state()
        print(f"✅ Paper portfolio reset to ${capital:,.2f}")


def print_performance(perf: dict):
    """Print performance stats."""
    print("\n" + "=" * 50)
    print("  PAPER TRADING PERFORMANCE")
    print("=" * 50)
    
    if 'message' in perf:
        print(f"\n   {perf['message']}")
        return
    
    print(f"\n   Total Trades: {perf['total_trades']}")
    print(f"   Wins: {perf['wins']} | Losses: {perf['losses']}")
    print(f"   Win Rate: {perf['win_rate']:.1f}%")
    print(f"\n   Total P&L: ${perf['total_pnl']:+,.2f}")
    print(f"   Avg Win: {perf['avg_win']:+.1f}%")
    print(f"   Avg Loss: {perf['avg_loss']:+.1f}%")
    print(f"\n   Current Value: ${perf['current_value']:,.2f}")
    print(f"   Total Return: {perf['return_pct']:+.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1000, help='Starting capital')
    parser.add_argument('--reset', action='store_true', help='Reset paper portfolio')
    parser.add_argument('--status', action='store_true', help='Show status only')
    args = parser.parse_args()
    
    TICKERS = [
        'SOXL', 'TQQQ', 'UPRO', 'HQU.TO', 'HXU.TO',
        'MSTR', 'COIN', 'TSLA', 'NVDA', 'GME',
        'PLTR', 'MARA', 'RIOT'
    ]
    
    pt = PaperTrader(capital=args.capital)
    
    if args.reset:
        pt.reset(args.capital)
    elif args.status:
        perf = pt.get_performance()
        print_performance(perf)
    else:
        pt.run_cycle(TICKERS)
        perf = pt.get_performance()
        print_performance(perf)
