"""
LVS Backtester v7.1 - FIXED
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from core.coherence import compute_coherence
from core.returns import compute_expected_return
from core.risk import compute_archon_deviation


class Backtester:
    def __init__(self, capital: float = 1000, hold_days: int = 5,
                 p_threshold: float = 0.30, max_positions: int = 3):
        self.initial_capital = capital
        self.capital = capital
        self.hold_days = hold_days
        self.p_threshold = p_threshold
        self.max_positions = max_positions
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.signals_log = []
        
    def fetch_historical(self, tickers: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        data = {}
        end = datetime.now()
        start = end - timedelta(days=days + 200)
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if len(df) > 200:
                    data[ticker] = df
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        return data
    
    def analyze_at_date(self, ticker: str, data: pd.DataFrame, end_idx: int) -> Optional[dict]:
        if end_idx < 200:
            return None
        historical = data.iloc[:end_idx+1]
        prices = historical['Close']
        volumes = historical['Volume']
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        if isinstance(volumes, pd.DataFrame):
            volumes = volumes.iloc[:, 0]
        try:
            coherence = compute_coherence(prices, volumes)
            archon = compute_archon_deviation(prices, volumes)
            exp_ret = compute_expected_return(prices, volumes, coherence['p'], archon['deviation'])
            ma200 = float(prices.iloc[-200:].mean())
            current = float(prices.iloc[-1])
            path = 'ASCENDING' if current >= ma200 else 'DESCENDING'
            return {
                'ticker': ticker, 'date': historical.index[-1], 'price': current,
                'p': coherence['p'], 'expected_return': exp_ret['expected_return'],
                'path': path, 'archon': archon['deviation'],
                'signal': 'BUY' if coherence['p'] >= self.p_threshold and exp_ret['expected_return'] > 0 else 'AVOID'
            }
        except:
            return None
    
    def run_backtest(self, tickers: List[str], test_days: int = 90) -> dict:
        print(f"\n📊 BACKTESTING: {test_days} days, ${self.initial_capital} capital")
        print(f"   Hold period: {self.hold_days} days")
        print(f"   P threshold: {self.p_threshold}")
        print("-" * 50)
        print("\nFetching historical data...")
        all_data = self.fetch_historical(tickers, days=test_days + 250)
        if not all_data:
            return {'error': 'No data fetched'}
        first_ticker = list(all_data.keys())[0]
        dates = all_data[first_ticker].index
        start_idx = 200
        end_idx = len(dates) - 1
        print(f"Testing from {dates[start_idx].strftime('%Y-%m-%d')} to {dates[end_idx].strftime('%Y-%m-%d')}")
        print("-" * 50)
        for day_idx in range(start_idx, end_idx + 1):
            self._check_exits(all_data, day_idx)
            signals = []
            for ticker, data in all_data.items():
                if day_idx < len(data):
                    analysis = self.analyze_at_date(ticker, data, day_idx)
                    if analysis:
                        signals.append(analysis)
                        self.signals_log.append(analysis)
            buy_signals = [s for s in signals if s['signal'] == 'BUY' and s['ticker'] not in self.positions]
            buy_signals.sort(key=lambda x: x['expected_return'], reverse=True)
            for sig in buy_signals:
                if len(self.positions) >= self.max_positions:
                    break
                if self.capital < 50:
                    break
                allocation = self.capital / (self.max_positions - len(self.positions))
                allocation = min(allocation, self.capital)
                shares = allocation / sig['price']
                self.positions[sig['ticker']] = {
                    'shares': shares, 'entry_price': sig['price'], 'entry_date': dates[day_idx],
                    'entry_idx': day_idx, 'entry_p': sig['p'], 'entry_er': sig['expected_return']
                }
                self.capital -= allocation
            portfolio_value = self._calculate_portfolio_value(all_data, day_idx)
            self.daily_values.append({'date': dates[day_idx], 'value': portfolio_value, 'positions': len(self.positions), 'cash': self.capital})
        self._close_all_positions(all_data, end_idx)
        return self._generate_report()
    
    def _check_exits(self, all_data, current_idx):
        to_exit = []
        for ticker, pos in self.positions.items():
            if current_idx - pos['entry_idx'] >= self.hold_days:
                to_exit.append(ticker)
        for ticker in to_exit:
            self._exit_position(ticker, all_data, current_idx)
    
    def _exit_position(self, ticker, all_data, current_idx):
        if ticker not in self.positions:
            return
        pos = self.positions[ticker]
        data = all_data.get(ticker)
        if data is None or current_idx >= len(data):
            return
        exit_price = data.iloc[current_idx]['Close']
        if isinstance(exit_price, pd.Series):
            exit_price = float(exit_price.iloc[0])
        else:
            exit_price = float(exit_price)
        proceeds = pos['shares'] * exit_price
        pnl = proceeds - (pos['shares'] * pos['entry_price'])
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
        self.trades.append({
            'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': data.index[current_idx],
            'entry_price': pos['entry_price'], 'exit_price': exit_price, 'shares': pos['shares'],
            'pnl': pnl, 'pnl_pct': pnl_pct, 'entry_p': pos['entry_p'], 'entry_er': pos['entry_er'], 'win': pnl > 0
        })
        self.capital += proceeds
        del self.positions[ticker]
    
    def _close_all_positions(self, all_data, current_idx):
        for ticker in list(self.positions.keys()):
            self._exit_position(ticker, all_data, current_idx)
    
    def _calculate_portfolio_value(self, all_data, current_idx):
        value = self.capital
        for ticker, pos in self.positions.items():
            data = all_data.get(ticker)
            if data is not None and current_idx < len(data):
                price = data.iloc[current_idx]['Close']
                if isinstance(price, pd.Series):
                    price = float(price.iloc[0])
                value += pos['shares'] * float(price)
        return value
    
    def _generate_report(self):
        if not self.trades:
            p_vals = [s['p'] for s in self.signals_log] if self.signals_log else [0]
            return {'error': 'No trades', 'summary': {'initial_capital': self.initial_capital, 'final_value': self.initial_capital, 'total_return_pct': 0, 'total_pnl': 0},
                    'trades': {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0},
                    'debug': {'signals': len(self.signals_log), 'avg_p': np.mean(p_vals), 'max_p': max(p_vals), 'min_p': min(p_vals)}}
        wins = [t for t in self.trades if t['win']]
        losses = [t for t in self.trades if not t['win']]
        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        return {
            'summary': {'initial_capital': self.initial_capital, 'final_value': final_value,
                       'total_return_pct': (final_value / self.initial_capital - 1) * 100, 'total_pnl': sum(t['pnl'] for t in self.trades)},
            'trades': {'total': len(self.trades), 'wins': len(wins), 'losses': len(losses),
                      'win_rate': len(wins) / len(self.trades) * 100,
                      'avg_win_pct': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
                      'avg_loss_pct': np.mean([t['pnl_pct'] for t in losses]) if losses else 0},
            'coherence': {'high_p_trades': len([t for t in self.trades if t['entry_p'] >= 0.6]),
                         'high_p_winrate': len([t for t in self.trades if t['entry_p'] >= 0.6 and t['win']]) / max(1, len([t for t in self.trades if t['entry_p'] >= 0.6])) * 100},
            'all_trades': self.trades, 'daily_values': self.daily_values
        }


def print_report(report):
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    if report.get('error'):
        print(f"\n⚠️  {report['error']}")
        if 'debug' in report:
            d = report['debug']
            print(f"   Signals analyzed: {d['signals']}")
            print(f"   P range: {d['min_p']:.3f} - {d['max_p']:.3f}")
            print(f"   Try: --threshold {d['min_p'] - 0.05:.2f}")
        return
    s = report['summary']
    print(f"\n💰 PERFORMANCE")
    print(f"   Initial: ${s['initial_capital']:,.2f}")
    print(f"   Final:   ${s['final_value']:,.2f}")
    print(f"   Return:  {s['total_return_pct']:+.1f}%")
    t = report['trades']
    print(f"\n📊 TRADES: {t['total']} ({t['wins']}W / {t['losses']}L = {t['win_rate']:.0f}%)")
    print(f"   Avg Win: {t['avg_win_pct']:+.1f}%  Avg Loss: {t['avg_loss_pct']:+.1f}%")
    if report.get('all_trades'):
        print(f"\n📝 LAST 10 TRADES:")
        for tr in report['all_trades'][-10:]:
            icon = "✅" if tr['win'] else "❌"
            print(f"   {icon} {tr['ticker']}: {tr['pnl_pct']:+.1f}% (p={tr['entry_p']:.2f})")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--capital', type=float, default=1000)
    parser.add_argument('--hold', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.30)
    args = parser.parse_args()
    TICKERS = ['SOXL', 'TQQQ', 'UPRO', 'HQU.TO', 'HXU.TO', 'MSTR', 'COIN', 'TSLA', 'NVDA', 'GME', 'PLTR', 'MARA', 'RIOT']
    bt = Backtester(capital=args.capital, hold_days=args.hold, p_threshold=args.threshold)
    report = bt.run_backtest(TICKERS, test_days=args.days)
    print_report(report)
    with open('backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
