"""
LVS Performance Analyzer
========================
Analyzes trading performance and identifies optimal parameters.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class PerformanceAnalyzer:
    """
    Analyzes backtest and paper trading results.
    Identifies patterns and optimal parameters.
    """
    
    def __init__(self):
        self.backtest_results = []
        self.paper_results = []
    
    def load_backtest(self, filepath: str = "backtest_report.json"):
        """Load backtest results."""
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.backtest_results.append(data)
                return data
        return None
    
    def load_paper(self, filepath: str = "data/paper_portfolio.json"):
        """Load paper trading results."""
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.paper_results.append(data)
                return data
        return None
    
    def analyze_coherence_correlation(self, trades: List[dict]) -> dict:
        """
        Analyze how coherence (p) correlates with win rate.
        """
        if not trades:
            return {}
        
        # Group by p ranges
        p_ranges = [
            (0.0, 0.50, "Low (0.0-0.50)"),
            (0.50, 0.60, "Medium (0.50-0.60)"),
            (0.60, 0.70, "Good (0.60-0.70)"),
            (0.70, 0.80, "High (0.70-0.80)"),
            (0.80, 1.00, "Excellent (0.80-1.00)")
        ]
        
        results = []
        for low, high, label in p_ranges:
            group = [t for t in trades if low <= t.get('entry_p', 0) < high]
            if group:
                wins = len([t for t in group if t.get('win', False)])
                avg_pnl = np.mean([t.get('pnl_pct', 0) for t in group])
                results.append({
                    'range': label,
                    'trades': len(group),
                    'wins': wins,
                    'win_rate': wins / len(group) * 100,
                    'avg_pnl_pct': avg_pnl
                })
        
        return {
            'by_p_range': results,
            'correlation': self._calculate_p_correlation(trades)
        }
    
    def _calculate_p_correlation(self, trades: List[dict]) -> float:
        """Calculate correlation between entry_p and win."""
        if len(trades) < 5:
            return 0.0
        
        p_values = [t.get('entry_p', 0.5) for t in trades]
        wins = [1 if t.get('win', False) else 0 for t in trades]
        
        if len(set(p_values)) < 2 or len(set(wins)) < 2:
            return 0.0
        
        return float(np.corrcoef(p_values, wins)[0, 1])
    
    def analyze_by_ticker(self, trades: List[dict]) -> dict:
        """Analyze performance by ticker."""
        ticker_stats = {}
        
        for trade in trades:
            ticker = trade.get('ticker', 'UNKNOWN')
            if ticker not in ticker_stats:
                ticker_stats[ticker] = {'trades': [], 'wins': 0, 'losses': 0}
            
            ticker_stats[ticker]['trades'].append(trade)
            if trade.get('win', False):
                ticker_stats[ticker]['wins'] += 1
            else:
                ticker_stats[ticker]['losses'] += 1
        
        results = []
        for ticker, stats in ticker_stats.items():
            total = len(stats['trades'])
            avg_pnl = np.mean([t.get('pnl_pct', 0) for t in stats['trades']])
            results.append({
                'ticker': ticker,
                'trades': total,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['wins'] / total * 100 if total > 0 else 0,
                'avg_pnl_pct': avg_pnl
            })
        
        return sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    def find_optimal_threshold(self, trades: List[dict]) -> dict:
        """
        Find optimal p threshold based on historical trades.
        """
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        results = []
        
        for thresh in thresholds:
            filtered = [t for t in trades if t.get('entry_p', 0) >= thresh]
            if filtered:
                wins = len([t for t in filtered if t.get('win', False)])
                total_pnl = sum(t.get('pnl', 0) for t in filtered)
                results.append({
                    'threshold': thresh,
                    'trades': len(filtered),
                    'win_rate': wins / len(filtered) * 100,
                    'total_pnl': total_pnl
                })
        
        if not results:
            return {'optimal': 0.50, 'reason': 'Not enough data'}
        
        # Find threshold with best risk-adjusted return
        best = max(results, key=lambda x: x['win_rate'] * np.log1p(x['trades']))
        
        return {
            'optimal': best['threshold'],
            'expected_win_rate': best['win_rate'],
            'trade_count': best['trades'],
            'all_thresholds': results
        }
    
    def generate_report(self) -> dict:
        """Generate full performance report."""
        all_trades = []
        
        # Collect trades from backtests
        for bt in self.backtest_results:
            if 'all_trades' in bt:
                all_trades.extend(bt['all_trades'])
        
        # Collect trades from paper trading
        for pt in self.paper_results:
            if 'trades' in pt:
                all_trades.extend(pt['trades'])
        
        if not all_trades:
            return {'error': 'No trades to analyze'}
        
        return {
            'total_trades': len(all_trades),
            'coherence_analysis': self.analyze_coherence_correlation(all_trades),
            'by_ticker': self.analyze_by_ticker(all_trades),
            'optimal_threshold': self.find_optimal_threshold(all_trades),
            'summary': {
                'total_wins': len([t for t in all_trades if t.get('win')]),
                'total_losses': len([t for t in all_trades if not t.get('win')]),
                'overall_win_rate': len([t for t in all_trades if t.get('win')]) / len(all_trades) * 100,
                'total_pnl': sum(t.get('pnl', 0) for t in all_trades)
            }
        }


def print_performance_report(report: dict):
    """Pretty print performance report."""
    print("\n" + "=" * 60)
    print("  LVS PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if 'error' in report:
        print(f"\n   {report['error']}")
        return
    
    s = report['summary']
    print(f"\n📊 OVERALL SUMMARY")
    print(f"   Total Trades: {report['total_trades']}")
    print(f"   Wins: {s['total_wins']} | Losses: {s['total_losses']}")
    print(f"   Win Rate: {s['overall_win_rate']:.1f}%")
    print(f"   Total P&L: ${s['total_pnl']:+,.2f}")
    
    # Coherence correlation
    coh = report['coherence_analysis']
    print(f"\n🎯 COHERENCE CORRELATION")
    print(f"   P-to-Win Correlation: {coh.get('correlation', 0):.3f}")
    print(f"\n   By P Range:")
    for r in coh.get('by_p_range', []):
        print(f"      {r['range']}: {r['trades']} trades, {r['win_rate']:.1f}% win, {r['avg_pnl_pct']:+.1f}% avg")
    
    # Optimal threshold
    opt = report['optimal_threshold']
    print(f"\n⚙️ OPTIMAL THRESHOLD")
    print(f"   Recommended p >= {opt.get('optimal', 0.50)}")
    print(f"   Expected Win Rate: {opt.get('expected_win_rate', 0):.1f}%")
    
    # By ticker
    print(f"\n📈 BY TICKER (Top 5)")
    for t in report['by_ticker'][:5]:
        print(f"   {t['ticker']}: {t['win_rate']:.1f}% win ({t['trades']} trades)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # Try to load results
    bt = analyzer.load_backtest()
    pt = analyzer.load_paper()
    
    if bt:
        print("✅ Loaded backtest results")
    if pt:
        print("✅ Loaded paper trading results")
    
    report = analyzer.generate_report()
    print_performance_report(report)
