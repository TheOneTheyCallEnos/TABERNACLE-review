#!/usr/bin/env python3
"""
LVS Market Navigator v7.1
=========================
CLI Interface with expanded volatile stock universe
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

import yfinance as yf

from core.navigator import LVSNavigator
from core.coherence import compute_coherence, compute_coherence_history, compute_dp_dt
from core.returns import compute_expected_return
from core.risk import compute_archon_deviation, check_abaddon_triggers
from storage.portfolio_manager import PortfolioManager
from notifications.push import send_notification


# === STOCK UNIVERSE (WEALTHSIMPLE CANADA) ===

# Blue chips (stable)
STOCKS_BLUECHIP = [
    'TD.TO', 'RY.TO', 'BMO.TO', 'BNS.TO', 'CM.TO',  # Banks
    'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO',           # Energy
    'CNR.TO', 'CP.TO',                                # Rail
    'AAPL', 'MSFT', 'GOOGL',                          # US Tech
]

# HIGH VOLATILITY - Big moves potential
STOCKS_VOLATILE = [
    # Leveraged ETFs (2x/3x moves)
    'HQU.TO',   # 2x NASDAQ Canada
    'HXU.TO',   # 2x TSX 60
    'HOU.TO',   # 2x Crude Oil
    'TQQQ',     # 3x NASDAQ US
    'SOXL',     # 3x Semiconductors
    'UPRO',     # 3x S&P 500
    
    # Crypto-linked (massive swings)
    'MSTR',     # MicroStrategy (Bitcoin proxy)
    'MARA',     # Marathon Digital (BTC mining)
    'COIN',     # Coinbase
    'RIOT',     # Riot Platforms
    'HUT.TO',   # Hut 8 Mining (Canadian)
    'BITF.TO',  # Bitfarms (Canadian)
    'GLXY.TO',  # Galaxy Digital (Canadian)
    
    # Meme / High Beta
    'GME',      # GameStop
    'AMC',      # AMC
    'TSLA',     # Tesla
    'NVDA',     # NVIDIA
    'PLTR',     # Palantir
    
    # Canadian Speculative
    'SHOP.TO',  # Shopify
    'BB.TO',    # BlackBerry
    'WEED.TO',  # Canopy Growth
    'ACB.TO',   # Aurora Cannabis
]

# Combined universe
STOCKS_ALL = STOCKS_BLUECHIP + STOCKS_VOLATILE

# Choose your mode
SCAN_MODE = 'VOLATILE'  # Options: 'BLUECHIP', 'VOLATILE', 'ALL'

def get_stock_list():
    if SCAN_MODE == 'BLUECHIP':
        return STOCKS_BLUECHIP
    elif SCAN_MODE == 'VOLATILE':
        return STOCKS_VOLATILE
    else:
        return STOCKS_ALL


def print_header():
    print("\n" + "=" * 60)
    print("  LVS MARKET NAVIGATOR v7.1")
    print("  Teleological Control System for Portfolio Navigation")
    print("  MODE: " + SCAN_MODE)
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")


def fetch_stock_data(ticker: str, period='1y') -> tuple:
    """Fetch price and volume data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if len(hist) < 50:
            return None, None
        return hist['Close'], hist['Volume']
    except Exception as e:
        return None, None


def run_daily_scan():
    """Run the daily market scan."""
    print_header()
    print("\n🔍 DAILY MARKET SCAN")
    print("-" * 60 + "\n")
    
    pm = PortfolioManager()
    account = pm.get_account()
    
    capital = account.get('capital', 1000)
    goal = account.get('goal', 100)
    timeframe = account.get('timeframe_days', 30)
    
    nav = LVSNavigator(capital, goal, timeframe)
    
    stocks = get_stock_list()
    
    print("Fetching market data...")
    for ticker in stocks:
        print(f"  {ticker}...", end=" ")
        prices, volumes = fetch_stock_data(ticker)
        if prices is not None:
            nav.add_stock_data(ticker, prices, volumes)
            print("✓")
        else:
            print("✗ (insufficient data)")
    
    print("\nAnalyzing coherence patterns...")
    
    report = nav.generate_full_report()
    
    # Achievability
    ach = report['achievability']
    print(f"\n📊 ACHIEVABILITY ANALYSIS")
    print(f"   Market Coherence: {ach.get('market_p', 0):.2f}")
    print(f"   Your Goal: ${goal:.2f} ({goal/capital*100:.1f}%)")
    print(f"   Safe Target: ${ach.get('safe_target', 0):.2f} ({ach.get('safe_return_pct', 0):.1f}%)")
    print(f"   Max Target: ${ach.get('max_target', 0):.2f} ({ach.get('max_return_pct', 0):.1f}%)")
    print(f"   Recommendation: {ach.get('recommendation', 'N/A')}")
    
    # Portfolio allocation
    port = report['portfolio']
    if port.get('success'):
        print(f"\n💼 OPTIMAL ALLOCATION (Market p={ach.get('market_p', 0):.2f})")
        for ticker, weight in sorted(port['weights'].items(), key=lambda x: -x[1]):
            if weight > 0.001:
                print(f"   {ticker}: {weight*100:.1f}%")
        print(f"\n   Expected Return: {port.get('expected_return', 0)*100:.1f}%")
        print(f"   Expected Profit: ${port.get('expected_profit', 0):.2f}")
    
    # Portfolio Telos
    telos = report.get('telos', {})
    if telos:
        print(f"\n🎯 PORTFOLIO TELOS")
        print(f"   Path: {telos.get('path', 'N/A')} ({telos.get('path_safety', 'N/A')})")
        print(f"   Current → Telos: ${telos.get('portfolio_current', 0):,.2f} → ${telos.get('portfolio_telos', 0):,.2f}")
    
    # Top opportunities
    analyses = report.get('stock_analyses', [])
    if analyses:
        df = pd.DataFrame(analyses)
        
        # Filter for BUY signals, no ABADDON
        buys = df[(df['signal'] == 'BUY') & (df['abaddon'] == False)]
        buys = buys.sort_values('expected_return', ascending=False).head(10)
        
        print(f"\n🏆 TOP OPPORTUNITIES")
        print(f"{'Ticker':<10} {'p':<8} {'E[r]':<10} {'Signal':<12} {'Path':<12}")
        print("-" * 55)
        
        for _, row in buys.iterrows():
            print(f"{row['ticker']:<10} {row['p']:.3f}    {row['expected_return']:+.1%}     {row['signal']:<12} {row['path']:<12}")
    
    # Alerts
    alerts = report.get('alerts', [])
    critical = [a for a in alerts if a['severity'] == 'CRITICAL']
    
    if critical:
        print(f"\n🚨 CRITICAL ALERTS ({len(critical)})")
        for alert in critical:
            print(f"   {alert['type']}: {alert['message']}")
            send_notification(
                f"ABADDON: {alert['ticker']}",
                alert['message'],
                priority='urgent'
            )
    
    # Opportunities
    opps = [a for a in alerts if a['type'] == 'OPPORTUNITY']
    if opps:
        print(f"\n🚀 OPPORTUNITIES ({len(opps)})")
        for alert in opps:
            print(f"   {alert['message']}")
            send_notification(
                f"Opportunity: {alert['ticker']}",
                alert['message'],
                priority='high'
            )
    
    print("\n" + "=" * 60)
    print("  Scan complete. Run 'python run.py analyze TICKER' for details.")
    print("=" * 60 + "\n")


def run_analysis(ticker: str):
    """Run deep analysis on a single ticker."""
    print_header()
    print(f"\n🔬 DEEP ANALYSIS: {ticker}")
    print("-" * 60 + "\n")
    
    prices, volumes = fetch_stock_data(ticker)
    if prices is None:
        print(f"❌ Could not fetch data for {ticker}")
        return
    
    # Coherence
    coh = compute_coherence(prices, volumes)
    coh_hist = compute_coherence_history(prices, volumes, lookback=30)
    dp_dt = compute_dp_dt(coh_hist)
    
    # Archon
    archon = compute_archon_deviation(prices, volumes)
    
    # Expected return
    exp_ret = compute_expected_return(prices, volumes, coh['p'], archon['deviation'])
    
    # ABADDON check
    abaddon = check_abaddon_triggers(coh['p'], dp_dt, archon['deviation'])
    
    # Price info
    current = prices.iloc[-1]
    ma200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
    path = 'ASCENDING' if current >= ma200 else 'DESCENDING'
    
    print(f"📈 Price: ${current:.2f} (MA200: ${ma200:.2f})")
    print(f"   Path: {path}")
    
    print(f"\n🎯 COHERENCE: {coh['p']:.3f}")
    print(f"   κ (Clarity):   {coh['kappa']:.3f}")
    print(f"   ρ (Precision): {coh['rho']:.3f}")
    print(f"   σ (Structure): {coh['sigma']:.3f}")
    print(f"   τ (Trust):     {coh['tau']:.3f}")
    
    print(f"\n📊 EXPECTED RETURN")
    print(f"   Velocity:    {exp_ret['velocity']*100:.1f}%")
    print(f"   Potential Q: {exp_ret['potential_Q']:.2f}")
    print(f"   E[r]:        {exp_ret['expected_return']*100:.1f}%")
    
    print(f"\n⚠️ ARCHON DEVIATION: {archon['deviation']:.3f}")
    print(f"   Dominant: {archon['dominant_archon']}")
    print(f"   Alert: {'🚨 YES' if archon['alert'] else '✅ No'}")
    
    # Signal
    if abaddon['triggered']:
        signal = "SELL (CRITICAL)"
        reason = ", ".join(abaddon['reasons'])
    elif coh['p'] >= 0.75 and path == 'ASCENDING':
        signal = "BUY (STRONG)"
        reason = f"High coherence ({coh['p']:.2f}), ascending path"
    elif coh['p'] >= 0.60:
        signal = "BUY (MODERATE)"
        reason = f"Good coherence ({coh['p']:.2f})"
    elif coh['p'] >= 0.50:
        signal = "BUY (WEAK)"
        reason = f"Moderate coherence ({coh['p']:.2f}), monitor closely"
    else:
        signal = "AVOID"
        reason = f"Low coherence ({coh['p']:.2f})"
    
    print(f"\n🚦 SIGNAL: {signal}")
    print(f"   Reason: {reason}")
    
    if abaddon['triggered']:
        print(f"\n🚨 ABADDON TRIGGERED: {', '.join(abaddon['reasons'])}")


def run_setup():
    """Configure account settings."""
    print_header()
    print("\n📋 ACCOUNT SETUP")
    print("-" * 40)
    
    pm = PortfolioManager()
    
    try:
        capital = float(input("Enter your starting capital ($): "))
        goal = float(input("Enter your profit goal ($): "))
        timeframe = int(input("Enter timeframe (days) [30]: ") or "30")
        
        pm.update_account(capital=capital, goal=goal, timeframe_days=timeframe)
        
        print(f"\n✅ Account configured!")
        print(f"   Capital: ${capital:,.2f}")
        print(f"   Goal: ${goal:,.2f} ({goal/capital*100:.1f}%)")
        print(f"   Timeframe: {timeframe} days")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")


def run_portfolio():
    """Display current portfolio."""
    print_header()
    print("\n💼 PORTFOLIO")
    print("-" * 40)
    
    pm = PortfolioManager()
    account = pm.get_account()
    holdings = pm.get_holdings()
    
    print(f"\nAccount:")
    print(f"   Capital: ${account.get('capital', 0):,.2f}")
    print(f"   Goal: ${account.get('goal', 0):,.2f}")
    print(f"   Timeframe: {account.get('timeframe_days', 30)} days")
    
    if holdings:
        print(f"\nHoldings:")
        total_value = 0
        for h in holdings:
            ticker = h['ticker']
            shares = h['shares']
            avg_price = h['avg_price']
            
            # Get current price
            prices, _ = fetch_stock_data(ticker, period='5d')
            current = prices.iloc[-1] if prices is not None else avg_price
            
            value = shares * current
            pnl = (current - avg_price) * shares
            pnl_pct = (current / avg_price - 1) * 100
            
            total_value += value
            
            print(f"   {ticker}: {shares} shares @ ${avg_price:.2f}")
            print(f"      Current: ${current:.2f} | Value: ${value:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        print(f"\n   Total Portfolio Value: ${total_value:,.2f}")
    else:
        print("\n   No holdings yet.")


def run_buy(ticker: str, shares: int):
    """Record a buy transaction."""
    pm = PortfolioManager()
    
    prices, _ = fetch_stock_data(ticker, period='5d')
    if prices is None:
        print(f"❌ Could not fetch price for {ticker}")
        return
    
    price = prices.iloc[-1]
    
    pm.add_transaction('BUY', ticker, shares, price)
    
    print(f"✅ Recorded: BUY {shares} {ticker} @ ${price:.2f}")
    print(f"   Total: ${shares * price:.2f}")


def run_sell(ticker: str, shares: int):
    """Record a sell transaction."""
    pm = PortfolioManager()
    
    prices, _ = fetch_stock_data(ticker, period='5d')
    if prices is None:
        print(f"❌ Could not fetch price for {ticker}")
        return
    
    price = prices.iloc[-1]
    
    pm.add_transaction('SELL', ticker, shares, price)
    
    print(f"✅ Recorded: SELL {shares} {ticker} @ ${price:.2f}")
    print(f"   Total: ${shares * price:.2f}")


def run_alerts():
    """Show recent alerts."""
    pm = PortfolioManager()
    alerts = pm.get_alerts(limit=20)
    
    print_header()
    print("\n🔔 RECENT ALERTS")
    print("-" * 40)
    
    if alerts:
        for alert in alerts:
            print(f"\n[{alert.get('timestamp', 'N/A')}]")
            print(f"   {alert.get('type', 'N/A')}: {alert.get('message', 'N/A')}")
    else:
        print("\n   No recent alerts.")


def print_usage():
    print("""
Usage: python run.py [command] [args]

Commands:
  (none)           Run daily scan
  setup            Configure account (capital, goal, timeframe)
  portfolio        View current holdings
  buy TICKER N     Record purchase of N shares
  sell TICKER N    Record sale of N shares
  analyze TICKER   Deep analysis of single stock
  alerts           View recent alerts
  
Examples:
  python run.py
  python run.py setup
  python run.py analyze MSTR
  python run.py buy TSLA 5
""")


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if len(args) == 0:
        run_daily_scan()
    elif args[0] == 'setup':
        run_setup()
    elif args[0] == 'portfolio':
        run_portfolio()
    elif args[0] == 'analyze' and len(args) > 1:
        run_analysis(args[1].upper())
    elif args[0] == 'buy' and len(args) > 2:
        run_buy(args[1].upper(), int(args[2]))
    elif args[0] == 'sell' and len(args) > 2:
        run_sell(args[1].upper(), int(args[2]))
    elif args[0] == 'alerts':
        run_alerts()
    else:
        print_usage()
