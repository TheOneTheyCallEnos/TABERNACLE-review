#!/usr/bin/env python3
"""
LVS Market Navigator v7 - GUI Interface
========================================
Visual interface using Streamlit.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

from core.navigator import LVSNavigator
from core.coherence import compute_coherence_history
from storage.portfolio_manager import PortfolioManager
from notifications.push import NotificationManager


# === CONFIGURATION ===
STOCK_UNIVERSE = {
    'TD.TO': 'Toronto-Dominion Bank',
    'RY.TO': 'Royal Bank of Canada',
    'BMO.TO': 'Bank of Montreal',
    'BNS.TO': 'Bank of Nova Scotia',
    'ENB.TO': 'Enbridge',
    'CNR.TO': 'CN Railway',
    'SHOP.TO': 'Shopify',
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
}


# === PAGE CONFIG ===
st.set_page_config(
    page_title="LVS Market Navigator",
    page_icon="🧭",
    layout="wide"
)


# === CACHING ===
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, days: int = 250):
    """Fetch and cache stock data."""
    try:
        stock = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = stock.history(start=start, end=end)
        if len(hist) < 60:
            return None, None
        return hist['Close'], hist['Volume']
    except:
        return None, None


# === INITIALIZE ===
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
pm = PortfolioManager(data_dir)


# === SIDEBAR ===
st.sidebar.title("🧭 LVS Navigator")
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔍 Daily Scan",
    "🔬 Stock Analysis",
    "💼 Portfolio",
    "⚙️ Settings"
])


# === DASHBOARD PAGE ===
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    account = pm.get_account()
    
    # Account summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Capital", f"${account.get('capital', 0):,.2f}")
    with col2:
        st.metric("Goal", f"${account.get('goal', 0):,.2f}")
    with col3:
        st.metric("Timeframe", f"{account.get('timeframe_days', 30)} days")
    with col4:
        goal_pct = (account.get('goal', 0) / account.get('capital', 1) * 100) if account.get('capital', 0) > 0 else 0
        st.metric("Target Return", f"{goal_pct:.1f}%")
    
    st.divider()
    
    # Quick scan
    st.subheader("Quick Market Coherence")
    
    with st.spinner("Scanning market..."):
        coherence_data = []
        for ticker, name in list(STOCK_UNIVERSE.items())[:6]:
            prices, volumes = fetch_stock_data(ticker, 100)
            if prices is not None:
                from core.coherence import compute_coherence
                coh = compute_coherence(prices, volumes)
                coherence_data.append({
                    'Ticker': ticker,
                    'Name': name,
                    'p': coh['p'],
                    'κ': coh['kappa'],
                    'ρ': coh['rho'],
                    'σ': coh['sigma'],
                    'τ': coh['tau']
                })
        
        if coherence_data:
            df = pd.DataFrame(coherence_data)
            
            # Color-coded coherence
            def color_coherence(val):
                if val >= 0.85:
                    return 'background-color: #90EE90'
                elif val >= 0.70:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C1'
            
            st.dataframe(
                df.style.applymap(color_coherence, subset=['p']),
                use_container_width=True
            )
            
            # Bar chart
            fig = px.bar(df, x='Ticker', y='p', 
                        title='Market Coherence (p)',
                        color='p',
                        color_continuous_scale=['red', 'yellow', 'green'])
            fig.add_hline(y=0.70, line_dash="dash", line_color="orange", 
                         annotation_text="Minimum threshold")
            fig.add_hline(y=0.90, line_dash="dash", line_color="green",
                         annotation_text="High coherence")
            st.plotly_chart(fig, use_container_width=True)


# === DAILY SCAN PAGE ===
elif page == "🔍 Daily Scan":
    st.title("🔍 Daily Market Scan")
    
    account = pm.get_account()
    
    if account.get('capital', 0) == 0:
        st.warning("⚠️ Please set up your account in Settings first.")
    else:
        if st.button("🚀 Run Full Scan", type="primary"):
            nav = LVSNavigator(
                capital=account['capital'],
                goal=account.get('goal', 0),
                timeframe_days=account.get('timeframe_days', 30)
            )
            
            progress = st.progress(0)
            status = st.empty()
            
            # Fetch data
            total = len(STOCK_UNIVERSE)
            for i, (ticker, name) in enumerate(STOCK_UNIVERSE.items()):
                status.text(f"Fetching {ticker}...")
                prices, volumes = fetch_stock_data(ticker)
                if prices is not None:
                    nav.add_stock_data(ticker, prices, volumes)
                progress.progress((i + 1) / total)
            
            status.text("Analyzing...")
            report = nav.generate_full_report()
            
            progress.empty()
            status.empty()
            
            # Display results
            st.subheader("📊 Achievability Analysis")
            ach = report['achievability']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Coherence", f"{ach.get('market_p', 0):.2f}")
                st.metric("Your Goal", f"${ach.get('goal', 0):,.2f}")
            with col2:
                st.metric("Safe Target", f"${ach.get('safe_target', 0):,.2f}")
                st.metric("Max Target", f"${ach.get('max_target', 0):,.2f}")
            
            if ach.get('achievable'):
                st.success(f"✅ {ach.get('recommendation', 'Goal achievable!')}")
            else:
                st.warning(f"⚠️ {ach.get('recommendation', 'Consider reducing target')}")
            
            # Portfolio allocation
            st.subheader("💼 Optimal Allocation")
            port = report['portfolio']
            
            if port.get('success'):
                weights = {k: v for k, v in port['weights'].items() if v > 0.01}
                
                fig = px.pie(
                    values=list(weights.values()),
                    names=list(weights.keys()),
                    title="Recommended Portfolio"
                )
                st.plotly_chart(fig)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Return", f"{port.get('expected_return', 0):.1%}")
                with col2:
                    st.metric("Expected Profit", f"${port.get('expected_profit', 0):,.2f}")
            
            # Stock rankings
            st.subheader("🏆 Stock Rankings")
            analyses = report.get('stock_analyses', [])
            if analyses:
                df = pd.DataFrame(analyses)
                display_cols = ['ticker', 'p', 'expected_return', 'signal', 'signal_strength', 'path']
                st.dataframe(df[display_cols], use_container_width=True)
            
            # Alerts
            alerts = report.get('alerts', [])
            if alerts:
                st.subheader("🚨 Alerts")
                for alert in alerts:
                    if alert['severity'] == 'CRITICAL':
                        st.error(f"🚨 {alert['ticker']}: {alert['message']}")
                    elif alert['severity'] == 'HIGH':
                        st.warning(f"⚠️ {alert['ticker']}: {alert['message']}")
                    else:
                        st.info(f"ℹ️ {alert['ticker']}: {alert['message']}")


# === STOCK ANALYSIS PAGE ===
elif page == "🔬 Stock Analysis":
    st.title("🔬 Deep Stock Analysis")
    
    ticker = st.selectbox("Select Stock", list(STOCK_UNIVERSE.keys()))
    
    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            prices, volumes = fetch_stock_data(ticker, 250)
            
            if prices is None:
                st.error("Could not fetch data")
            else:
                account = pm.get_account()
                nav = LVSNavigator(
                    capital=account.get('capital', 10000),
                    goal=account.get('goal', 1000),
                    timeframe_days=account.get('timeframe_days', 30)
                )
                nav.add_stock_data(ticker, prices, volumes)
                
                analysis = nav.analyze_stock(ticker)
                
                # Price chart with MA200
                st.subheader("📈 Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices.index, y=prices.values,
                    name='Price', line=dict(color='blue')
                ))
                ma200 = prices.rolling(200).mean()
                fig.add_trace(go.Scatter(
                    x=ma200.index, y=ma200.values,
                    name='MA200 (Telos)', line=dict(color='orange', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Coherence breakdown
                st.subheader("🎯 Coherence Analysis")
                coh = analysis['coherence']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Global Coherence (p)", f"{coh['p']:.3f}")
                    st.metric("Path", analysis['path'])
                with col2:
                    # Gauge chart for p
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=coh['p'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'steps': [
                                {'range': [0, 0.65], 'color': "lightcoral"},
                                {'range': [0.65, 0.85], 'color': "lightyellow"},
                                {'range': [0.85, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.70
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Component breakdown
                st.subheader("Component Breakdown")
                components = pd.DataFrame({
                    'Component': ['κ (Clarity)', 'ρ (Precision)', 'σ (Structure)', 'τ (Trust)'],
                    'Value': [coh['kappa'], coh['rho'], coh['sigma'], coh['tau']]
                })
                fig = px.bar(components, x='Component', y='Value', 
                            color='Value', color_continuous_scale=['red', 'yellow', 'green'])
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Expected return
                st.subheader("📊 Expected Return")
                er = analysis['expected_return']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{er['expected_return']:.1%}")
                with col2:
                    st.metric("Velocity (Kinetic)", f"{er['velocity']:.1%}")
                with col3:
                    st.metric("Potential Q", f"{er['potential_Q']:.2f}")
                
                # Signal
                st.subheader("🚦 Signal")
                sig = analysis['signal']
                
                if sig['action'] == 'BUY':
                    st.success(f"✅ {sig['action']} ({sig['strength']})")
                elif sig['action'] in ['SELL', 'AVOID']:
                    st.error(f"🚨 {sig['action']} ({sig['strength']})")
                else:
                    st.warning(f"⚠️ {sig['action']} ({sig['strength']})")
                
                st.write(f"Reason: {sig['reason']}")
                
                # Archon analysis
                st.subheader("⚠️ Archon Deviation")
                archon = analysis['archon']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Deviation", f"{archon['deviation']:.3f}",
                             delta="ALERT!" if archon['alert'] else "Safe")
                    st.write(f"Dominant: {archon['dominant_archon']}")
                with col2:
                    archon_df = pd.DataFrame({
                        'Archon': ['Tyrant', 'Fragmentor', 'Noise-Lord', 'Bias'],
                        'Value': [archon['tyrant'], archon['fragmentor'], 
                                 archon['noise_lord'], archon['bias']]
                    })
                    fig = px.bar(archon_df, x='Archon', y='Value')
                    fig.add_hline(y=0.15, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)


# === PORTFOLIO PAGE ===
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Management")
    
    tab1, tab2, tab3 = st.tabs(["Holdings", "Record Trade", "History"])
    
    with tab1:
        holdings = pm.get_holdings()
        
        if not holdings:
            st.info("No holdings yet. Record a trade to get started.")
        else:
            # Fetch current prices
            current_prices = {}
            for h in holdings:
                try:
                    stock = yf.Ticker(h['ticker'])
                    current_prices[h['ticker']] = stock.history(period='1d')['Close'].iloc[-1]
                except:
                    current_prices[h['ticker']] = h['avg_price']
            
            portfolio = pm.calculate_portfolio_value(current_prices)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
            with col2:
                st.metric("Holdings", f"${portfolio['holdings_value']:,.2f}")
            with col3:
                st.metric("Cash", f"${portfolio['cash']:,.2f}")
            with col4:
                st.metric("P&L", f"${portfolio['total_pnl']:+,.2f}",
                         delta=f"{portfolio['total_pnl_pct']:+.1f}%")
            
            # Holdings table
            df = pd.DataFrame(portfolio['holdings'])
            st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("Record Trade")
        
        col1, col2 = st.columns(2)
        with col1:
            trade_type = st.selectbox("Type", ["BUY", "SELL"])
            ticker = st.text_input("Ticker", placeholder="e.g., TD.TO").upper()
        with col2:
            shares = st.number_input("Shares", min_value=0.0, step=1.0)
            price = st.number_input("Price ($)", min_value=0.0, step=0.01)
        
        if st.button("Record Trade", type="primary"):
            if ticker and shares > 0:
                if trade_type == "BUY":
                    pm.add_holding(ticker, shares, price if price > 0 else None)
                    st.success(f"Recorded: BUY {shares} {ticker}")
                else:
                    pm.remove_holding(ticker, shares, price if price > 0 else None)
                    st.success(f"Recorded: SELL {shares} {ticker}")
                st.rerun()
            else:
                st.error("Please enter ticker and shares")
    
    with tab3:
        transactions = pm.get_transactions()
        if transactions:
            st.dataframe(pd.DataFrame(transactions), use_container_width=True)
        else:
            st.info("No transactions yet.")


# === SETTINGS PAGE ===
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Account Configuration")
    
    account = pm.get_account()
    
    with st.form("account_form"):
        capital = st.number_input("Starting Capital ($)", 
                                  value=float(account.get('capital', 0)),
                                  min_value=0.0, step=100.0)
        goal = st.number_input("Profit Goal ($)",
                              value=float(account.get('goal', 0)),
                              min_value=0.0, step=100.0)
        timeframe = st.number_input("Timeframe (days)",
                                    value=int(account.get('timeframe_days', 30)),
                                    min_value=1, max_value=365)
        
        if st.form_submit_button("Save Settings", type="primary"):
            pm.set_account(capital, goal, timeframe)
            st.success("✅ Settings saved!")
            st.rerun()
    
    st.divider()
    
    st.subheader("Notifications")
    st.info("""
    **Push Notifications Setup (iPhone)**
    
    1. Download **ntfy** app from App Store
    2. Subscribe to topic: `lvs-navigator` (or your custom topic)
    3. The app will send alerts for:
       - 🚨 ABADDON (urgent sell signals)
       - 🚀 Strong buy opportunities
       - ⚠️ Coherence decay warnings
    """)
    
    settings = pm.get_settings()
    webhook = st.text_input("Custom Webhook URL (optional)", 
                           value=settings.get('notification_webhook', ''))
    
    if st.button("Save Notification Settings"):
        pm.set_notification_webhook(webhook)
        st.success("Saved!")


# === FOOTER ===
st.sidebar.divider()
st.sidebar.caption("LVS Market Navigator v7")
st.sidebar.caption("Based on LVS v9.0 Theorem")
st.sidebar.caption("© 2025")
