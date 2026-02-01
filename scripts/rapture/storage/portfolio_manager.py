"""
Portfolio Manager
=================
Handles persistent storage of portfolio data:
- Holdings (purchases, sells)
- Performance history
- Alert history

Data is stored in JSON for simplicity and portability.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class PortfolioManager:
    """
    Manages portfolio data with persistent storage.
    
    Data file: data/portfolio.json
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "portfolio.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or initialize data
        self.data = self._load_data()
    
    def _load_data(self) -> dict:
        """Load data from file or initialize."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize empty structure
        return {
            'account': {
                'capital': 0,
                'goal': 0,
                'timeframe_days': 30,
                'created_at': datetime.now().isoformat()
            },
            'holdings': [],  # Current holdings
            'transactions': [],  # Buy/sell history
            'alerts_history': [],  # Past alerts
            'daily_snapshots': [],  # Daily portfolio values
            'settings': {
                'notifications_enabled': True,
                'notification_webhook': ''
            }
        }
    
    def _save_data(self):
        """Save data to file."""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    # === ACCOUNT MANAGEMENT ===
    
    def set_account(self, capital: float, goal: float, timeframe_days: int = 30):
        """Set account parameters."""
        self.data['account'] = {
            'capital': capital,
            'goal': goal,
            'timeframe_days': timeframe_days,
            'updated_at': datetime.now().isoformat()
        }
        self._save_data()
    
    def get_account(self) -> dict:
        """Get account parameters."""
        return self.data['account']
    
    # === HOLDINGS MANAGEMENT ===
    
    def add_holding(self, ticker: str, shares: float, price: float, 
                    date: Optional[str] = None):
        """Add a new holding (BUY transaction)."""
        if date is None:
            date = datetime.now().isoformat()
        
        # Check if we already have this ticker
        existing = None
        for h in self.data['holdings']:
            if h['ticker'] == ticker:
                existing = h
                break
        
        if existing:
            # Average in
            total_shares = existing['shares'] + shares
            total_cost = (existing['shares'] * existing['avg_price']) + (shares * price)
            existing['shares'] = total_shares
            existing['avg_price'] = total_cost / total_shares
            existing['updated_at'] = date
        else:
            # New holding
            self.data['holdings'].append({
                'ticker': ticker,
                'shares': shares,
                'avg_price': price,
                'created_at': date,
                'updated_at': date
            })
        
        # Record transaction
        self.data['transactions'].append({
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'total': shares * price,
            'date': date
        })
        
        self._save_data()
    
    def remove_holding(self, ticker: str, shares: float, price: float,
                       date: Optional[str] = None):
        """Remove shares from holding (SELL transaction)."""
        if date is None:
            date = datetime.now().isoformat()
        
        for h in self.data['holdings']:
            if h['ticker'] == ticker:
                if shares >= h['shares']:
                    # Sell all
                    self.data['holdings'].remove(h)
                else:
                    h['shares'] -= shares
                    h['updated_at'] = date
                break
        
        # Record transaction
        self.data['transactions'].append({
            'type': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'total': shares * price,
            'date': date
        })
        
        self._save_data()
    
    def get_holdings(self) -> List[dict]:
        """Get current holdings."""
        return self.data['holdings']
    
    def get_holding(self, ticker: str) -> Optional[dict]:
        """Get specific holding."""
        for h in self.data['holdings']:
            if h['ticker'] == ticker:
                return h
        return None
    
    # === TRANSACTION HISTORY ===
    
    def get_transactions(self, limit: int = 50) -> List[dict]:
        """Get recent transactions."""
        return self.data['transactions'][-limit:]
    
    # === ALERT MANAGEMENT ===
    
    def add_alert(self, alert: dict):
        """Add alert to history."""
        alert['timestamp'] = datetime.now().isoformat()
        self.data['alerts_history'].append(alert)
        
        # Keep only last 100 alerts
        if len(self.data['alerts_history']) > 100:
            self.data['alerts_history'] = self.data['alerts_history'][-100:]
        
        self._save_data()
    
    def get_alerts(self, limit: int = 20) -> List[dict]:
        """Get recent alerts."""
        return self.data['alerts_history'][-limit:]
    
    # === DAILY SNAPSHOTS ===
    
    def add_snapshot(self, total_value: float, holdings_value: float,
                     cash: float, pnl: float, pnl_pct: float):
        """Add daily portfolio snapshot."""
        self.data['daily_snapshots'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_value': total_value,
            'holdings_value': holdings_value,
            'cash': cash,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        
        # Keep only last 365 days
        if len(self.data['daily_snapshots']) > 365:
            self.data['daily_snapshots'] = self.data['daily_snapshots'][-365:]
        
        self._save_data()
    
    def get_snapshots(self) -> pd.DataFrame:
        """Get snapshot history as DataFrame."""
        if not self.data['daily_snapshots']:
            return pd.DataFrame()
        return pd.DataFrame(self.data['daily_snapshots'])
    
    # === PORTFOLIO VALUE CALCULATION ===
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> dict:
        """
        Calculate current portfolio value.
        
        current_prices: {ticker: price}
        """
        holdings_value = 0
        holdings_detail = []
        
        for h in self.data['holdings']:
            ticker = h['ticker']
            shares = h['shares']
            avg_price = h['avg_price']
            
            current_price = current_prices.get(ticker, avg_price)
            market_value = shares * current_price
            cost_basis = shares * avg_price
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            holdings_value += market_value
            
            holdings_detail.append({
                'ticker': ticker,
                'shares': shares,
                'avg_price': avg_price,
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        initial_capital = self.data['account'].get('capital', 0)
        
        # Calculate cash (simplified - capital minus what's invested)
        total_invested = sum(h['shares'] * h['avg_price'] for h in self.data['holdings'])
        cash = max(0, initial_capital - total_invested)
        
        total_value = holdings_value + cash
        total_pnl = total_value - initial_capital
        total_pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        
        return {
            'total_value': total_value,
            'holdings_value': holdings_value,
            'cash': cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'initial_capital': initial_capital,
            'goal': self.data['account'].get('goal', 0),
            'goal_progress_pct': (total_pnl / self.data['account'].get('goal', 1) * 100) if self.data['account'].get('goal', 0) > 0 else 0,
            'holdings': holdings_detail
        }
    
    # === SETTINGS ===
    
    def set_notification_webhook(self, webhook: str):
        """Set notification webhook URL."""
        self.data['settings']['notification_webhook'] = webhook
        self._save_data()
    
    def get_settings(self) -> dict:
        """Get settings."""
        return self.data['settings']
    
    # === EXPORT ===
    
    def export_to_csv(self, output_dir: str = "."):
        """Export data to CSV files."""
        # Holdings
        if self.data['holdings']:
            pd.DataFrame(self.data['holdings']).to_csv(
                os.path.join(output_dir, 'holdings.csv'), index=False)
        
        # Transactions
        if self.data['transactions']:
            pd.DataFrame(self.data['transactions']).to_csv(
                os.path.join(output_dir, 'transactions.csv'), index=False)
        
        # Snapshots
        if self.data['daily_snapshots']:
            pd.DataFrame(self.data['daily_snapshots']).to_csv(
                os.path.join(output_dir, 'snapshots.csv'), index=False)
