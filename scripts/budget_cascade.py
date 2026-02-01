#!/usr/bin/env python3
"""
Budget Cascade Calculator - Comprehensive Financial Management CLI
==================================================================

A complete tool for Virgil to manage Enos's 2026 Visa payoff plan.

QUICK REFERENCE:
----------------
# Basic operations
python budget_cascade.py                              # Show summary
python budget_cascade.py --output cascade             # Full cascade table
python budget_cascade.py --output keys                # Values for manual updates

# Balance changes
python budget_cascade.py --visa-balance 17500         # What-if: new balance
python budget_cascade.py --visa-balance 17500 --save  # Save to YAML

# What-if scenarios
python budget_cascade.py --extra-visa-payment 500     # Extra payment
python budget_cascade.py --add-charge 200 "Dinner"    # One-time charge
python budget_cascade.py --add-income 1000 "Bonus" --allocation visa

# Subscriptions
python budget_cascade.py --add-subscription "Service" 29.99 --day 15
python budget_cascade.py --remove-subscription "Netflix"
python budget_cascade.py --modify-subscription "Claude MAX" 156.00

# Compare scenarios
python budget_cascade.py --compare --extra-visa-payment 1000
python budget_cascade.py --compare --visa-balance 18000

# Markdown regeneration
python budget_cascade.py --regenerate-markdown

Author: Virgil (for Enos)
Version: 3.0 - January 8, 2026
"""

import argparse
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ============================================================================
# YAML HANDLING
# ============================================================================

YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
YAML_PATH = PROJECT_ROOT / "03_LL_RELATION" / "FINANCE" / "budget_data_2026.yaml"
MARKDOWN_PATH = PROJECT_ROOT / "03_LL_RELATION" / "FINANCE" / "2026_Budget_Corrected.md"

# Markers for markdown regeneration
CASCADE_START = "<!-- BUDGET_CASCADE_START -->"
CASCADE_END = "<!-- BUDGET_CASCADE_END -->"


# ============================================================================
# ANSI COLORS
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def color(text: str, c: str) -> str:
    """Wrap text in color codes."""
    return f"{c}{text}{Colors.END}"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PayPeriod:
    """Represents a single pay period."""
    date_str: str
    month: int
    day: int
    days_since_last: int
    income: float
    cash_expenses: float
    is_mid_month: bool
    special_visa_charges: float = 0.0
    special_notes: str = ""
    extra_to_visa: float = 0.0


@dataclass
class PeriodResult:
    """Results for a single pay period."""
    period: PayPeriod
    visa_start: float
    interest: float
    subs_charged: float
    special_charges: float
    visa_before_payment: float
    available_for_split: float
    visa_payment: float
    visa_end: float
    affirm_end: float
    nest_end: float
    prov_end: float
    wealth_total: float
    is_payoff: bool
    remaining_after_payoff: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class CascadeResults:
    """Complete results from a cascade calculation."""
    periods: List[PeriodResult]
    starting_visa: float
    payoff_date: Optional[str]
    payoff_amount: float
    total_interest_paid: float
    year_end_visa: float
    year_end_affirm: float
    year_end_nest: float
    year_end_prov: float
    year_end_wealth: float
    year_end_net_worth: float
    warnings: List[str]


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

def get_default_config() -> dict:
    """Return built-in default configuration (v2.2)."""
    return {
        'metadata': {
            'version': '2.2',
            'last_updated': '2026-01-08',
            'notes': 'Built-in defaults (Claude MAX 20x upgrade)'
        },
        'balances': {
            'visa': {
                'posted': 14924.27,
                'pending': 2124.65,
                'effective': 17048.92,
                'apr': 21.99,
                'daily_rate': 0.000602
            },
            'affirm': {'total': 10166.07}
        },
        'subscriptions': {
            'monthly': [
                {'name': 'Google Workspace', 'amount': 33.85, 'day': 1},
                {'name': 'iCloud', 'amount': 13.64, 'day': 8},
                {'name': 'Spotify', 'amount': 14.21, 'day': 8},
                {'name': 'Gemini Pro', 'amount': 30.23, 'day': 11},
                {'name': 'Claude MAX', 'amount': 123.78, 'day': 12},
                {'name': 'Cursor', 'amount': 90.00, 'day': 16},
                {'name': 'Netflix', 'amount': 21.27, 'day': 16},
                {'name': 'Breathwrk', 'amount': 13.44, 'day': 17},
                {'name': 'ToDoIst', 'amount': 6.15, 'day': 17},
                {'name': 'Day One', 'amount': 13.43, 'day': 17},
                {'name': 'QuickBooks', 'amount': 24.94, 'day': 18},
                {'name': 'Google Backup', 'amount': 4.47, 'day': 21},
            ],
            'totals': {
                'first_half': 215.71,
                'second_half': 173.70,
                'monthly': 389.41
            },
            'yearly': [
                {'name': 'CAA', 'date': '2026-05-17', 'amount': 78.75},
                {'name': 'Squarespace', 'date': '2026-08-29', 'amount': 17.00},
                {'name': 'Calendly/HubSpot/DocuSign', 'date': '2026-09-02', 'amount': 863.10},
                {'name': 'TD Annual Fee', 'date': '2026-09-15', 'amount': 139.00},
            ]
        },
        'expenses': {
            'totals': {
                'mid_month_base': 1065.00,
                'mid_month_affirm_jan': 76.42,
                'mid_month_affirm_feb_on': 433.57,
                'end_of_month': 1375.00
            }
        },
        'allocation': {
            'debt_mode': {'visa': 0.50, 'affirm_extra': 0.10, 'nest_egg': 0.30, 'provisions': 0.10},
            'wealth_mode': {'nest_egg': 0.50, 'provisions': 0.30, 'affirm_extra': 0.20}
        },
        'income': {
            'events': [
                {'date': '2026-01-15', 'amount': 275.00, 'allocation': 'visa_direct', 'description': 'Partner reimbursement'},
                {'date': '2026-01-31', 'amount': 3600.00, 'description': 'Davis prorated'},
                {'date': '2026-04-15', 'amount': 6500.00, 'allocation': 'visa_direct', 'description': 'Tax return'}
            ]
        },
        'one_time_charges': []
    }


# ============================================================================
# YAML I/O
# ============================================================================

def load_yaml_data(yaml_path: Path = YAML_PATH) -> dict:
    """Load budget data from YAML file or return defaults."""
    if not YAML_AVAILABLE:
        print(color("⚠️  PyYAML not installed. Using built-in defaults.", Colors.YELLOW))
        print("   For full features: pip install pyyaml\n")
        return get_default_config()
    
    if not yaml_path.exists():
        print(color(f"⚠️  YAML not found: {yaml_path}", Colors.YELLOW))
        print("   Using built-in defaults.\n")
        return get_default_config()
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_data(data: dict, yaml_path: Path = YAML_PATH) -> bool:
    """Save budget data to YAML file."""
    if not YAML_AVAILABLE:
        print(color("❌ Cannot save: PyYAML not installed", Colors.RED))
        print("   Install with: pip install pyyaml")
        return False
    
    data['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(color(f"✅ Saved to {yaml_path}", Colors.GREEN))
    return True


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class BudgetConfig:
    """Configuration loaded from YAML data."""
    
    def __init__(self, data: dict):
        self.data = data
        
        # Balances
        bal = data['balances']
        self.starting_visa = bal['visa']['effective']
        self.daily_rate = bal['visa']['daily_rate']
        self.starting_affirm = bal['affirm']['total']
        
        # Calculate subscription totals from monthly list
        self._calculate_subscription_totals()
        
        # Yearly charges
        self.yearly_charges = []
        for charge in data['subscriptions'].get('yearly', []):
            dt = datetime.strptime(charge['date'], "%Y-%m-%d")
            self.yearly_charges.append({
                'month': dt.month,
                'day': dt.day,
                'amount': charge['amount'],
                'name': charge['name']
            })
        
        # One-time charges
        self.one_time_charges = []
        for charge in data.get('one_time_charges', []):
            if not charge.get('included_in_pending', False):
                dt = datetime.strptime(charge['date'], "%Y-%m-%d")
                self.one_time_charges.append({
                    'month': dt.month,
                    'day': dt.day,
                    'amount': charge['amount'],
                    'description': charge.get('description', '')
                })
        
        # Expenses
        exp = data['expenses']
        self.mid_month_base = exp['totals']['mid_month_base']
        self.affirm_jan = exp['totals']['mid_month_affirm_jan']
        self.affirm_feb_on = exp['totals']['mid_month_affirm_feb_on']
        self.end_of_month_expenses = exp['totals']['end_of_month']
        
        # Allocation
        alloc = data['allocation']
        self.debt_mode = alloc['debt_mode']
        self.wealth_mode = alloc['wealth_mode']
        
        # Income events
        self.income_events = {}
        for event in data['income'].get('events', []):
            self.income_events[event['date']] = event
    
    def _calculate_subscription_totals(self):
        """Calculate first/second half subscription totals from monthly list."""
        monthly = self.data['subscriptions'].get('monthly', [])
        
        first_half = sum(s['amount'] for s in monthly if s['day'] <= 15)
        second_half = sum(s['amount'] for s in monthly if s['day'] > 15)
        
        self.subs_first_half = round(first_half, 2)
        self.subs_second_half = round(second_half, 2)
        
        # Validate against stored totals if present
        stored = self.data['subscriptions'].get('totals', {})
        if stored:
            if abs(self.subs_first_half - stored.get('first_half', 0)) > 0.01:
                print(color(f"⚠️  First-half subs mismatch: calculated ${self.subs_first_half}, stored ${stored.get('first_half')}", Colors.YELLOW))
            if abs(self.subs_second_half - stored.get('second_half', 0)) > 0.01:
                print(color(f"⚠️  Second-half subs mismatch: calculated ${self.subs_second_half}, stored ${stored.get('second_half')}", Colors.YELLOW))
    
    def get_subscription(self, name: str) -> Optional[dict]:
        """Find a subscription by name (case-insensitive)."""
        for sub in self.data['subscriptions'].get('monthly', []):
            if sub['name'].lower() == name.lower():
                return sub
        return None
    
    def add_subscription(self, name: str, amount: float, day: int):
        """Add a new subscription."""
        self.data['subscriptions'].setdefault('monthly', []).append({
            'name': name,
            'amount': amount,
            'day': day
        })
        self._calculate_subscription_totals()
    
    def remove_subscription(self, name: str) -> bool:
        """Remove a subscription by name."""
        monthly = self.data['subscriptions'].get('monthly', [])
        for i, sub in enumerate(monthly):
            if sub['name'].lower() == name.lower():
                del monthly[i]
                self._calculate_subscription_totals()
                return True
        return False
    
    def modify_subscription(self, name: str, new_amount: float) -> bool:
        """Modify a subscription amount."""
        for sub in self.data['subscriptions'].get('monthly', []):
            if sub['name'].lower() == name.lower():
                sub['amount'] = new_amount
                self._calculate_subscription_totals()
                return True
        return False
    
    def add_one_time_charge(self, amount: float, description: str, date_str: str):
        """Add a one-time Visa charge."""
        self.data.setdefault('one_time_charges', []).append({
            'date': date_str,
            'amount': amount,
            'description': description,
            'included_in_pending': False
        })
        # Refresh config
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        self.one_time_charges.append({
            'month': dt.month,
            'day': dt.day,
            'amount': amount,
            'description': description
        })
    
    def add_income_event(self, amount: float, description: str, date_str: str, allocation: str = 'split'):
        """Add a one-time income event."""
        self.data['income'].setdefault('events', []).append({
            'date': date_str,
            'amount': amount,
            'description': description,
            'allocation': 'visa_direct' if allocation == 'visa' else 'split'
        })
        self.income_events[date_str] = self.data['income']['events'][-1]


# ============================================================================
# PAY PERIOD GENERATION
# ============================================================================

def get_pay_periods(config: BudgetConfig) -> List[PayPeriod]:
    """Generate all pay periods for 2026."""
    
    periods = []
    
    # Helper to get special charges for a period
    def get_special_charges(month: int, start_day: int, end_day: int) -> Tuple[float, str]:
        total = 0.0
        notes = []
        for yc in config.yearly_charges:
            if yc['month'] == month and start_day < yc['day'] <= end_day:
                total += yc['amount']
                notes.append(yc['name'])
        for oc in config.one_time_charges:
            if oc['month'] == month and start_day < oc['day'] <= end_day:
                total += oc['amount']
                notes.append(oc.get('description', 'One-time'))
        return total, ", ".join(notes)
    
    # January 15 - special
    jan15_event = config.income_events.get("2026-01-15", {})
    jan15_extra = jan15_event.get('amount', 0) if jan15_event.get('allocation') == 'visa_direct' else 0
    special, notes = get_special_charges(1, 0, 15)
    
    periods.append(PayPeriod(
        "January 15", 1, 15, 7,
        income=2000.00,
        cash_expenses=120 + 150 + 195 + config.affirm_jan,
        is_mid_month=True,
        special_visa_charges=special,
        extra_to_visa=jan15_extra,
        special_notes=notes or "First consulting payment"
    ))
    
    # January 31
    jan31_event = config.income_events.get("2026-01-31", {})
    jan31_income = jan31_event.get('amount', 2000.00)
    special, notes = get_special_charges(1, 15, 31)
    
    periods.append(PayPeriod(
        "January 31", 1, 31, 16,
        income=jan31_income,
        cash_expenses=config.end_of_month_expenses,
        is_mid_month=False,
        special_visa_charges=special,
        special_notes=notes or "Davis prorated"
    ))
    
    # February through December
    months = [
        (2, "February", 28, 13),
        (3, "March", 31, 16),
        (4, "April", 30, 15),
        (5, "May", 31, 16),
        (6, "June", 30, 15),
        (7, "July", 31, 16),
        (8, "August", 31, 16),
        (9, "September", 30, 15),
        (10, "October", 31, 16),
        (11, "November", 30, 15),
        (12, "December", 31, 16),
    ]
    
    for month_num, month_name, last_day, days_mid_to_end in months:
        # Mid-month (15th)
        mid_date = f"2026-{month_num:02d}-15"
        mid_event = config.income_events.get(mid_date, {})
        extra_to_visa = mid_event.get('amount', 0) if mid_event.get('allocation') == 'visa_direct' else 0
        special, notes = get_special_charges(month_num, 0, 15)
        
        # Check for income events that replace regular income
        base_income = 4000.00  # Davis + Consulting
        
        periods.append(PayPeriod(
            f"{month_name} 15", month_num, 15, 15,
            income=base_income,
            cash_expenses=config.mid_month_base + config.affirm_feb_on,
            is_mid_month=True,
            special_visa_charges=special,
            extra_to_visa=extra_to_visa,
            special_notes=notes or ("TAX RETURN" if extra_to_visa > 0 else "")
        ))
        
        # End of month
        end_date = f"2026-{month_num:02d}-{last_day}"
        end_event = config.income_events.get(end_date, {})
        extra_to_visa = end_event.get('amount', 0) if end_event.get('allocation') == 'visa_direct' else 0
        special, notes = get_special_charges(month_num, 15, last_day)
        
        periods.append(PayPeriod(
            f"{month_name} {last_day}", month_num, last_day, days_mid_to_end,
            income=2000.00,  # Davis only
            cash_expenses=config.end_of_month_expenses,
            is_mid_month=False,
            special_visa_charges=special,
            extra_to_visa=extra_to_visa,
            special_notes=notes
        ))
    
    return periods


# ============================================================================
# CALCULATION ENGINE
# ============================================================================

def calculate_cascade(
    config: BudgetConfig,
    starting_visa: float = None,
    extra_visa_payment: float = 0,
    extra_payment_date: str = None
) -> CascadeResults:
    """Calculate the entire cascade and return comprehensive results."""
    
    periods = get_pay_periods(config)
    results = []
    warnings = []
    
    visa = starting_visa if starting_visa is not None else config.starting_visa
    affirm = config.starting_affirm
    nest = 0.0
    prov = 0.0
    visa_paid_off = False
    extra_payment_applied = False
    
    total_interest = 0.0
    payoff_date = None
    payoff_amount = 0.0
    
    for period in periods:
        period_warnings = []
        
        # Subscription charge
        subs = config.subs_first_half if period.is_mid_month else config.subs_second_half
        
        # Interest (only pre-payoff)
        if not visa_paid_off:
            interest = visa * config.daily_rate * period.days_since_last
            total_interest += interest
        else:
            interest = 0
        
        # Balance before payment
        visa_before = visa + interest + subs + period.special_visa_charges
        
        # Available for split
        available = period.income - period.cash_expenses
        
        # Sanity check: negative available
        if available < 0:
            period_warnings.append(f"⚠️ Negative cash: ${available:.2f}")
            warnings.append(f"{period.date_str}: Expenses (${period.cash_expenses:.2f}) exceed income (${period.income:.2f})")
        
        # Extra payments
        period_extra = period.extra_to_visa
        
        # Apply one-time extra payment
        if not extra_payment_applied and extra_visa_payment > 0:
            if extra_payment_date:
                # Apply on specific date
                target = f"{period.month:02d}-{period.day:02d}"
                if extra_payment_date.endswith(target):
                    period_extra += extra_visa_payment
                    extra_payment_applied = True
            else:
                # Apply on first period
                period_extra += extra_visa_payment
                extra_payment_applied = True
        
        if not visa_paid_off:
            # PRE-PAYOFF: 50/10/30/10
            visa_from_split = available * config.debt_mode['visa']
            total_visa_payment = visa_from_split + period_extra
            
            # Check for payoff
            if visa_before <= available + period_extra:
                visa_payment = visa_before
                remaining = available + period_extra - visa_before
                
                nest_add = remaining * config.wealth_mode['nest_egg']
                prov_add = remaining * config.wealth_mode['provisions']
                affirm_extra = remaining * config.wealth_mode['affirm_extra']
                
                visa = 0.0
                visa_paid_off = True
                is_payoff = True
                remaining_after_payoff = remaining
                payoff_date = period.date_str
                payoff_amount = visa_before
            else:
                visa_payment = total_visa_payment
                visa = visa_before - visa_payment
                
                nest_add = available * config.debt_mode['nest_egg']
                prov_add = available * config.debt_mode['provisions']
                affirm_extra = available * config.debt_mode['affirm_extra']
                
                is_payoff = False
                remaining_after_payoff = 0.0
                
                # Warning: balance still high late in year
                if period.month >= 10 and visa > 2000:
                    period_warnings.append(f"⚠️ High balance in {period.date_str}")
        else:
            # POST-PAYOFF: Pay subs, then 50/30/20
            visa_payment = subs + period.special_visa_charges
            visa = 0.0
            
            remaining = available - visa_payment
            
            nest_add = remaining * config.wealth_mode['nest_egg']
            prov_add = remaining * config.wealth_mode['provisions']
            affirm_extra = remaining * config.wealth_mode['affirm_extra']
            
            is_payoff = False
            remaining_after_payoff = 0.0
            visa_before = subs + period.special_visa_charges
        
        # Update Affirm
        affirm -= affirm_extra
        
        # Sanity check: negative affirm
        if affirm < 0:
            period_warnings.append(f"⚠️ Affirm negative: ${affirm:.2f}")
        
        # Update wealth
        nest += nest_add
        prov += prov_add
        
        result = PeriodResult(
            period=period,
            visa_start=visa + visa_payment if not visa_paid_off or is_payoff else 0.0,
            interest=interest,
            subs_charged=subs,
            special_charges=period.special_visa_charges,
            visa_before_payment=visa_before,
            available_for_split=available,
            visa_payment=visa_payment,
            visa_end=visa,
            affirm_end=affirm,
            nest_end=nest,
            prov_end=prov,
            wealth_total=nest + prov,
            is_payoff=is_payoff,
            remaining_after_payoff=remaining_after_payoff,
            warnings=period_warnings
        )
        
        results.append(result)
    
    # Final results
    final = results[-1]
    
    # Warning: not paid off by year end
    if not payoff_date:
        warnings.append(f"⚠️ Visa NOT paid off by Dec 31, 2026! Remaining: ${final.visa_end:.2f}")
    
    return CascadeResults(
        periods=results,
        starting_visa=starting_visa if starting_visa else config.starting_visa,
        payoff_date=payoff_date,
        payoff_amount=payoff_amount,
        total_interest_paid=total_interest,
        year_end_visa=final.visa_end,
        year_end_affirm=final.affirm_end,
        year_end_nest=final.nest_end,
        year_end_prov=final.prov_end,
        year_end_wealth=final.wealth_total,
        year_end_net_worth=final.wealth_total - final.affirm_end,
        warnings=warnings
    )


# ============================================================================
# OUTPUT FORMATTERS
# ============================================================================

def fmt(amount: float) -> str:
    """Format currency."""
    return f"${amount:,.2f}"


def print_summary(results: CascadeResults):
    """Print summary of key metrics."""
    
    print("\n" + "=" * 65)
    print(color("📊 BUDGET SUMMARY", Colors.BOLD))
    print("=" * 65)
    
    print(f"\n💳 Starting Visa: {color(fmt(results.starting_visa), Colors.CYAN)}")
    
    if results.payoff_date:
        print(f"🎉 Payoff Date: {color(results.payoff_date, Colors.GREEN)}")
        print(f"   Payoff Amount: {fmt(results.payoff_amount)}")
    else:
        print(color(f"⚠️  NOT PAID OFF by Dec 31, 2026", Colors.RED))
        print(f"   Remaining: {fmt(results.year_end_visa)}")
    
    print(f"\n💰 Total Interest Paid: {color(fmt(results.total_interest_paid), Colors.YELLOW)}")
    
    print(f"\n📅 Year-End (December 31, 2026):")
    print(f"   Visa: {fmt(results.year_end_visa)}")
    print(f"   Affirm: {fmt(results.year_end_affirm)}")
    print(f"   Nest Egg: {color(fmt(results.year_end_nest), Colors.GREEN)}")
    print(f"   Provisions: {color(fmt(results.year_end_prov), Colors.GREEN)}")
    print(f"   Total Wealth: {color(fmt(results.year_end_wealth), Colors.BOLD)}")
    print(f"   Net Worth: {color(fmt(results.year_end_net_worth), Colors.BOLD)}")
    
    if results.warnings:
        print(f"\n{color('⚠️  WARNINGS:', Colors.YELLOW)}")
        for w in results.warnings:
            print(f"   {w}")
    
    print("\n" + "=" * 65)


def print_cascade(results: CascadeResults):
    """Print full cascade table."""
    
    print("\n" + "=" * 115)
    print(color("VISA BALANCE CASCADE", Colors.BOLD))
    print("=" * 115)
    print(f"{'Period':<18} {'Start':>12} {'Interest':>10} {'Subs':>10} {'Special':>10} "
          f"{'Before':>12} {'Payment':>12} {'End':>12}")
    print("-" * 115)
    
    for r in results.periods:
        marker = color(" 🎉", Colors.GREEN) if r.is_payoff else ""
        end_color = Colors.GREEN if r.visa_end == 0 else ""
        print(f"{r.period.date_str:<18} "
              f"{fmt(r.visa_start):>12} "
              f"{fmt(r.interest):>10} "
              f"{fmt(r.subs_charged):>10} "
              f"{fmt(r.special_charges):>10} "
              f"{fmt(r.visa_before_payment):>12} "
              f"{fmt(r.visa_payment):>12} "
              f"{color(fmt(r.visa_end), end_color):>12}{marker}")
    
    print("=" * 115)
    print(f"\n💰 Total Interest Paid: {fmt(results.total_interest_paid)}")


def print_keys(results: CascadeResults):
    """Print key values for manual markdown updates."""
    
    print("\n" + "=" * 80)
    print(color("KEY VALUES FOR MANUAL UPDATES", Colors.BOLD))
    print("=" * 80)
    print(f"{'Period':<18} {'Visa':>12} {'Nest':>12} {'Prov':>12} {'Affirm':>12} {'Wealth':>12}")
    print("-" * 80)
    
    for r in results.periods:
        marker = " *PAYOFF*" if r.is_payoff else ""
        print(f"{r.period.date_str:<18} "
              f"{fmt(r.visa_end):>12} "
              f"{fmt(r.nest_end):>12} "
              f"{fmt(r.prov_end):>12} "
              f"{fmt(r.affirm_end):>12} "
              f"{fmt(r.wealth_total):>12}{marker}")
    
    print("=" * 80)
    print("\n📋 Copy these values to update 2026_Budget_Corrected.md")


def print_json(results: CascadeResults):
    """Print results as JSON."""
    output = {
        'starting_visa': results.starting_visa,
        'payoff_date': results.payoff_date,
        'payoff_amount': round(results.payoff_amount, 2),
        'total_interest': round(results.total_interest_paid, 2),
        'year_end': {
            'visa': round(results.year_end_visa, 2),
            'affirm': round(results.year_end_affirm, 2),
            'nest': round(results.year_end_nest, 2),
            'prov': round(results.year_end_prov, 2),
            'wealth': round(results.year_end_wealth, 2),
            'net_worth': round(results.year_end_net_worth, 2),
        },
        'periods': []
    }
    
    for r in results.periods:
        output['periods'].append({
            'date': r.period.date_str,
            'visa_end': round(r.visa_end, 2),
            'nest': round(r.nest_end, 2),
            'prov': round(r.prov_end, 2),
            'affirm': round(r.affirm_end, 2),
            'is_payoff': r.is_payoff,
        })
    
    print(json.dumps(output, indent=2))


def print_compare(current: CascadeResults, scenario: CascadeResults, scenario_desc: str):
    """Print side-by-side comparison."""
    
    print("\n" + "=" * 75)
    print(color("📊 SCENARIO COMPARISON", Colors.BOLD))
    print("=" * 75)
    print(f"{'Metric':<25} {'CURRENT':>15} {'SCENARIO':>15} {'DIFFERENCE':>15}")
    print("-" * 75)
    
    # Starting Visa
    diff = scenario.starting_visa - current.starting_visa
    diff_str = f"+{fmt(diff)}" if diff > 0 else fmt(diff)
    print(f"{'Starting Visa':<25} {fmt(current.starting_visa):>15} {fmt(scenario.starting_visa):>15} {diff_str:>15}")
    
    # Payoff Date
    curr_date = current.payoff_date or "Not paid off"
    scen_date = scenario.payoff_date or "Not paid off"
    if current.payoff_date and scenario.payoff_date:
        # Simple month comparison
        curr_month = datetime.strptime(current.payoff_date.split()[0], "%B").month
        scen_month = datetime.strptime(scenario.payoff_date.split()[0], "%B").month
        diff_months = scen_month - curr_month
        diff_str = f"+{diff_months} mo" if diff_months > 0 else f"{diff_months} mo"
    else:
        diff_str = "—"
    print(f"{'Payoff Date':<25} {curr_date:>15} {scen_date:>15} {diff_str:>15}")
    
    # Payoff Amount
    if current.payoff_date and scenario.payoff_date:
        diff = scenario.payoff_amount - current.payoff_amount
        diff_str = f"+{fmt(diff)}" if diff > 0 else fmt(diff)
        print(f"{'Payoff Amount':<25} {fmt(current.payoff_amount):>15} {fmt(scenario.payoff_amount):>15} {diff_str:>15}")
    
    # Total Interest
    diff = scenario.total_interest_paid - current.total_interest_paid
    diff_color = Colors.RED if diff > 0 else Colors.GREEN
    diff_str = f"+{fmt(diff)}" if diff > 0 else fmt(diff)
    print(f"{'Total Interest':<25} {fmt(current.total_interest_paid):>15} {fmt(scenario.total_interest_paid):>15} {color(diff_str, diff_color):>15}")
    
    # Year-end Net Worth
    diff = scenario.year_end_net_worth - current.year_end_net_worth
    diff_color = Colors.GREEN if diff > 0 else Colors.RED
    diff_str = f"+{fmt(diff)}" if diff > 0 else fmt(diff)
    print(f"{'Year-End Net Worth':<25} {fmt(current.year_end_net_worth):>15} {fmt(scenario.year_end_net_worth):>15} {color(diff_str, diff_color):>15}")
    
    # Year-end Wealth
    diff = scenario.year_end_wealth - current.year_end_wealth
    diff_color = Colors.GREEN if diff > 0 else Colors.RED
    diff_str = f"+{fmt(diff)}" if diff > 0 else fmt(diff)
    print(f"{'Year-End Wealth':<25} {fmt(current.year_end_wealth):>15} {fmt(scenario.year_end_wealth):>15} {color(diff_str, diff_color):>15}")
    
    print("=" * 75)
    print(f"\n📝 Scenario: {scenario_desc}")
    
    # Summary insight
    interest_diff = scenario.total_interest_paid - current.total_interest_paid
    if interest_diff < 0:
        print(color(f"✅ This scenario saves {fmt(abs(interest_diff))} in interest!", Colors.GREEN))
    elif interest_diff > 0:
        print(color(f"⚠️  This scenario costs {fmt(interest_diff)} more in interest", Colors.YELLOW))


# ============================================================================
# MARKDOWN REGENERATION
# ============================================================================

def generate_period_markdown(r: PeriodResult) -> str:
    """Generate markdown for a single pay period."""
    lines = []
    
    # Header
    emoji = " 🎉" if r.is_payoff else ""
    if r.period.special_notes and "TAX" in r.period.special_notes:
        emoji = " 🎯 TAX RETURN"
    lines.append(f"#### {r.period.date_str}{emoji}")
    
    # Income
    if r.period.is_mid_month:
        if r.period.month == 1:
            lines.append(f"**Income:** $2,275.00 (Consulting $2,000 + Partner reimbursement $275)")
        else:
            lines.append(f"**Income:** $4,000.00 (Davis $2,000 + Consulting $2,000)")
    else:
        if r.period.month == 1:
            lines.append(f"**Income:** $3,600.00 (Davis prorated)")
        else:
            lines.append(f"**Income:** $2,000.00 (Davis)")
    
    # Expenses
    if r.period.is_mid_month:
        if r.period.month == 1:
            lines.append("- Misc $310 (from chequing)")
            lines.append("- Food $240 (from chequing)")
            lines.append("- Cousin Gifts $150 (from chequing)")
        else:
            lines.append("- Misc $300")
            lines.append("- Food $300")
        lines.append("- Hydro $120")
        lines.append("- Internet $150")
        lines.append("- Phone Bill $195")
        affirm = 76.42 if r.period.month == 1 else 433.57
        lines.append(f"- Affirm ${affirm:.2f}")
    else:
        lines.append("- Misc $300")
        lines.append("- Food $300")
        lines.append("- TD Chequing Monthly Fee $25")
        lines.append("- Rent $750")
    
    lines.append(f"= **${r.available_for_split:,.2f} available for split**")
    
    # Special notes
    if r.period.special_charges > 0:
        lines.append(f"⚠️ {r.period.special_notes}: ${r.period.special_charges:.2f}")
    
    # Split
    if r.visa_end > 0 or r.is_payoff:
        lines.append("\n**💰 50/10/30/10 SPLIT:**")
        visa_pct = r.available_for_split * 0.5
        lines.append(f"- Visa ${visa_pct:.2f} (50%)")
        lines.append(f"- Extra Affirm ${r.available_for_split * 0.1:.2f} (10%)")
        lines.append(f"- Nest +${r.available_for_split * 0.3:.2f} (30%)")
        lines.append(f"- Prov +${r.available_for_split * 0.1:.2f} (10%)")
    
    # Visa Status
    lines.append("\n**Visa Status:**")
    
    if r.is_payoff:
        line = f"- Balance at payment: ${r.visa_start:.2f} + ${r.interest:.2f} interest ({r.period.days_since_last} days) + ${r.subs_charged:.2f} subs"
        if r.special_charges > 0:
            line += f" + ${r.special_charges:.2f} special"
        line += f" = **${r.visa_before_payment:.2f}**"
        lines.append(line)
        lines.append(f"- Available for payoff: Full available ${r.available_for_split:.2f}")
        lines.append(f"- **Balance ${r.visa_before_payment:.2f} < Available → PAYOFF!**")
        lines.append(f"\n**🎉 VISA PAYOFF ACTION:**")
        lines.append(f"- Pay off entire Visa balance: **${r.visa_before_payment:.2f}**")
        lines.append(f"- Remaining for post-payoff split: ${r.remaining_after_payoff:.2f}")
    elif r.visa_end == 0:
        # Post-payoff
        line = f"- Balance at payment: $0.00 + ${r.subs_charged:.2f} subs"
        if r.special_charges > 0:
            line += f" + ${r.special_charges:.2f} special"
        line += f" = **${r.visa_before_payment:.2f}**"
        lines.append(line)
        lines.append(f"- Pay off subscription charges: -${r.visa_payment:.2f}")
    else:
        line = f"- Balance at payment: ${r.visa_start:.2f} + ${r.interest:.2f} interest ({r.period.days_since_last} days) + ${r.subs_charged:.2f} subs"
        if r.special_charges > 0:
            line += f" + ${r.special_charges:.2f} special"
        line += f" = ${r.visa_before_payment:.2f}"
        lines.append(line)
        lines.append(f"- Payment: -${r.visa_payment:.2f}")
    
    check = " ✅" if r.visa_end == 0 else ""
    lines.append(f"- **New Balance: ${r.visa_end:.2f}**{check}")
    
    # Wealth
    lines.append(f"\n**💎 Wealth: Nest ${r.nest_end:,.2f} | Prov ${r.prov_end:,.2f} | Total ${r.wealth_total:,.2f}**")
    
    return "\n".join(lines)


def regenerate_markdown(results: CascadeResults, markdown_path: Path = MARKDOWN_PATH) -> bool:
    """Regenerate the cascade section of the markdown file."""
    
    if not markdown_path.exists():
        print(color(f"❌ Markdown file not found: {markdown_path}", Colors.RED))
        return False
    
    content = markdown_path.read_text()
    
    # Check for markers
    if CASCADE_START not in content:
        print(color("❌ CASCADE_START marker not found in markdown", Colors.RED))
        print(f"   Add '{CASCADE_START}' and '{CASCADE_END}' markers to the file")
        return False
    
    if CASCADE_END not in content:
        print(color("❌ CASCADE_END marker not found in markdown", Colors.RED))
        return False
    
    # Generate new content
    cascade_sections = []
    current_month = None
    
    for r in results.periods:
        month_name = r.period.date_str.split()[0]
        
        # Add month header if new month
        if month_name != current_month:
            if current_month:
                cascade_sections.append("---")
            cascade_sections.append(f"\n## {month_name} 2026\n")
            current_month = month_name
        
        cascade_sections.append(generate_period_markdown(r))
        cascade_sections.append("\n---\n")
    
    new_cascade = "\n".join(cascade_sections)
    
    # Replace content between markers
    pattern = f"{CASCADE_START}.*?{CASCADE_END}"
    replacement = f"{CASCADE_START}\n{new_cascade}\n{CASCADE_END}"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    markdown_path.write_text(new_content)
    
    print(color(f"✅ Regenerated cascade section in {markdown_path}", Colors.GREEN))
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Budget Cascade Calculator - Comprehensive Financial CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Basic:
    %(prog)s                                    # Show summary
    %(prog)s --output cascade                   # Full cascade table
    %(prog)s --output keys                      # Values for manual updates
    %(prog)s --output json                      # JSON output

  Balance Changes:
    %(prog)s --visa-balance 17500               # What-if: new balance
    %(prog)s --visa-balance 17500 --save        # Save to YAML

  What-If Scenarios:
    %(prog)s --extra-visa-payment 500           # Extra payment
    %(prog)s --add-charge 200 "Dinner"          # One-time charge
    %(prog)s --compare --visa-balance 18000     # Compare scenarios

  Subscriptions:
    %(prog)s --add-subscription "New" 29.99 --day 15
    %(prog)s --remove-subscription "Netflix"
    %(prog)s --modify-subscription "Claude MAX" 156.00

  Markdown:
    %(prog)s --regenerate-markdown              # Update markdown file
        """
    )
    
    # Output options
    parser.add_argument('--output', '-o', 
                       choices=['summary', 'cascade', 'json', 'keys'],
                       default='summary', help='Output format')
    
    # Balance overrides
    parser.add_argument('--visa-balance', type=float,
                       help='Override starting Visa balance')
    parser.add_argument('--extra-visa-payment', type=float, default=0,
                       help='Add one-time extra Visa payment')
    
    # Subscription management
    parser.add_argument('--add-subscription', nargs=2, metavar=('NAME', 'AMOUNT'),
                       help='Add new subscription')
    parser.add_argument('--day', type=int, default=15,
                       help='Day of month for new subscription')
    parser.add_argument('--remove-subscription', metavar='NAME',
                       help='Remove subscription (what-if)')
    parser.add_argument('--modify-subscription', nargs=2, metavar=('NAME', 'AMOUNT'),
                       help='Modify subscription amount')
    
    # One-time events
    parser.add_argument('--add-charge', nargs=2, metavar=('AMOUNT', 'DESC'),
                       help='Add one-time Visa charge')
    parser.add_argument('--add-income', nargs=2, metavar=('AMOUNT', 'DESC'),
                       help='Add one-time income')
    parser.add_argument('--date', default=datetime.now().strftime("2026-%m-%d"),
                       help='Date for charge/income (YYYY-MM-DD)')
    parser.add_argument('--allocation', choices=['visa', 'split'], default='split',
                       help='Income allocation: visa (direct) or split')
    
    # Compare mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare current vs scenario')
    
    # Markdown
    parser.add_argument('--regenerate-markdown', action='store_true',
                       help='Regenerate markdown cascade section')
    
    # Persistence
    parser.add_argument('--save', action='store_true',
                       help='Save changes to YAML file')
    parser.add_argument('--yaml', type=str, default=str(YAML_PATH),
                       help='Path to YAML data file')
    
    # Misc
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Load data
    yaml_path = Path(args.yaml)
    data = load_yaml_data(yaml_path)
    config = BudgetConfig(data)
    
    # Track if we're doing a what-if
    scenario_desc = None
    modified = False
    
    # Apply modifications
    if args.add_subscription:
        name, amount = args.add_subscription
        config.add_subscription(name, float(amount), args.day)
        scenario_desc = f"Add subscription: {name} ${amount}/mo on day {args.day}"
        modified = True
        print(color(f"➕ Added subscription: {name} ${amount}/mo", Colors.GREEN))
    
    if args.remove_subscription:
        name = args.remove_subscription
        if config.remove_subscription(name):
            scenario_desc = f"Remove subscription: {name}"
            modified = True
            print(color(f"➖ Removed subscription: {name}", Colors.GREEN))
        else:
            print(color(f"❌ Subscription not found: {name}", Colors.RED))
            return
    
    if args.modify_subscription:
        name, amount = args.modify_subscription
        if config.modify_subscription(name, float(amount)):
            scenario_desc = f"Modify {name} to ${amount}/mo"
            modified = True
            print(color(f"✏️  Modified subscription: {name} → ${amount}/mo", Colors.GREEN))
        else:
            print(color(f"❌ Subscription not found: {name}", Colors.RED))
            return
    
    if args.add_charge:
        amount, desc = args.add_charge
        config.add_one_time_charge(float(amount), desc, args.date)
        scenario_desc = f"Add charge: {desc} ${amount} on {args.date}"
        modified = True
        print(color(f"💳 Added charge: {desc} ${amount}", Colors.GREEN))
    
    if args.add_income:
        amount, desc = args.add_income
        config.add_income_event(float(amount), desc, args.date, args.allocation)
        alloc_str = "→ Visa" if args.allocation == 'visa' else "(split)"
        scenario_desc = f"Add income: {desc} ${amount} {alloc_str}"
        modified = True
        print(color(f"💰 Added income: {desc} ${amount} {alloc_str}", Colors.GREEN))
    
    # Determine starting balance
    starting_visa = args.visa_balance if args.visa_balance else None
    if starting_visa:
        scenario_desc = scenario_desc or f"Visa balance: {fmt(starting_visa)}"
    
    if args.extra_visa_payment > 0:
        scenario_desc = scenario_desc or f"Extra payment: {fmt(args.extra_visa_payment)}"
    
    # Compare mode
    if args.compare:
        print(color("\n🔄 Calculating comparison...", Colors.CYAN))
        
        # Current scenario
        current_config = BudgetConfig(load_yaml_data(yaml_path))
        current = calculate_cascade(current_config)
        
        # Modified scenario
        scenario = calculate_cascade(config, starting_visa, args.extra_visa_payment)
        
        print_compare(current, scenario, scenario_desc or "Custom scenario")
        return
    
    # Calculate
    results = calculate_cascade(config, starting_visa, args.extra_visa_payment)
    
    # Output
    if args.output == 'summary':
        print(f"\n🏦 {color('BUDGET CASCADE CALCULATOR', Colors.BOLD)}")
        print(f"   Starting Visa: {fmt(results.starting_visa)}")
        if args.extra_visa_payment > 0:
            print(f"   Extra Payment: {fmt(args.extra_visa_payment)}")
        if scenario_desc:
            print(f"   Scenario: {scenario_desc}")
        print_summary(results)
    elif args.output == 'cascade':
        print_cascade(results)
    elif args.output == 'keys':
        print_keys(results)
    elif args.output == 'json':
        print_json(results)
    
    # Regenerate markdown
    if args.regenerate_markdown:
        regenerate_markdown(results)
    
    # Save if requested
    if args.save:
        if args.visa_balance:
            data['balances']['visa']['effective'] = args.visa_balance
        if modified:
            # Already modified in-place via config
            pass
        data['metadata']['notes'] = scenario_desc or f"Updated {datetime.now()}"
        save_yaml_data(data, yaml_path)


if __name__ == "__main__":
    main()
