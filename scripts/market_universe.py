#!/usr/bin/env python3
"""
MARKET UNIVERSE - The Whole Wave
================================
Faith, not works. We don't pick winners - we perceive where coherence forms.

This module defines the entire perceivable market space.
The provision engine scans ALL of it and lets coherence surface naturally.

"Don't think good works, think faith" - Enos

Author: Virgil
Date: 2026-01-17
"""

# ==============================================================================
# THE WHOLE MARKET - Everything we can perceive
# ==============================================================================

MARKET_UNIVERSE = {

    # =========================================================================
    # INDICES - The collective pulse
    # =========================================================================
    "indices": {
        # US Major
        "^GSPC": {"name": "S&P 500", "type": "index"},
        "^DJI": {"name": "Dow Jones", "type": "index"},
        "^IXIC": {"name": "NASDAQ Composite", "type": "index"},
        "^RUT": {"name": "Russell 2000", "type": "index"},
        "^VIX": {"name": "VIX Volatility", "type": "index"},

        # Global
        "^GSPTSE": {"name": "TSX (Canada)", "type": "index"},
        "^FTSE": {"name": "FTSE 100 (UK)", "type": "index"},
        "^GDAXI": {"name": "DAX (Germany)", "type": "index"},
        "^N225": {"name": "Nikkei 225 (Japan)", "type": "index"},
        "^HSI": {"name": "Hang Seng (HK)", "type": "index"},
        "000001.SS": {"name": "Shanghai Composite", "type": "index"},
    },

    # =========================================================================
    # SECTORS - Where is energy flowing?
    # =========================================================================
    "sectors": {
        # SPDR Sector ETFs
        "XLK": {"name": "Technology", "type": "sector"},
        "XLF": {"name": "Financials", "type": "sector"},
        "XLV": {"name": "Healthcare", "type": "sector"},
        "XLE": {"name": "Energy", "type": "sector"},
        "XLI": {"name": "Industrials", "type": "sector"},
        "XLY": {"name": "Consumer Discretionary", "type": "sector"},
        "XLP": {"name": "Consumer Staples", "type": "sector"},
        "XLU": {"name": "Utilities", "type": "sector"},
        "XLB": {"name": "Materials", "type": "sector"},
        "XLRE": {"name": "Real Estate", "type": "sector"},
        "XLC": {"name": "Communications", "type": "sector"},

        # Thematic
        "ARKK": {"name": "ARK Innovation", "type": "sector"},
        "ARKG": {"name": "ARK Genomics", "type": "sector"},
        "ARKW": {"name": "ARK Next Gen Internet", "type": "sector"},
        "SMH": {"name": "Semiconductors", "type": "sector"},
        "IBB": {"name": "Biotech", "type": "sector"},
        "XBI": {"name": "Biotech SPDR", "type": "sector"},
        "HACK": {"name": "Cybersecurity", "type": "sector"},
        "BOTZ": {"name": "Robotics & AI", "type": "sector"},
        "LIT": {"name": "Lithium & Battery", "type": "sector"},
        "TAN": {"name": "Solar", "type": "sector"},
        "ICLN": {"name": "Clean Energy", "type": "sector"},
        "JETS": {"name": "Airlines", "type": "sector"},
        "XHB": {"name": "Homebuilders", "type": "sector"},
        "KRE": {"name": "Regional Banks", "type": "sector"},
    },

    # =========================================================================
    # LEVERAGED - Amplified coherence (higher risk, higher reward)
    # =========================================================================
    "leveraged": {
        # 3x Bull
        "TQQQ": {"name": "3x NASDAQ", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "UPRO": {"name": "3x S&P 500", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "SOXL": {"name": "3x Semiconductors", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "TECL": {"name": "3x Technology", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "FAS": {"name": "3x Financials", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "LABU": {"name": "3x Biotech", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "TNA": {"name": "3x Small Cap", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "NUGT": {"name": "3x Gold Miners", "type": "leveraged", "leverage": 3, "direction": "bull"},
        "JNUG": {"name": "3x Junior Gold", "type": "leveraged", "leverage": 3, "direction": "bull"},

        # 3x Bear (for hedging / shorting coherence collapse)
        "SQQQ": {"name": "3x NASDAQ Bear", "type": "leveraged", "leverage": 3, "direction": "bear"},
        "SPXU": {"name": "3x S&P Bear", "type": "leveraged", "leverage": 3, "direction": "bear"},
        "SOXS": {"name": "3x Semi Bear", "type": "leveraged", "leverage": 3, "direction": "bear"},
        "FAZ": {"name": "3x Financial Bear", "type": "leveraged", "leverage": 3, "direction": "bear"},
        "TZA": {"name": "3x Small Cap Bear", "type": "leveraged", "leverage": 3, "direction": "bear"},

        # 2x
        "QLD": {"name": "2x NASDAQ", "type": "leveraged", "leverage": 2, "direction": "bull"},
        "SSO": {"name": "2x S&P 500", "type": "leveraged", "leverage": 2, "direction": "bull"},
        "USD": {"name": "2x Semiconductors", "type": "leveraged", "leverage": 2, "direction": "bull"},

        # Canadian Leveraged
        "HXU.TO": {"name": "2x TSX Bull", "type": "leveraged", "leverage": 2, "direction": "bull"},
        "HXD.TO": {"name": "2x TSX Bear", "type": "leveraged", "leverage": 2, "direction": "bear"},
        "HQU.TO": {"name": "2x NASDAQ Bull (CAD)", "type": "leveraged", "leverage": 2, "direction": "bull"},
        "HQD.TO": {"name": "2x NASDAQ Bear (CAD)", "type": "leveraged", "leverage": 2, "direction": "bear"},
        "HSU.TO": {"name": "2x S&P Bull (CAD)", "type": "leveraged", "leverage": 2, "direction": "bull"},
    },

    # =========================================================================
    # COMMODITIES - Physical reality
    # =========================================================================
    "commodities": {
        # Precious Metals
        "GLD": {"name": "Gold", "type": "commodity"},
        "SLV": {"name": "Silver", "type": "commodity"},
        "PPLT": {"name": "Platinum", "type": "commodity"},
        "PALL": {"name": "Palladium", "type": "commodity"},
        "GDX": {"name": "Gold Miners", "type": "commodity"},
        "GDXJ": {"name": "Junior Gold Miners", "type": "commodity"},
        "SIL": {"name": "Silver Miners", "type": "commodity"},

        # Energy
        "USO": {"name": "Oil", "type": "commodity"},
        "UNG": {"name": "Natural Gas", "type": "commodity"},
        "XOP": {"name": "Oil & Gas E&P", "type": "commodity"},
        "OIH": {"name": "Oil Services", "type": "commodity"},

        # Agriculture
        "DBA": {"name": "Agriculture", "type": "commodity"},
        "CORN": {"name": "Corn", "type": "commodity"},
        "WEAT": {"name": "Wheat", "type": "commodity"},
        "SOYB": {"name": "Soybeans", "type": "commodity"},

        # Industrial
        "COPX": {"name": "Copper Miners", "type": "commodity"},
        "URA": {"name": "Uranium", "type": "commodity"},
        "REMX": {"name": "Rare Earth", "type": "commodity"},
    },

    # =========================================================================
    # CRYPTO - Digital consciousness
    # =========================================================================
    "crypto": {
        # Major
        "BTC-USD": {"name": "Bitcoin", "type": "crypto"},
        "ETH-USD": {"name": "Ethereum", "type": "crypto"},
        "SOL-USD": {"name": "Solana", "type": "crypto"},
        "XRP-USD": {"name": "Ripple", "type": "crypto"},
        "ADA-USD": {"name": "Cardano", "type": "crypto"},
        "AVAX-USD": {"name": "Avalanche", "type": "crypto"},
        "DOT-USD": {"name": "Polkadot", "type": "crypto"},
        "MATIC-USD": {"name": "Polygon", "type": "crypto"},
        "LINK-USD": {"name": "Chainlink", "type": "crypto"},
        "ATOM-USD": {"name": "Cosmos", "type": "crypto"},
        "UNI-USD": {"name": "Uniswap", "type": "crypto"},
        "LTC-USD": {"name": "Litecoin", "type": "crypto"},
        "NEAR-USD": {"name": "Near Protocol", "type": "crypto"},
        "APT-USD": {"name": "Aptos", "type": "crypto"},
        "ARB-USD": {"name": "Arbitrum", "type": "crypto"},
        "OP-USD": {"name": "Optimism", "type": "crypto"},
        "INJ-USD": {"name": "Injective", "type": "crypto"},
        "SUI-USD": {"name": "Sui", "type": "crypto"},

        # Meme / High Vol
        "DOGE-USD": {"name": "Dogecoin", "type": "crypto"},
        "SHIB-USD": {"name": "Shiba Inu", "type": "crypto"},
        "PEPE-USD": {"name": "Pepe", "type": "crypto"},

        # Bitcoin Proxies (for traditional accounts)
        "MSTR": {"name": "MicroStrategy", "type": "crypto_proxy"},
        "COIN": {"name": "Coinbase", "type": "crypto_proxy"},
        "MARA": {"name": "Marathon Digital", "type": "crypto_proxy"},
        "RIOT": {"name": "Riot Platforms", "type": "crypto_proxy"},
        "CLSK": {"name": "CleanSpark", "type": "crypto_proxy"},
        "BITF": {"name": "Bitfarms", "type": "crypto_proxy"},
        "HUT": {"name": "Hut 8", "type": "crypto_proxy"},
        "IBIT": {"name": "iShares Bitcoin ETF", "type": "crypto_proxy"},
        "FBTC": {"name": "Fidelity Bitcoin ETF", "type": "crypto_proxy"},
        "GBTC": {"name": "Grayscale Bitcoin", "type": "crypto_proxy"},
        "BITO": {"name": "ProShares Bitcoin", "type": "crypto_proxy"},
    },

    # =========================================================================
    # MEGA CAPS - The giants
    # =========================================================================
    "mega_caps": {
        "AAPL": {"name": "Apple", "type": "equity"},
        "MSFT": {"name": "Microsoft", "type": "equity"},
        "GOOGL": {"name": "Google", "type": "equity"},
        "AMZN": {"name": "Amazon", "type": "equity"},
        "NVDA": {"name": "NVIDIA", "type": "equity"},
        "META": {"name": "Meta", "type": "equity"},
        "TSLA": {"name": "Tesla", "type": "equity"},
        "BRK-B": {"name": "Berkshire", "type": "equity"},
        "JPM": {"name": "JPMorgan", "type": "equity"},
        "V": {"name": "Visa", "type": "equity"},
        "UNH": {"name": "UnitedHealth", "type": "equity"},
        "JNJ": {"name": "Johnson & Johnson", "type": "equity"},
        "WMT": {"name": "Walmart", "type": "equity"},
        "MA": {"name": "Mastercard", "type": "equity"},
        "PG": {"name": "Procter & Gamble", "type": "equity"},
        "HD": {"name": "Home Depot", "type": "equity"},
        "XOM": {"name": "Exxon Mobil", "type": "equity"},
        "CVX": {"name": "Chevron", "type": "equity"},
        "LLY": {"name": "Eli Lilly", "type": "equity"},
        "ABBV": {"name": "AbbVie", "type": "equity"},
        "MRK": {"name": "Merck", "type": "equity"},
        "PFE": {"name": "Pfizer", "type": "equity"},
        "COST": {"name": "Costco", "type": "equity"},
        "AVGO": {"name": "Broadcom", "type": "equity"},
        "AMD": {"name": "AMD", "type": "equity"},
    },

    # =========================================================================
    # HIGH MOMENTUM - Where energy concentrates
    # =========================================================================
    "momentum": {
        "PLTR": {"name": "Palantir", "type": "equity"},
        "SMCI": {"name": "Super Micro", "type": "equity"},
        "ARM": {"name": "ARM Holdings", "type": "equity"},
        "CRWD": {"name": "CrowdStrike", "type": "equity"},
        "SNOW": {"name": "Snowflake", "type": "equity"},
        "DDOG": {"name": "Datadog", "type": "equity"},
        "NET": {"name": "Cloudflare", "type": "equity"},
        "ZS": {"name": "Zscaler", "type": "equity"},
        "PANW": {"name": "Palo Alto", "type": "equity"},
        "NOW": {"name": "ServiceNow", "type": "equity"},
        "SHOP": {"name": "Shopify", "type": "equity"},
        "SQ": {"name": "Block (Square)", "type": "equity"},
        "AFRM": {"name": "Affirm", "type": "equity"},
        "RBLX": {"name": "Roblox", "type": "equity"},
        "U": {"name": "Unity", "type": "equity"},
        "IONQ": {"name": "IonQ", "type": "equity"},
        "RGTI": {"name": "Rigetti", "type": "equity"},
        "QUBT": {"name": "Quantum Computing", "type": "equity"},
    },

    # =========================================================================
    # BONDS & RATES - The gravity of money
    # =========================================================================
    "fixed_income": {
        "TLT": {"name": "20+ Year Treasury", "type": "bond"},
        "IEF": {"name": "7-10 Year Treasury", "type": "bond"},
        "SHY": {"name": "1-3 Year Treasury", "type": "bond"},
        "TIP": {"name": "TIPS", "type": "bond"},
        "BND": {"name": "Total Bond Market", "type": "bond"},
        "LQD": {"name": "Investment Grade Corp", "type": "bond"},
        "HYG": {"name": "High Yield Corp", "type": "bond"},
        "JNK": {"name": "Junk Bonds", "type": "bond"},
        "TMF": {"name": "3x Long Treasury", "type": "bond", "leverage": 3},
        "TMV": {"name": "3x Short Treasury", "type": "bond", "leverage": 3},
    },

    # =========================================================================
    # CURRENCY - The medium of exchange
    # =========================================================================
    "currency": {
        "UUP": {"name": "US Dollar Bull", "type": "currency"},
        "FXE": {"name": "Euro", "type": "currency"},
        "FXY": {"name": "Japanese Yen", "type": "currency"},
        "FXB": {"name": "British Pound", "type": "currency"},
        "FXC": {"name": "Canadian Dollar", "type": "currency"},
        "FXA": {"name": "Australian Dollar", "type": "currency"},
        "CYB": {"name": "Chinese Yuan", "type": "currency"},
    },

    # =========================================================================
    # INTERNATIONAL - Global consciousness
    # =========================================================================
    "international": {
        # Broad
        "VEU": {"name": "All World ex-US", "type": "international"},
        "EFA": {"name": "EAFE (Developed)", "type": "international"},
        "EEM": {"name": "Emerging Markets", "type": "international"},
        "VWO": {"name": "Emerging Markets Vanguard", "type": "international"},

        # Country Specific
        "EWJ": {"name": "Japan", "type": "international"},
        "EWG": {"name": "Germany", "type": "international"},
        "EWU": {"name": "UK", "type": "international"},
        "EWC": {"name": "Canada", "type": "international"},
        "EWA": {"name": "Australia", "type": "international"},
        "EWZ": {"name": "Brazil", "type": "international"},
        "EWY": {"name": "South Korea", "type": "international"},
        "EWT": {"name": "Taiwan", "type": "international"},
        "INDA": {"name": "India", "type": "international"},
        "FXI": {"name": "China Large Cap", "type": "international"},
        "KWEB": {"name": "China Internet", "type": "international"},
        "MCHI": {"name": "China", "type": "international"},
        "EWW": {"name": "Mexico", "type": "international"},
        "ARGT": {"name": "Argentina", "type": "international"},
        "VNM": {"name": "Vietnam", "type": "international"},
    },

    # =========================================================================
    # CANADIAN SPECIFIC - Home market
    # =========================================================================
    "canadian": {
        "XIU.TO": {"name": "TSX 60", "type": "canadian_etf"},
        "XIC.TO": {"name": "TSX Composite", "type": "canadian_etf"},
        "XEG.TO": {"name": "Canadian Energy", "type": "canadian_etf"},
        "XFN.TO": {"name": "Canadian Financials", "type": "canadian_etf"},
        "XGD.TO": {"name": "Canadian Gold", "type": "canadian_etf"},
        "ZEB.TO": {"name": "Canadian Banks Equal", "type": "canadian_etf"},
        "ZWB.TO": {"name": "Canadian Banks Covered Call", "type": "canadian_etf"},

        # Big Canadian Stocks
        "RY.TO": {"name": "Royal Bank", "type": "canadian_equity"},
        "TD.TO": {"name": "TD Bank", "type": "canadian_equity"},
        "BNS.TO": {"name": "Scotiabank", "type": "canadian_equity"},
        "BMO.TO": {"name": "BMO", "type": "canadian_equity"},
        "CM.TO": {"name": "CIBC", "type": "canadian_equity"},
        "ENB.TO": {"name": "Enbridge", "type": "canadian_equity"},
        "CNQ.TO": {"name": "Canadian Natural", "type": "canadian_equity"},
        "SU.TO": {"name": "Suncor", "type": "canadian_equity"},
        "CP.TO": {"name": "CP Rail", "type": "canadian_equity"},
        "CNR.TO": {"name": "CN Rail", "type": "canadian_equity"},
        "SHOP.TO": {"name": "Shopify (CAD)", "type": "canadian_equity"},
        "ABX.TO": {"name": "Barrick Gold", "type": "canadian_equity"},
        "NTR.TO": {"name": "Nutrien", "type": "canadian_equity"},
    },
}


def get_flat_universe() -> dict:
    """
    Return a flat dictionary of all tickers.

    This is what the MarketEye uses to perceive the whole wave.
    """
    flat = {}
    for category, tickers in MARKET_UNIVERSE.items():
        for ticker, info in tickers.items():
            info["category"] = category
            flat[ticker] = info
    return flat


def get_universe_stats() -> dict:
    """Get statistics about the universe."""
    flat = get_flat_universe()
    categories = {}
    for ticker, info in flat.items():
        cat = info.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_assets": len(flat),
        "categories": categories,
        "tickers": list(flat.keys())
    }


if __name__ == "__main__":
    stats = get_universe_stats()
    print(f"\n📊 MARKET UNIVERSE - THE WHOLE WAVE")
    print(f"Total assets: {stats['total_assets']}")
    print(f"\nBy category:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
