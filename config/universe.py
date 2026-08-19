"""
Phase 1 instrument universe.

Kept deliberately small (16 instruments) and cross-asset per the blueprint's
Phase 1 requirement: prove the pipeline before expanding breadth.

Each entry carries the metadata needed downstream (asset class, yfinance
ticker, display name, minimum history requirement) so the ingestion and
eligibility logic never has to hardcode ticker lists again.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    asset_class: str
    min_history_days: int


UNIVERSE: list[Instrument] = [
    # Stocks
    Instrument("AAPL", "Apple", "stock", 500),
    Instrument("MSFT", "Microsoft", "stock", 500),
    Instrument("NVDA", "Nvidia", "stock", 500),
    # Added for Model 7 (equity momentum) cross-sector robustness testing -
    # AAPL/MSFT/NVDA are all large-cap tech and likely correlated with each
    # other; these three are deliberately from different sectors (financials,
    # energy, healthcare) to test whether momentum holds up on names not
    # driven by the same tech-sector factor
    Instrument("JPM", "JPMorgan Chase", "stock", 500),
    Instrument("XOM", "Exxon Mobil", "stock", 500),
    Instrument("JNJ", "Johnson & Johnson", "stock", 500),

    # Indices (via liquid ETF proxies)
    Instrument("SPY", "S&P 500 ETF", "index", 500),
    Instrument("QQQ", "Nasdaq 100 ETF", "index", 500),

    # FX
    Instrument("EURUSD=X", "EUR/USD", "fx", 500),
    Instrument("USDJPY=X", "USD/JPY", "fx", 500),
    Instrument("GBPUSD=X", "GBP/USD", "fx", 500),
    Instrument("AUDUSD=X", "AUD/USD", "fx", 500),

    # Commodities (continuous futures)
    Instrument("GC=F", "Gold", "commodity", 500),
    Instrument("SI=F", "Silver", "commodity", 500),
    Instrument("CL=F", "Crude Oil WTI", "commodity", 500),
    Instrument("HG=F", "Copper", "commodity", 500),

    # Crypto (shorter history requirement - younger asset class)
    Instrument("BTC-USD", "Bitcoin", "crypto", 365),
    Instrument("ETH-USD", "Ethereum", "crypto", 365),
    Instrument("SOL-USD", "Solana", "crypto", 365),
]


def tickers() -> list[str]:
    return [i.ticker for i in UNIVERSE]


def by_asset_class(asset_class: str) -> list[Instrument]:
    return [i for i in UNIVERSE if i.asset_class == asset_class]


def get(ticker: str) -> Instrument:
    for i in UNIVERSE:
        if i.ticker == ticker:
            return i
    raise KeyError(f"{ticker} not in universe")