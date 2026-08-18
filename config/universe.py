"""
Phase 1 instrument universe.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    asset_class: str
    min_history_days: int


UNIVERSE: list[Instrument] = [
    Instrument("AAPL", "Apple", "stock", 500),
    Instrument("MSFT", "Microsoft", "stock", 500),
    Instrument("NVDA", "Nvidia", "stock", 500),
    Instrument("SPY", "S&P 500 ETF", "index", 500),
    Instrument("QQQ", "Nasdaq 100 ETF", "index", 500),
    Instrument("EURUSD=X", "EUR/USD", "fx", 500),
    Instrument("USDJPY=X", "USD/JPY", "fx", 500),
    Instrument("GBPUSD=X", "GBP/USD", "fx", 500),
    Instrument("AUDUSD=X", "AUD/USD", "fx", 500),
    Instrument("GC=F", "Gold", "commodity", 500),
    Instrument("SI=F", "Silver", "commodity", 500),
    Instrument("CL=F", "Crude Oil WTI", "commodity", 500),
    Instrument("HG=F", "Copper", "commodity", 500),
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