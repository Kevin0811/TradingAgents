"""Enum definitions for TradingAgents API."""

from __future__ import annotations

from enum import StrEnum


class AssetType(StrEnum):
    """Supported asset types for analysis."""

    STOCK = "stock"
    CRYPTO = "crypto"


class AnalystType(StrEnum):
    """Available analyst agents.

    ``social`` is the core's key for the social-media sentiment analyst.
    ``sentiment`` is the API's older name for it, still accepted and mapped to
    ``social`` before a request reaches the core (see
    ``tradingagents.api.domain.analysts``).
    """

    MARKET = "market"
    SOCIAL = "social"
    SENTIMENT = "sentiment"  # alias of SOCIAL
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class ReportType(StrEnum):
    """Type of fundamental data to retrieve."""

    ALL = "all"
    OVERVIEW = "overview"
    BALANCE_SHEET = "balance_sheet"
    CASHFLOW = "cashflow"
    INCOME_STATEMENT = "income_statement"


class ReportFreq(StrEnum):
    """Reporting frequency for financial statements."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class IndicatorName(StrEnum):
    """Technical indicator names accepted by the data vendors.

    These are the exact keys both vendor implementations key off — see
    ``best_ind_params`` in ``dataflows/y_finance.py`` and
    ``supported_indicators`` in ``dataflows/alpha_vantage_indicator.py``.
    Anything outside this set is rejected by the vendor, so the enum must
    track it rather than invent friendlier aliases.
    """

    CLOSE_50_SMA = "close_50_sma"
    CLOSE_200_SMA = "close_200_sma"
    CLOSE_10_EMA = "close_10_ema"
    MACD = "macd"
    MACDS = "macds"
    MACDH = "macdh"
    RSI = "rsi"
    BOLL = "boll"
    BOLL_UB = "boll_ub"
    BOLL_LB = "boll_lb"
    ATR = "atr"
    VWMA = "vwma"
    MFI = "mfi"  # yfinance only


class SymbolMarket(StrEnum):
    """Markets covered by the supported-symbols list."""

    TW = "tw"
    US = "us"
    JP = "jp"
    CRYPTO = "crypto"
    FX = "fx"


class SymbolType(StrEnum):
    """Instrument types on the supported-symbols list."""

    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    CURRENCY = "currency"


class SymbolListStatus(StrEnum):
    """Load state of one market's supported-symbols list."""

    READY = "ready"  # list loaded (it may still be stale; see ``stale``)
    LOADING = "loading"  # no list yet; a background fetch is queued or running
    UNAVAILABLE = "unavailable"  # no list and no fetch in progress (e.g. it failed)
    UNCOVERED = "uncovered"  # GET /symbols/check only: no list covers the ticker
