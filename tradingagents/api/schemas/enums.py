"""Enum definitions for TradingAgents API."""

from __future__ import annotations

from enum import Enum


class AssetType(str, Enum):
    """Supported asset types for analysis."""

    STOCK = "stock"
    CRYPTO = "crypto"


class AnalystType(str, Enum):
    """Available analyst agents."""

    MARKET = "market"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class ReportType(str, Enum):
    """Type of fundamental data to retrieve."""

    ALL = "all"
    OVERVIEW = "overview"
    BALANCE_SHEET = "balance_sheet"
    CASHFLOW = "cashflow"
    INCOME_STATEMENT = "income_statement"


class ReportFreq(str, Enum):
    """Reporting frequency for financial statements."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class IndicatorName(str, Enum):
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
