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
    BALANCE_SHEET = "balance_sheet"
    CASHFLOW = "cashflow"
    INCOME_STATEMENT = "income_statement"


class Country(str, Enum):
    """Country codes for global news and macro indicators."""

    US = "US"
    CN = "CN"
    JP = "JP"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    TW = "TW"
    KR = "KR"
    SG = "SG"


class IndicatorName(str, Enum):
    """Common technical indicator names."""

    RSI = "rsi"
    MACD = "macd"
    SIGNAL = "signal"
    HISTOGRAM = "histogram"
    CLOSE_50_SMA = "close_50_sma"
    CLOSE_200_SMA = "close_200_sma"
    CLOSE_10_SMA = "close_10_sma"
    CLOSE_20_SMA = "close_20_sma"
    BOLLINGER_UPPER = "bollinger_upper"
    BOLLINGER_MIDDLE = "bollinger_middle"
    BOLLINGER_LOWER = "bollinger_lower"
    VOLUME = "volume"
    ATR = "atr"
    ADX = "adx"
    STOCH_K = "stoch_k"
    STOCH_D = "stoch_d"