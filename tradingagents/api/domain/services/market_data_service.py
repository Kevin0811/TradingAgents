"""Market data service - provides access to market data and indicators."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
)

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for accessing market data, indicators, and fundamentals."""

    def get_stock_data(self, ticker: str, trade_date: str, period: int = 365) -> str:
        """Fetch OHLCV stock data.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.
            period: Days of historical data.

        Returns:
            CSV-formatted stock data.
        """
        return get_stock_data(ticker, trade_date, period=period)

    def get_indicators(
        self,
        ticker: str,
        trade_date: str,
        indicator_names: list[str] | None = None,
    ) -> str:
        """Fetch technical indicators.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.
            indicator_names: List of indicator names.

        Returns:
            CSV-formatted indicator data.
        """
        stock_data = get_stock_data(ticker, trade_date)
        if indicator_names:
            return get_indicators(stock_data, indicator_names=indicator_names)
        return get_indicators(stock_data)

    def get_fundamentals(
        self,
        ticker: str,
        trade_date: str,
        report_type: str = "all",
    ) -> dict | str:
        """Fetch fundamental data.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.
            report_type: Type of fundamental data.

        Returns:
            Fundamental data dict or string.
        """
        if report_type == "all":
            return {
                "fundamentals": get_fundamentals(ticker, trade_date),
                "balance_sheet": get_balance_sheet(ticker, trade_date),
                "cashflow": get_cashflow(ticker, trade_date),
                "income_statement": get_income_statement(ticker, trade_date),
            }
        elif report_type == "balance_sheet":
            return get_balance_sheet(ticker, trade_date)
        elif report_type == "cashflow":
            return get_cashflow(ticker, trade_date)
        elif report_type == "income_statement":
            return get_income_statement(ticker, trade_date)
        else:
            return get_fundamentals(ticker, trade_date)

    def get_news(
        self,
        ticker: str,
        trade_date: str,
        country: str | None = None,
    ) -> list[dict] | str:
        """Fetch news data.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.
            country: Optional country code for global news.

        Returns:
            News items list or formatted string.
        """
        if country:
            return get_global_news(country=country, date=trade_date)
        return get_news(ticker=ticker, date=trade_date)

    def get_macro_indicators(self, country: str = "US") -> str:
        """Fetch macro indicators for a country.

        Args:
            country: Country code.

        Returns:
            Macro indicator data string.
        """
        return get_macro_indicators(country=country)

    def get_prediction_markets(self, ticker: str, trade_date: str) -> str:
        """Fetch prediction market data.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.

        Returns:
            Prediction market data string.
        """
        return get_prediction_markets(ticker=ticker, date=trade_date)