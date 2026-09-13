"""Market data service - provides access to market data and indicators.

The data functions exposed by ``agent_utils`` are LangChain ``StructuredTool``
objects (``@tool``-decorated), not plain callables, so every call here goes
through ``.invoke({...})``. That is the documented entry point, it validates
arguments against each tool's own schema, and it preserves wrapper behaviour
such as the comma-separated indicator splitting in ``get_indicators``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

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
from tradingagents.api.core.exceptions import (
    DataNotFoundError,
    ExternalServiceError,
    InvalidRequestError,
)
from tradingagents.api.domain.vendor_reports import (
    parse_indicator_report,
    parse_ohlcv_csv,
)
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.interface import NO_DATA_SENTINEL, UNAVAILABLE_SENTINEL
from tradingagents.dataflows.symbol_utils import bare_crypto_to_pair, normalize_symbol

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"

# Used when the caller does not name any indicator: one trend, one momentum,
# one volatility and one band measure, so a bare request is still informative
# without fanning out into a dozen vendor calls.
DEFAULT_INDICATORS = ("close_50_sma", "close_200_sma", "macd", "rsi", "boll", "atr")

# The chart default, kept separate from DEFAULT_INDICATORS. That set answers
# "tell me something useful about this symbol" in as few vendor calls as
# possible; this one has to *draw*, so it carries the Bollinger envelope and
# the MACD signal/histogram that a plot needs and a summary does not.
CHART_INDICATORS = (
    "close_50_sma",
    "close_200_sma",
    "boll",
    "boll_ub",
    "boll_lb",
    "rsi",
    "macd",
    "macds",
    "macdh",
)

# Indicators are computed from ``stockstats_utils.load_ohlcv``, which downloads a
# fixed five-year window. Asking for a longer history would silently return bars
# with no indicator values attached, so the window is refused instead.
MAX_HISTORY_DAYS = 1825

DEFAULT_NEWS_LOOKBACK_DAYS = 7


class MarketDataService:
    """Service for accessing market data, indicators, and fundamentals."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ticker(ticker: str) -> str:
        """Resolve an API-level ticker to the symbol the vendor is queried with.

        Applies the bare-crypto convenience (``BTC`` -> ``BTC-USD``) that
        ``normalize_symbol`` deliberately withholds, then the shared broker /
        forex / crypto mapping. Running ``normalize_symbol`` here as well as
        downstream is harmless — it is idempotent and purely syntactic — and
        lets the response report the symbol actually queried.
        """
        return normalize_symbol(bare_crypto_to_pair(ticker) or ticker)

    @staticmethod
    def _parse_date(value: str, field: str = "trade_date") -> datetime:
        try:
            return datetime.strptime(value, DATE_FORMAT)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(
                f"Invalid {field}",
                detail=f"Expected YYYY-MM-DD, got {value!r}",
            ) from exc

    @classmethod
    def _window(cls, end_date: str, days_back: int) -> tuple[str, str]:
        """Return ``(start_date, end_date)`` for a trailing window.

        ``days_back`` counts backwards from ``end_date`` exclusive of the
        endpoint itself, so a caller asking for 0 gets a single-day window.
        """
        end_dt = cls._parse_date(end_date)
        start_dt = end_dt - timedelta(days=max(days_back, 0))
        return start_dt.strftime(DATE_FORMAT), end_dt.strftime(DATE_FORMAT)

    @staticmethod
    def _call(tool, payload: dict):
        """Invoke a data tool, translating vendor failures into API errors.

        Failures reach us two ways. Some propagate as exceptions; but the
        routing layer also *returns* prose sentinels for the two cases it
        expects an agent to read and keep going (no data for the symbol, and an
        optional category that degraded). An HTTP caller needs a status code
        instead, so both are turned back into typed errors here — otherwise
        every one of them is a 200 carrying an apology.
        """
        try:
            result = tool.invoke(payload)
        except NoMarketDataError as exc:
            raise DataNotFoundError("No market data available", detail=str(exc)) from exc
        except (VendorNotConfiguredError, VendorRateLimitError) as exc:
            # A server-side vendor problem, not something the caller can fix.
            # Checked before ValueError: VendorNotConfiguredError is also one.
            raise ExternalServiceError("Data vendor unavailable", detail=str(exc)) from exc
        except ValueError as exc:
            # Unsupported indicator name, unknown method, bad vendor arguments.
            raise InvalidRequestError("Invalid data request", detail=str(exc)) from exc

        if isinstance(result, str):
            text = result.lstrip()
            if text.startswith(NO_DATA_SENTINEL):
                raise DataNotFoundError("No market data available", detail=text)
            if text.startswith(UNAVAILABLE_SENTINEL):
                raise ExternalServiceError("Data vendor unavailable", detail=text)
        return result

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_stock_data(self, ticker: str, trade_date: str, period: int = 365) -> dict:
        """Fetch OHLCV stock data as records.

        Args:
            ticker: Ticker symbol.
            trade_date: End of the window, YYYY-MM-DD.
            period: Window length in calendar days, inclusive of ``trade_date``
                (so ``period=1`` requests that single day).

        Returns:
            Dict with the resolved symbol, the window, and a ``rows`` list of
            price observations. The vendor's text report is parsed here so the
            HTTP caller gets data instead of a CSV blob.
        """
        symbol = self._resolve_ticker(ticker)
        start_date, end_date = self._window(trade_date, period - 1)
        report = self._call(
            get_stock_data,
            {"symbol": symbol, "start_date": start_date, "end_date": end_date},
        )
        rows = parse_ohlcv_csv(report)
        if not rows:
            # The vendors raise NoMarketDataError for an empty result, so an
            # unparsable report means the format moved out from under us.
            # Surfacing it as "no data" would hide a real defect.
            raise ExternalServiceError(
                "Unreadable vendor response",
                detail=f"No price rows could be parsed for {symbol}",
            )
        return {
            "ticker": ticker,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(rows),
            "rows": rows,
        }

    def get_indicators(
        self,
        ticker: str,
        trade_date: str,
        indicator_names: list[str] | None = None,
        look_back_days: int = 30,
    ) -> dict:
        """Fetch technical indicators as records.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date, YYYY-MM-DD.
            indicator_names: Indicator names; defaults to DEFAULT_INDICATORS.
            look_back_days: How many days of values to return per indicator.

        Returns:
            Dict with one entry per indicator under ``indicators``, plus any
            per-indicator failures under ``errors``.
        """
        symbol = self._resolve_ticker(ticker)
        names = indicator_names or list(DEFAULT_INDICATORS)
        # The tool splits a comma-separated string itself and reports unknown
        # names per-indicator instead of failing the whole request.
        report = self._call(
            get_indicators,
            {
                "symbol": symbol,
                "indicator": ",".join(names),
                "curr_date": trade_date,
                "look_back_days": look_back_days,
            },
        )
        series, errors = parse_indicator_report(report)
        return {
            "ticker": ticker,
            "symbol": symbol,
            "trade_date": trade_date,
            "look_back_days": look_back_days,
            "indicators": series,
            "errors": errors,
        }

    def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        indicator_names: list[str] | None = None,
    ) -> dict:
        """Fetch OHLCV bars and indicator series over one explicit window.

        The two existing endpoints cannot be combined by a caller without
        re-deriving the window: ``get_stock_data`` counts calendar days back
        from ``trade_date``, ``get_indicators`` counts ``look_back_days`` back
        from it, and the two are capped differently. A chart needs both series
        over the *same* dates, so the alignment is done here once rather than in
        every consumer.

        Args:
            ticker: Ticker symbol.
            start_date: Start of the window, YYYY-MM-DD (inclusive).
            end_date: End of the window, YYYY-MM-DD (inclusive).
            indicator_names: Indicator names; ``None`` means CHART_INDICATORS,
                an empty list means bars only. The distinction matters — a
                caller that wants a bare price line should not pay for nine
                vendor calls to discard them.

        Returns:
            Dict with the resolved symbol, the window, ``bars``, ``indicators``
            and any per-indicator failures under ``errors``.
        """
        symbol = self._resolve_ticker(ticker)
        start_dt = self._parse_date(start_date, "start_date")
        end_dt = self._parse_date(end_date, "end_date")

        span = (end_dt - start_dt).days
        if span < 0:
            raise InvalidRequestError(
                "Invalid date range",
                detail=f"start_date {start_date} is after end_date {end_date}",
            )
        if span > MAX_HISTORY_DAYS:
            raise InvalidRequestError(
                "Date range too long",
                detail=(
                    f"Requested {span} days; the indicator source only holds "
                    f"{MAX_HISTORY_DAYS} days of history"
                ),
            )

        report = self._call(
            get_stock_data,
            {"symbol": symbol, "start_date": start_date, "end_date": end_date},
        )
        bars = parse_ohlcv_csv(report)
        if not bars:
            # Same reasoning as get_stock_data: the vendors raise
            # NoMarketDataError for an empty result, so an unparsable report is
            # a format change on our side, not an absent symbol.
            raise ExternalServiceError(
                "Unreadable vendor response",
                detail=f"No price rows could be parsed for {symbol}",
            )

        names = list(CHART_INDICATORS) if indicator_names is None else indicator_names
        series: list[dict] = []
        errors: list[str] = []
        if names:
            indicator_report = self._call(
                get_indicators,
                {
                    "symbol": symbol,
                    "indicator": ",".join(names),
                    # The indicator window walks backwards from curr_date, so
                    # the end of the price window is where it has to start.
                    "curr_date": end_date,
                    "look_back_days": span,
                },
            )
            series, errors = parse_indicator_report(indicator_report)

        return {
            "ticker": ticker,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(bars),
            "bars": bars,
            "indicators": series,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(
        self,
        ticker: str,
        trade_date: str,
        report_type: str = "all",
        freq: str = "quarterly",
    ) -> dict | str:
        """Fetch fundamental data.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date, YYYY-MM-DD.
            report_type: all | overview | balance_sheet | cashflow | income_statement.
            freq: Reporting frequency for the statements: annual | quarterly.

        Returns:
            Fundamental data dict (report_type="all") or string.
        """
        symbol = self._resolve_ticker(ticker)
        statements = {
            "balance_sheet": get_balance_sheet,
            "cashflow": get_cashflow,
            "income_statement": get_income_statement,
        }

        def _statement(tool):
            # freq and curr_date are passed by name: positionally, trade_date
            # would land in the tool's `freq` slot.
            return self._call(
                tool, {"ticker": symbol, "freq": freq, "curr_date": trade_date}
            )

        if report_type == "all":
            return {
                "fundamentals": self._call(
                    get_fundamentals, {"ticker": symbol, "curr_date": trade_date}
                ),
                **{name: _statement(tool) for name, tool in statements.items()},
            }
        if report_type in statements:
            return _statement(statements[report_type])
        return self._call(get_fundamentals, {"ticker": symbol, "curr_date": trade_date})

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        trade_date: str,
        look_back_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    ) -> str:
        """Fetch news for a single ticker.

        Args:
            ticker: Ticker symbol.
            trade_date: End of the window, YYYY-MM-DD.
            look_back_days: How many days before ``trade_date`` to include.

        Returns:
            Formatted news report.
        """
        symbol = self._resolve_ticker(ticker)
        start_date, end_date = self._window(trade_date, look_back_days)
        return self._call(
            get_news,
            {"ticker": symbol, "start_date": start_date, "end_date": end_date},
        )

    def get_global_news(
        self,
        trade_date: str,
        look_back_days: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Fetch global/macro market news.

        Args:
            trade_date: End of the window, YYYY-MM-DD.
            look_back_days: Window length; None inherits the configured default.
            limit: Max articles; None inherits the configured default.

        Returns:
            Formatted global news report.
        """
        self._parse_date(trade_date)
        return self._call(
            get_global_news,
            {
                "curr_date": trade_date,
                "look_back_days": look_back_days,
                "limit": limit,
            },
        )

    # ------------------------------------------------------------------
    # Macro / prediction markets
    # ------------------------------------------------------------------

    def get_macro_indicators(
        self,
        indicator: str,
        trade_date: str,
        look_back_days: int | None = None,
    ) -> str:
        """Fetch a macroeconomic series from FRED.

        Args:
            indicator: Friendly alias ('cpi', 'fed_funds_rate', '10y_treasury',
                ...) or a raw FRED series ID such as 'CPIAUCSL'.
            trade_date: End of the window, YYYY-MM-DD.
            look_back_days: Window length; None uses a 1-year window.

        Returns:
            Formatted macro series report.
        """
        self._parse_date(trade_date)
        return self._call(
            get_macro_indicators,
            {
                "indicator": indicator,
                "curr_date": trade_date,
                "look_back_days": look_back_days,
            },
        )

    def get_prediction_markets(self, topic: str, limit: int | None = None) -> str:
        """Fetch market-implied probabilities for a forward-looking topic.

        Args:
            topic: Event keyword(s), e.g. 'Fed rate cut'.
            limit: Max markets to return; None uses the vendor default.

        Returns:
            Formatted prediction market report.
        """
        return self._call(get_prediction_markets, {"topic": topic, "limit": limit})
