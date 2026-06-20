"""Analysis service - orchestrates the full trading analysis pipeline."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.api.core.exceptions import AnalysisError
from tradingagents.api.domain.entities import AnalysisResult
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for running the full trading analysis pipeline.

    This service orchestrates the multi-agent analysis flow:
    1. Analysts (market, sentiment, news, fundamentals) gather data
    2. Researchers debate the investment thesis
    3. Research Manager synthesizes into an investment plan
    4. Trader converts plan into a transaction proposal
    5. Risk Management debates risk factors
    6. Portfolio Manager produces final trade decision
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        debug: bool = False,
    ):
        """Initialize the analysis service.

        Args:
            config: Optional configuration dict.
            debug: Enable debug mode.
        """
        self._config = config
        self._debug = debug

    def run_analysis(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str = "stock",
        selected_analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals"),
    ) -> AnalysisResult:
        """Run the full trading analysis pipeline.

        Args:
            ticker: Ticker symbol to analyze.
            trade_date: Trading date in YYYY-MM-DD format.
            asset_type: Asset type ("stock" or "crypto").
            selected_analysts: Tuple of analyst types to include.

        Returns:
            AnalysisResult with all reports and decisions.

        Raises:
            AnalysisError: If the analysis fails.
        """
        try:
            graph = TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=self._debug,
                config=self._config,
            )

            final_state, signal = graph.propagate(ticker, trade_date, asset_type=asset_type)

            return AnalysisResult(
                ticker=ticker,
                trade_date=trade_date,
                asset_type=asset_type,
                market_report=final_state.get("market_report"),
                sentiment_report=final_state.get("sentiment_report"),
                news_report=final_state.get("news_report"),
                fundamentals_report=final_state.get("fundamentals_report"),
                investment_plan=final_state.get("investment_plan"),
                trader_investment_plan=final_state.get("trader_investment_plan"),
                final_trade_decision=final_state.get("final_trade_decision"),
                signal=signal,
            )

        except Exception as e:
            logger.exception("Analysis failed for %s on %s", ticker, trade_date)
            raise AnalysisError(
                message=f"Analysis failed for {ticker} on {trade_date}",
                detail=str(e),
            ) from e