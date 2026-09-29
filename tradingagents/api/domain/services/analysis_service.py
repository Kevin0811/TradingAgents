"""Analysis service - orchestrates the full trading analysis pipeline."""

from __future__ import annotations

import logging
import threading
from typing import Any

from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.domain.analysts import to_core_analyst_keys
from tradingagents.api.domain.entities import AnalysisResult
from tradingagents.api.infrastructure.llm_callbacks import (
    CancelOnLLMStart,
    OllamaCacheTrimHandler,
)
from tradingagents.api.infrastructure.ollama_cache_trimmer import (
    OllamaCacheTrimmer,
    ollama_native_url,
)
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
        cache_trimmer: OllamaCacheTrimmer | None = None,
    ):
        """Initialize the analysis service.

        Args:
            config: Optional configuration dict.
            debug: Enable debug mode.
            cache_trimmer: The app's shared Ollama prompt-cache trimmer; used
                only by runs whose provider is Ollama.
        """
        self._config = config
        self._debug = debug
        self._cache_trimmer = cache_trimmer

    def run_analysis(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str = "stock",
        selected_analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals"),
        overrides: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AnalysisResult:
        """Run the full trading analysis pipeline.

        Args:
            ticker: Ticker symbol to analyze.
            trade_date: Trading date in YYYY-MM-DD format.
            asset_type: Asset type ("stock" or "crypto").
            selected_analysts: Tuple of analyst types to include.
            overrides: Optional per-run config overrides (e.g. max_debate_rounds,
                deep_think_llm) merged on top of the service's base config. None
                values are ignored, so callers can pass a sparse dict.
            cancel_event: Set to cancel the run: it stops at its next LLM call.
                A cancel during the last step may come too late for the core's
                own side effects: the Portfolio Manager's decision can already
                be in the core's decision log (``memory_log.store_decision``)
                when the run is cancelled. The API discards the result all the
                same; it cannot undo that write (the core is not edited here).

        Returns:
            AnalysisResult with all reports and decisions.

        Raises:
            AnalysisCancelled: ``cancel_event`` was set before the run ended.
            AnalysisError: If the analysis fails.
        """
        try:
            effective_config = self._config
            if overrides:
                clean_overrides = {k: v for k, v in overrides.items() if v is not None}
                if clean_overrides:
                    effective_config = {**(self._config or {}), **clean_overrides}

            graph = TradingAgentsGraph(
                # API names ('sentiment') -> the core's keys ('social').
                selected_analysts=to_core_analyst_keys(selected_analysts),
                debug=self._debug,
                config=effective_config,
                callbacks=self._callbacks(effective_config, cancel_event) or None,
            )

            final_state, signal = graph.propagate(ticker, trade_date, asset_type=asset_type)
            if cancel_event is not None and cancel_event.is_set():
                # Cancelled after its last LLM call: the result is discarded.
                raise AnalysisCancelled("analysis cancelled")

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
            if cancel_event is not None and cancel_event.is_set():
                # Whatever the core made of the cancel exception on its way up
                # (re-raised, wrapped, or a later error), the run was cancelled.
                # The caller logs the one INFO line per cancelled run.
                logger.debug("Analysis of %s on %s cancelled", ticker, trade_date)
                raise AnalysisCancelled("analysis cancelled") from e
            logger.exception("Analysis failed for %s on %s", ticker, trade_date)
            raise AnalysisError(
                message=f"Analysis failed for {ticker} on {trade_date}",
                detail=str(e),
            ) from e

    def _callbacks(
        self, config: dict[str, Any] | None, cancel_event: threading.Event | None
    ) -> list:
        """The LLM callback handlers for one run (the core attaches them to every LLM)."""
        callbacks: list = []
        if cancel_event is not None:
            callbacks.append(CancelOnLLMStart(cancel_event))
        trimmer = self._cache_trimmer
        config = config or {}
        provider = str(config.get("llm_provider") or "").strip().lower()
        if trimmer is not None and trimmer.enabled and provider == "ollama":
            # The run's own models only: quick and deep may differ, and other
            # loaded models (another app's) are never unloaded.
            models = [config.get("quick_think_llm"), config.get("deep_think_llm")]
            callbacks.append(
                OllamaCacheTrimHandler(
                    trimmer,
                    ollama_native_url(config.get("backend_url")),
                    [m for m in models if m],
                )
            )
        return callbacks
