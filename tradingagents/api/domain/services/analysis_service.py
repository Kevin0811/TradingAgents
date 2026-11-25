"""Analysis service - orchestrates the full trading analysis pipeline."""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.runnables import Runnable, RunnableBinding, RunnableWithFallbacks

from tradingagents.api.core.discussion_log import DiscussionLog
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.core.safe_errors import error_text, traceback_is_safe
from tradingagents.api.domain.analysts import to_core_analyst_keys
from tradingagents.api.domain.entities import AnalysisResult
from tradingagents.api.domain.horizon import horizon_context
from tradingagents.api.domain.services.config_overrides import CLEARED_KEYS
from tradingagents.api.domain.services.decision_service import pm_time_horizon
from tradingagents.api.infrastructure.discussion_capture import (
    DISCUSSION_EMITTER,
    TaskEventEmitter,
    analyst_agent_ids,
)
from tradingagents.api.infrastructure.llm_callbacks import (
    TIER_METADATA_KEY,
    TIERS,
    CancelOnLLMStart,
    OllamaCacheTrimHandler,
    TokenUsageRecorder,
)
from tradingagents.api.infrastructure.ollama_cache_trimmer import (
    OllamaCacheTrimmer,
    ollama_native_url,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients import tier_provider

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
        events: DiscussionLog | None = None,
        investment_horizon: str | None = None,
    ) -> AnalysisResult:
        """Run the full trading analysis pipeline.

        Args:
            ticker: Ticker symbol to analyze.
            trade_date: Trading date in YYYY-MM-DD format.
            asset_type: Asset type ("stock" or "crypto").
            selected_analysts: Tuple of analyst types to include.
            overrides: Optional per-run config overrides (e.g. max_debate_rounds,
                deep_think_llm) merged on top of the service's base config. None
                values are ignored, so callers can pass a sparse dict, except
                for ``CLEARED_KEYS``, where None clears the base config's value.
            cancel_event: Set to cancel the run: it stops at its next LLM call.
                A cancel during the last step may come too late for the core's
                own side effects: the Portfolio Manager's decision can already
                be in the core's decision log (``memory_log.store_decision``)
                when the run is cancelled. The API discards the result all the
                same; it cannot undo that write (the core is not edited here).
            events: The task's discussion log. When given, the run's turns and
                tool calls are recorded there (see ``discussion_capture``); any
                turn still open when the run ends is recorded as aborted.
            investment_horizon: The caller's horizon ("1w", "1m", "3m", "6m",
                "12m"), passed to the core as its ``portfolio`` input (see
                ``tradingagents.api.domain.horizon``), which also keeps a
                checkpoint from resuming across horizons. None passes no
                portfolio, as before.

        Returns:
            AnalysisResult with all reports and decisions, and the run's token
            usage.

        Raises:
            AnalysisCancelled: ``cancel_event`` was set before the run ended.
            AnalysisError: If the analysis fails.
            Either carries the run's token usage so far in ``token_usage``
            (``partial`` true).
        """
        usage: TokenUsageRecorder | None = None
        # Set here, on the worker thread that runs the graph: a context
        # variable set by whoever submitted the job would not reach it.
        emitter = TaskEventEmitter(events) if events is not None else None
        emitter_token = DISCUSSION_EMITTER.set(emitter) if emitter is not None else None
        try:
            effective_config = self._config
            if overrides:
                clean_overrides = {
                    k: v for k, v in overrides.items() if v is not None or k in CLEARED_KEYS
                }
                if clean_overrides:
                    effective_config = {**(self._config or {}), **clean_overrides}

            # The config the core runs on: it falls back the same way.
            usage = TokenUsageRecorder(_tier_models(effective_config or DEFAULT_CONFIG))
            # API names ('sentiment') -> the core's keys ('social').
            core_analysts = to_core_analyst_keys(selected_analysts)
            if emitter is not None:
                run_config = effective_config or DEFAULT_CONFIG
                emitter.start(
                    ticker=ticker,
                    trade_date=trade_date,
                    analysts=analyst_agent_ids(core_analysts),
                    debate_rounds=int(run_config.get("max_debate_rounds", 1)),
                    risk_rounds=int(run_config.get("max_risk_discuss_rounds", 1)),
                    checkpoint_enabled=bool(run_config.get("checkpoint_enabled")),
                )
            graph = TradingAgentsGraph(
                selected_analysts=core_analysts,
                debug=self._debug,
                config=effective_config,
                callbacks=self._callbacks(effective_config, cancel_event, usage),
            )
            _tag_tiers(graph)

            # Horizon only, never an empty PortfolioContext ("No current position").
            # Without a horizon the call is exactly the one made before horizons.
            portfolio = horizon_context(investment_horizon)
            final_state, signal = graph.propagate(
                ticker,
                trade_date,
                asset_type=asset_type,
                **({"portfolio": portfolio} if portfolio is not None else {}),
            )
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
                investment_horizon=portfolio.horizon if portfolio is not None else None,
                time_horizon=pm_time_horizon(final_state.get("final_trade_decision")),
                token_usage=usage.summary(partial=False),
            )

        except Exception as e:
            # The calls made before it stopped (an analyst still running in
            # another thread may add more, too late for this snapshot).
            token_usage = usage.summary(partial=True) if usage is not None else None
            if cancel_event is not None and cancel_event.is_set():
                # Whatever the core made of the cancel exception on its way up
                # (re-raised, wrapped, or a later error), the run was cancelled.
                # The caller logs the one INFO line per cancelled run.
                logger.debug("Analysis of %s on %s cancelled", ticker, trade_date)
                raise AnalysisCancelled("analysis cancelled", token_usage=token_usage) from e
            # error_text: an LLM client's pydantic error would otherwise carry
            # its arguments, API key included, to the task status and the log.
            logger.error(
                "Analysis failed for %s on %s: %s", ticker, trade_date, error_text(e),
                exc_info=traceback_is_safe(e),
            )
            raise AnalysisError(
                message=f"Analysis failed for {ticker} on {trade_date}",
                detail=error_text(e),
                token_usage=token_usage,
            ) from e
        finally:
            if emitter is not None:
                try:
                    emitter.close()
                finally:
                    DISCUSSION_EMITTER.reset(emitter_token)

    def _callbacks(
        self,
        config: dict[str, Any] | None,
        cancel_event: threading.Event | None,
        usage: TokenUsageRecorder,
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
        # After the cancel handler: LangChain re-raises the first handler error
        # of a call's start, so a call it refuses never reaches the recorder,
        # which would otherwise keep that call's run_id with no end to pop it.
        # (Either way the refused call is not counted: calls count at their end.)
        callbacks.append(usage)
        return callbacks


def _tier_models(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Each tier's (provider, configured model) in ``config``, for the usage report."""
    return {
        tier: (
            tier_provider(config, tier) if config.get("llm_provider") else "",
            str(config.get(f"{tier}_think_llm") or ""),
        )
        for tier in TIERS
    }


def _tag_tiers(graph: Any) -> None:
    """Tag each tier's model with its tier, for :class:`TokenUsageRecorder`.

    The core builds one model object per tier and hands that same object to
    every agent (bound runnables wrap it, they do not copy it), so its metadata,
    set here after construction, reaches the callbacks of each of its calls.
    Existing metadata is kept.

    Tagging never fails the run. A tier attribute that is a wrapper (a binding
    such as ``with_retry()`` or ``.bind()``, or ``with_fallbacks()``) has the
    model(s) inside it tagged. A model that cannot take metadata is skipped
    with a warning: its calls then carry no tier and appear in no usage entry,
    neither as calls nor as tokens.
    """
    for tier in TIERS:
        llm = getattr(graph, f"{tier}_thinking_llm", None)
        if llm is None:
            # Only a stand-in graph lacks one; a core that renamed these is
            # caught by test_api_token_usage, which builds the real graph.
            logger.debug("No %s-tier model on the graph; its token usage is not counted", tier)
            continue
        for untagged in _tag_model(llm, tier):
            logger.warning(
                "Cannot tag the %s-tier model (%s) with its tier; its LLM calls are "
                "left out of the run's token usage",
                tier,
                type(untagged).__name__,
            )


def _tag_model(llm: Any, tier: str) -> list[Any]:
    """Tag ``llm``, or the model(s) a wrapper holds, with ``tier``; return what could not be."""
    if isinstance(llm, RunnableWithFallbacks):
        untagged = _tag_model(llm.runnable, tier)
        for fallback in llm.fallbacks:
            untagged += _tag_model(fallback, tier)
        return untagged
    bound = getattr(llm, "bound", None)
    if isinstance(llm, RunnableBinding) or isinstance(bound, Runnable):
        # RunnableBinding and its subclasses (RunnableRetry, a provider's own
        # binding): the calls are the bound model's, with its metadata.
        return _tag_model(bound, tier)
    try:
        metadata = getattr(llm, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        llm.metadata = {**metadata, TIER_METADATA_KEY: tier}
    except Exception:  # noqa: BLE001 - a usage tag must never fail the run
        logger.debug("Setting metadata on %r failed", llm, exc_info=True)
        return [llm]
    return []
