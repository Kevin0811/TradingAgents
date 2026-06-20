"""Analyst service - executes individual analyst agents."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    resolve_instrument_identity,
)
from tradingagents.api.core.exceptions import AnalysisError
from tradingagents.api.domain.entities import AnalystReport
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)

# Map of analyst type to creation function
ANALYST_CREATORS = {
    "market": create_market_analyst,
    "sentiment": create_sentiment_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
}


class AnalystService:
    """Service for running individual analyst agents."""

    def __init__(self, llm_factory: callable, config: dict[str, Any] | None = None):
        """Initialize the analyst service.

        Args:
            llm_factory: Factory function to create LLM instances.
            config: Optional configuration dict.
        """
        self._llm_factory = llm_factory
        self._config = config

    def run_analyst(
        self,
        analyst_type: str,
        ticker: str,
        trade_date: str,
        asset_type: str = "stock",
    ) -> AnalystReport:
        """Run a single analyst.

        Args:
            analyst_type: Type of analyst to run.
            ticker: Ticker symbol.
            trade_date: Trading date.
            asset_type: Asset type.

        Returns:
            AnalystReport with the analysis results.

        Raises:
            AnalysisError: If the analyst type is unknown or execution fails.
        """
        creator = ANALYST_CREATORS.get(analyst_type)
        if creator is None:
            raise AnalysisError(
                message=f"Unknown analyst type: {analyst_type}",
                detail=f"Valid analyst types: {list(ANALYST_CREATORS.keys())}",
            )

        try:
            if analyst_type == "market":
                report = self._run_market_analyst(ticker, trade_date, asset_type)
            else:
                report = self._run_graph_analyst(analyst_type, ticker, trade_date, asset_type)

            return AnalystReport(
                ticker=ticker,
                trade_date=trade_date,
                analyst_type=analyst_type,
                report_content=report,
            )

        except AnalysisError:
            raise
        except Exception as e:
            logger.exception(
                "%s analyst failed for %s on %s",
                analyst_type, ticker, trade_date,
            )
            raise AnalysisError(
                message=f"{analyst_type} analyst failed for {ticker} on {trade_date}",
                detail=str(e),
            ) from e

    def _run_market_analyst(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str,
    ) -> str:
        """Run market analyst with tool-calling capability."""
        from tradingagents.agents.utils.agent_utils import get_indicators, get_stock_data, get_verified_market_snapshot

        identity = resolve_instrument_identity(ticker)
        instrument_context = build_instrument_context(ticker, asset_type, identity)

        tools = [get_stock_data, get_indicators, get_verified_market_snapshot]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant** indicators for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = self._llm_factory()
        chain = prompt | llm.bind_tools(tools)

        messages = [
            HumanMessage(content=f"Analyze {ticker} for date {trade_date}. {instrument_context}")
        ]

        result = chain.invoke({"messages": messages})
        return result.content if len(result.tool_calls) == 0 else ""

    def _run_graph_analyst(
        self,
        analyst_type: str,
        ticker: str,
        trade_date: str,
        asset_type: str,
    ) -> str:
        """Run analyst using TradingAgentsGraph for non-market analysts."""
        # Map analyst type to graph analyst selection
        analyst_map = {
            "sentiment": ("social",),
            "news": ("news",),
            "fundamentals": ("fundamentals",),
        }

        selected = analyst_map.get(analyst_type, (analyst_type,))
        graph = TradingAgentsGraph(
            selected_analysts=selected,
            debug=False,
            config=self._config,
        )

        identity = resolve_instrument_identity(ticker)
        instrument_context = build_instrument_context(ticker, asset_type, identity)

        init_state = self._create_init_state(ticker, trade_date, asset_type, instrument_context)
        final_state = graph.graph.invoke(init_state)

        # Map analyst type to state key
        state_key_map = {
            "sentiment": "sentiment_report",
            "news": "news_report",
            "fundamentals": "fundamentals_report",
        }

        key = state_key_map.get(analyst_type, "")
        return final_state.get(key, "")

    def _create_init_state(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str,
        instrument_context: str,
    ) -> dict[str, Any]:
        """Create initial state for graph execution."""
        return {
            "messages": [],
            "company_of_interest": ticker,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": trade_date,
            "sender": "user",
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "final_trade_decision": "",
            "past_context": "",
        }