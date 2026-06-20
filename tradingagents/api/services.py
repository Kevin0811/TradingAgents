"""Business logic services for the TradingAgents API.

This module provides service functions that wrap the agent execution
and data access tools, returning structured data suitable for API responses.
"""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def run_full_analysis(
    ticker: str,
    trade_date: str,
    asset_type: str = "stock",
    selected_analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals"),
    debug: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full trading analysis pipeline.

    Args:
        ticker: Ticker symbol to analyze.
        trade_date: Trading date in YYYY-MM-DD format.
        asset_type: Asset type ("stock" or "crypto").
        selected_analysts: Tuple of analyst types to include.
        debug: Enable debug mode.
        config: Optional configuration dict.

    Returns:
        Dict with analysis results.
    """
    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=debug,
        config=config,
    )

    final_state, signal = graph.propagate(ticker, trade_date, asset_type=asset_type)

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "asset_type": asset_type,
        "market_report": final_state.get("market_report"),
        "sentiment_report": final_state.get("sentiment_report"),
        "news_report": final_state.get("news_report"),
        "fundamentals_report": final_state.get("fundamentals_report"),
        "investment_plan": final_state.get("investment_plan"),
        "trader_investment_plan": final_state.get("trader_investment_plan"),
        "final_trade_decision": final_state.get("final_trade_decision"),
        "signal": signal,
    }


def _create_market_analyst_node(llm: Any) -> callable:
    """Create a market analyst node function."""
    return create_market_analyst(llm)


# ---------------------------------------------------------------------------
# Individual analyst services
# ---------------------------------------------------------------------------


def run_market_analyst(
    ticker: str,
    trade_date: str,
    asset_type: str = "stock",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the market analyst.

    Args:
        ticker: Ticker symbol.
        trade_date: Trading date.
        asset_type: Asset type.
        config: Optional configuration.

    Returns:
        Dict with market report.
    """
    from tradingagents.agents.utils.agent_utils import get_language_instruction
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage

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

    llm = _get_llm(config)
    chain = prompt | llm.bind_tools(tools)

    messages = [
        HumanMessage(content=f"Analyze {ticker} for date {trade_date}. {instrument_context}")
    ]

    result = chain.invoke({"messages": messages})

    report = result.content if len(result.tool_calls) == 0 else ""

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "analyst_type": "market",
        "report": report,
    }


def run_sentiment_analyst(
    ticker: str,
    trade_date: str,
    asset_type: str = "stock",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the sentiment analyst.

    Returns:
        Dict with sentiment report.
    """
    graph = TradingAgentsGraph(
        selected_analysts=("social",),
        debug=False,
        config=config,
    )

    identity = resolve_instrument_identity(ticker)
    instrument_context = build_instrument_context(ticker, asset_type, identity)

    init_state = {
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

    final_state = graph.graph.invoke(init_state)

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "analyst_type": "sentiment",
        "report": final_state.get("sentiment_report", ""),
    }


def run_news_analyst(
    ticker: str,
    trade_date: str,
    asset_type: str = "stock",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the news analyst.

    Returns:
        Dict with news report.
    """
    graph = TradingAgentsGraph(
        selected_analysts=("news",),
        debug=False,
        config=config,
    )

    identity = resolve_instrument_identity(ticker)
    instrument_context = build_instrument_context(ticker, asset_type, identity)

    init_state = {
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

    final_state = graph.graph.invoke(init_state)

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "analyst_type": "news",
        "report": final_state.get("news_report", ""),
    }


def run_fundamentals_analyst(
    ticker: str,
    trade_date: str,
    asset_type: str = "stock",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the fundamentals analyst.

    Returns:
        Dict with fundamentals report.
    """
    graph = TradingAgentsGraph(
        selected_analysts=("fundamentals",),
        debug=False,
        config=config,
    )

    identity = resolve_instrument_identity(ticker)
    instrument_context = build_instrument_context(ticker, asset_type, identity)

    init_state = {
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

    final_state = graph.graph.invoke(init_state)

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "analyst_type": "fundamentals",
        "report": final_state.get("fundamentals_report", ""),
    }


# ---------------------------------------------------------------------------
# Data access services
# ---------------------------------------------------------------------------


def fetch_stock_data(ticker: str, trade_date: str, period: int = 365) -> str:
    """Fetch OHLCV stock data for a ticker.

    Args:
        ticker: Ticker symbol.
        trade_date: Trading date.
        period: Days of historical data.

    Returns:
        CSV-formatted stock data.
    """
    return get_stock_data(ticker, trade_date, period=period)


def fetch_indicators(
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


def fetch_fundamentals(
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


def fetch_news(
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


def fetch_macro_indicators(country: str = "US") -> str:
    """Fetch macro indicators for a country.

    Args:
        country: Country code.

    Returns:
        Macro indicator data string.
    """
    return get_macro_indicators(country=country)


def fetch_prediction_markets(ticker: str, trade_date: str) -> str:
    """Fetch prediction market data.

    Args:
        ticker: Ticker symbol.
        trade_date: Trading date.

    Returns:
        Prediction market data string.
    """
    return get_prediction_markets(ticker=ticker, date=trade_date)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_llm(config: dict[str, Any] | None = None) -> Any:
    """Get a deep-thinking LLM client."""
    from tradingagents.llm_clients import create_llm_client
    from tradingagents.default_config import DEFAULT_CONFIG

    effective_config = config or DEFAULT_CONFIG
    provider = effective_config.get("llm_provider", "openai")

    kwargs = {}
    provider_lower = provider.lower()
    if provider_lower == "google":
        thinking_level = effective_config.get("google_thinking_level")
        if thinking_level:
            kwargs["thinking_level"] = thinking_level
    elif provider_lower == "openai":
        reasoning_effort = effective_config.get("openai_reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    elif provider_lower == "anthropic":
        effort = effective_config.get("anthropic_effort")
        if effort:
            kwargs["effort"] = effort

    temperature = effective_config.get("temperature")
    if temperature is not None and temperature != "":
        kwargs["temperature"] = float(temperature)

    client = create_llm_client(
        provider=provider,
        model=effective_config.get("deep_think_llm", "o3-mini"),
        base_url=effective_config.get("backend_url"),
        **kwargs,
    )
    return client.get_llm()