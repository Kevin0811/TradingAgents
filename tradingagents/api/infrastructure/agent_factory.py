"""Agent factory for creating analyst agents."""

from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)


class AgentFactory:
    """Factory for creating analyst agents."""

    def __init__(self, llm_factory: Callable[[], Any]):
        """Initialize the agent factory.

        Args:
            llm_factory: Factory function that returns an LLM instance.
        """
        self._llm_factory = llm_factory

    def create_market_analyst(self) -> Callable:
        """Create a market analyst node.

        Returns:
            Market analyst node function.
        """
        llm = self._llm_factory()
        return create_market_analyst(llm)

    def create_sentiment_analyst(self) -> Callable:
        """Create a sentiment analyst node.

        Returns:
            Sentiment analyst node function.
        """
        llm = self._llm_factory()
        return create_sentiment_analyst(llm)

    def create_news_analyst(self) -> Callable:
        """Create a news analyst node.

        Returns:
            News analyst node function.
        """
        llm = self._llm_factory()
        return create_news_analyst(llm)

    def create_fundamentals_analyst(self) -> Callable:
        """Create a fundamentals analyst node.

        Returns:
            Fundamentals analyst node function.
        """
        llm = self._llm_factory()
        return create_fundamentals_analyst(llm)

    def create_analyst(self, analyst_type: str) -> Callable:
        """Create an analyst by type.

        Args:
            analyst_type: Type of analyst to create.

        Returns:
            Analyst node function.

        Raises:
            ValueError: If analyst type is unknown.
        """
        creators = {
            "market": self.create_market_analyst,
            "sentiment": self.create_sentiment_analyst,
            "news": self.create_news_analyst,
            "fundamentals": self.create_fundamentals_analyst,
        }

        creator = creators.get(analyst_type)
        if creator is None:
            raise ValueError(
                f"Unknown analyst type: {analyst_type}. "
                f"Valid types: {list(creators.keys())}"
            )

        return creator()