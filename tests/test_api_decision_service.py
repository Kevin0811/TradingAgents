"""Tests for the TradingAgents API DecisionService markdown parsing.

Covers regressions the API's hand-rolled markdown parsers reintroduced
against two main-branch fixes:
- 43fc275: an unparseable rating must surface as REVIEW, not a silent Hold.
- 1c44dd1: price fields must clean currency symbols/thousands separators and
  never salvage a percentage into an absolute price.
"""

from __future__ import annotations

import pytest

from tradingagents.api.domain.services.decision_service import DecisionService
from tradingagents.api.schemas.response import (
    PortfolioDecisionRating,
    ResearchPlanRecommendation,
)


@pytest.fixture
def service() -> DecisionService:
    return DecisionService(state_repository=None)


@pytest.mark.unit
class TestParsePortfolioDecision:
    def test_unparseable_rating_yields_review_not_hold(self, service):
        text = "**Executive Summary**: n/a\n\n**Investment Thesis**: n/a\n"
        decision = service.parse_portfolio_decision(text)
        assert decision.rating == PortfolioDecisionRating.REVIEW

    def test_recognized_rating_parses_normally(self, service):
        text = (
            "**Rating**: Overweight\n\n"
            "**Executive Summary**: buy some\n\n"
            "**Investment Thesis**: good company\n"
        )
        decision = service.parse_portfolio_decision(text)
        assert decision.rating == PortfolioDecisionRating.OVERWEIGHT

    def test_price_target_with_currency_and_thousands_separator(self, service):
        text = (
            "**Rating**: Hold\n\n"
            "**Executive Summary**: x\n\n"
            "**Investment Thesis**: y\n\n"
            "**Price Target**: $1,234.56\n"
        )
        decision = service.parse_portfolio_decision(text)
        assert decision.price_target == 1234.56

    def test_price_target_percentage_is_dropped_not_salvaged(self, service):
        text = (
            "**Rating**: Hold\n\n"
            "**Executive Summary**: x\n\n"
            "**Investment Thesis**: y\n\n"
            "**Price Target**: 15%\n"
        )
        decision = service.parse_portfolio_decision(text)
        assert decision.price_target is None


@pytest.mark.unit
class TestParseTraderProposal:
    def test_entry_price_with_currency_symbol(self, service):
        text = "**Action**: Buy\n\n**Reasoning**: r\n\n**Entry Price**: $189.50\n"
        proposal = service.parse_trader_proposal(text)
        assert proposal.entry_price == 189.50

    def test_stop_loss_percentage_becomes_none(self, service):
        text = "**Action**: Buy\n\n**Reasoning**: r\n\n**Stop Loss**: 15%\n"
        proposal = service.parse_trader_proposal(text)
        assert proposal.stop_loss is None


@pytest.mark.unit
class TestParseResearchPlan:
    def test_unparseable_recommendation_yields_review_not_hold(self, service):
        text = "**Rationale**: none\n\n**Strategic Actions**: none\n"
        plan = service.parse_research_plan(text)
        assert plan.recommendation == ResearchPlanRecommendation.REVIEW

    def test_recognized_recommendation_parses_normally(self, service):
        text = (
            "**Recommendation**: Sell\n\n"
            "**Rationale**: bad company\n\n"
            "**Strategic Actions**: exit position\n"
        )
        plan = service.parse_research_plan(text)
        assert plan.recommendation == ResearchPlanRecommendation.SELL
