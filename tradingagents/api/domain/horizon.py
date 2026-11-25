"""The caller's investment horizon, as the core's decision agents see it.

The core takes an optional ``portfolio`` in ``TradingAgentsGraph.propagate``:
``portfolio.render(ticker)`` becomes the run's ``portfolio_context`` (shown to
the trader, the three risk debaters and the portfolio manager) and
``portfolio.fingerprint()`` joins the checkpoint run signature. The API has no
holdings to pass, only a horizon, so :class:`HorizonContext` stands in for the
core's ``PortfolioContext`` with those two methods.

It must not be an empty ``PortfolioContext``: that renders "No current position
in X", telling the agents the caller holds nothing. Holdings are unknown here,
so the block keeps the core's own "not provided" notice (read from the core,
so the two never drift apart) after the horizon sentence. Every sentence is
fixed per enum value; no caller text reaches a prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from tradingagents.agents.context import get_portfolio_context_from_state

# The span each horizon value names, in the rendered sentence.
HORIZON_SPANS: dict[str, str] = {
    "1w": "the next week",
    "1m": "the next month",
    "3m": "the next 3 months",
    "6m": "the next 6 months",
    "12m": "the next 12 months",
}


def horizon_sentence(horizon: str) -> str:
    """The fixed sentence for one horizon value."""
    return (
        f"Investment horizon: evaluate the position over {HORIZON_SPANS[horizon]}; "
        "weigh short-term news and price moves accordingly."
    )


def holdings_unknown_notice() -> str:
    """The core's notice for a run without portfolio context (holdings unknown)."""
    return get_portfolio_context_from_state({})


@dataclass(frozen=True)
class HorizonContext:
    """A run's portfolio input carrying only the investment horizon.

    Duck-typed for the core's ``portfolio`` argument: ``render`` and
    ``fingerprint`` are all the core calls on it.
    """

    horizon: str

    def __post_init__(self) -> None:
        if self.horizon not in HORIZON_SPANS:
            raise ValueError(f"unknown investment horizon: {self.horizon!r}")

    def render(self, ticker: str) -> str:
        """The block for the decision agents: the horizon, then holdings unknown."""
        return f"{horizon_sentence(self.horizon)}\n{holdings_unknown_notice()}"

    def fingerprint(self) -> str:
        """Stable digest of the horizon, so a run cannot resume another horizon's checkpoint."""
        return hashlib.sha256(f"investment_horizon={self.horizon}".encode()).hexdigest()[:12]


def horizon_context(horizon: str | None) -> HorizonContext | None:
    """The core's ``portfolio`` argument for ``horizon``; None (no context) when unset."""
    if horizon is None:
        return None
    return HorizonContext(str(getattr(horizon, "value", horizon)))
