"""Tests for the analysis request's investment horizon (issue #31).

``investment_horizon`` ('1w'/'1m'/'3m'/'6m'/'12m', optional) reaches the core as
its ``portfolio`` input, a horizon-only stand-in that renders a fixed sentence
plus the core's "holdings not provided" notice; the result echoes it beside the
portfolio manager's own ``time_horizon``. Also: ``trade_date`` is refused at
request time when the core's ``create_run_state`` would refuse it (non-canonical
or in the future).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.discussion_fakes import TICKER, TRADE_DATE, install_offline, run_config
from tradingagents.agents.context import get_portfolio_context_from_state
from tradingagents.api import config as api_config
from tradingagents.api.app import create_app
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.domain.horizon import (
    HORIZON_SPANS,
    HorizonContext,
    horizon_context,
    horizon_sentence,
)
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.decision_service import (
    parse_portfolio_decision,
    pm_time_horizon,
)
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.schemas.request import AnalyzeRequest, SingleAnalystRequest
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus
from tradingagents.dataflows import date_window
from tradingagents.graph import trading_graph

GRAPH = "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
TASKS = "/api/v1/analyze/tasks"
HORIZONS = ("1w", "1m", "3m", "6m", "12m")

PM_DECISION = (
    "**Rating**: Overweight\n\n"
    "**Executive Summary**: Add on weakness.\n\n"
    "**Investment Thesis**: Margins recover.\n\n"
    "**Price Target**: 210.5\n\n"
    "**Time Horizon**: 6-9 months"
)


@pytest.fixture(autouse=True)
def _not_ollama(monkeypatch):
    """Pin a provider that is not Ollama (no cache trimmer reaching for a server)."""
    monkeypatch.setitem(api_config.DEFAULT_CONFIG, "llm_provider", "openai")


def _graph(final_state: dict | None = None) -> MagicMock:
    """A TradingAgentsGraph class stand-in whose propagate returns ``final_state``."""
    graph_cls = MagicMock()
    graph_cls.return_value.propagate.return_value = (final_state or {}, "Overweight")
    return graph_cls


def _portfolio_passed(graph_cls: MagicMock):
    call = graph_cls.return_value.propagate.call_args
    return call.kwargs.get("portfolio")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequestSchemas:
    @pytest.mark.parametrize("model", [AnalyzeRequest, TaskCreateRequest])
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_each_horizon_is_accepted(self, model, horizon):
        request = model(ticker="AAPL", trade_date="2026-06-01", investment_horizon=horizon)
        assert request.investment_horizon == horizon

    @pytest.mark.parametrize("model", [AnalyzeRequest, TaskCreateRequest])
    def test_absent_or_null_is_none(self, model):
        assert model(ticker="AAPL", trade_date="2026-06-01").investment_horizon is None
        assert (
            model(ticker="AAPL", trade_date="2026-06-01", investment_horizon=None)
            .investment_horizon is None
        )

    @pytest.mark.parametrize("model", [AnalyzeRequest, TaskCreateRequest])
    @pytest.mark.parametrize("bad", ["2w", "12M", "1y", "", "6 months", 6])
    def test_any_other_value_is_refused(self, model, bad):
        with pytest.raises(ValidationError):
            model(ticker="AAPL", trade_date="2026-06-01", investment_horizon=bad)

    @pytest.mark.parametrize("path", [TASKS, "/api/v1/analyze"])
    def test_an_invalid_horizon_is_422_over_http(self, path):
        with patch(GRAPH, _graph()) as graph_cls:
            client = TestClient(create_app())
            resp = client.post(
                path,
                json={"ticker": "AAPL", "trade_date": "2026-06-01", "investment_horizon": "2y"},
            )
        assert resp.status_code == 422
        assert "investment_horizon" in resp.text
        graph_cls.assert_not_called()


# ---------------------------------------------------------------------------
# The horizon-only portfolio input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHorizonContext:
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_render_is_the_horizon_then_holdings_unknown(self, horizon):
        text = HorizonContext(horizon).render("AAPL")
        assert text.splitlines()[0] == horizon_sentence(horizon)
        assert HORIZON_SPANS[horizon] in text
        # The core's own notice for a run without holdings, verbatim.
        assert text.endswith(get_portfolio_context_from_state({}))
        assert "do not assume a flat book" in text
        assert "No current position" not in text
        assert "Portfolio at the analysis date" not in text

    def test_the_exact_sentence(self):
        assert horizon_sentence("12m") == (
            "Investment horizon: evaluate the position over the next 12 months; "
            "weigh short-term news and price moves accordingly."
        )

    def test_fingerprint_is_stable_and_differs_per_horizon(self):
        prints = {h: HorizonContext(h).fingerprint() for h in HORIZONS}
        assert len(set(prints.values())) == len(HORIZONS)
        assert prints == {h: HorizonContext(h).fingerprint() for h in HORIZONS}
        assert all(len(p) == 12 for p in prints.values())

    def test_unknown_horizon_is_refused(self):
        with pytest.raises(ValueError):
            HorizonContext("2y")

    def test_none_is_no_portfolio(self):
        assert horizon_context(None) is None
        assert horizon_context("3m") == HorizonContext("3m")

    def test_core_signature_distinguishes_horizons(self, tmp_path, monkeypatch):
        """The core's checkpoint signature changes with the horizon (and from none)."""
        install_offline(monkeypatch)
        graph = trading_graph.TradingAgentsGraph(
            selected_analysts=("market",), config=run_config(tmp_path))
        signatures = {graph._run_signature("stock", horizon_context(h)) for h in HORIZONS}
        signatures.add(graph._run_signature("stock", None))
        assert len(signatures) == len(HORIZONS) + 1


# ---------------------------------------------------------------------------
# Guard against the real core: the horizon lands in portfolio_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRealCore:
    def test_create_run_state_puts_the_horizon_in_portfolio_context(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        graph = trading_graph.TradingAgentsGraph(
            selected_analysts=("market",), config=run_config(tmp_path))
        state = graph.create_run_state(TICKER, TRADE_DATE, "stock", horizon_context("6m"))
        assert state["portfolio_context"] == HorizonContext("6m").render(TICKER)
        assert horizon_sentence("6m") in state["portfolio_context"]

    def test_create_run_state_without_horizon_is_unchanged(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        graph = trading_graph.TradingAgentsGraph(
            selected_analysts=("market",), config=run_config(tmp_path))
        assert graph.create_run_state(TICKER, TRADE_DATE, "stock", None)["portfolio_context"] == ""

    def test_a_full_offline_run_shows_the_horizon_to_the_decision_agents(
        self, tmp_path, monkeypatch
    ):
        model = install_offline(monkeypatch)
        result = AnalysisService(config=run_config(tmp_path)).run_analysis(
            ticker=TICKER, trade_date=TRADE_DATE, selected_analysts=("market",),
            investment_horizon="3m",
        )
        sentence = horizon_sentence("3m")
        shown = [p for p in model.prompts if sentence in p]
        # Trader, three risk debaters, portfolio manager (at least once each).
        assert len(shown) >= 5
        assert not any("No current position" in p for p in model.prompts)
        assert result.investment_horizon == "3m"
        # The canned PM decision has no time horizon: the core writes "not provided".
        assert result.time_horizon is None

    def test_a_full_offline_run_without_horizon_shows_none(self, tmp_path, monkeypatch):
        model = install_offline(monkeypatch)
        result = AnalysisService(config=run_config(tmp_path)).run_analysis(
            ticker=TICKER, trade_date=TRADE_DATE, selected_analysts=("market",),
        )
        assert not any("Investment horizon:" in p for p in model.prompts)
        assert result.investment_horizon is None


# ---------------------------------------------------------------------------
# Threading and the result
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThreadingAndResult:
    def test_absent_horizon_passes_no_portfolio(self):
        with patch(GRAPH, _graph()) as graph_cls:
            result = AnalysisService().run_analysis("AAPL", "2026-06-01")
        assert _portfolio_passed(graph_cls) is None
        assert result.to_dict()["investment_horizon"] is None
        assert result.to_dict()["time_horizon"] is None

    def test_horizon_reaches_propagate_and_result(self):
        state = {"final_trade_decision": PM_DECISION}
        with patch(GRAPH, _graph(state)) as graph_cls:
            result = AnalysisService().run_analysis(
                "AAPL", "2026-06-01", investment_horizon="12m")
        assert _portfolio_passed(graph_cls) == HorizonContext("12m")
        assert result.to_dict()["investment_horizon"] == "12m"
        assert result.to_dict()["time_horizon"] == "6-9 months"

    def test_task_result_over_http(self):
        state = {"final_trade_decision": PM_DECISION}
        with patch(GRAPH, _graph(state)) as graph_cls:
            app = create_app()
            client = TestClient(app)
            created = client.post(
                TASKS,
                json={"ticker": "AAPL", "trade_date": "2026-06-01", "investment_horizon": "6m"},
            )
            assert created.status_code == 202, created.text
            app.state.task_worker.shutdown(wait=True)
            task = client.get(f"{TASKS}/{created.json()['task_id']}").json()
        assert task["status"] == "completed", task
        assert task["investment_horizon"] == "6m"
        assert task["result"]["investment_horizon"] == "6m"
        assert task["result"]["time_horizon"] == "6-9 months"
        assert _portfolio_passed(graph_cls) == HorizonContext("6m")

    def test_task_without_horizon_over_http(self):
        with patch(GRAPH, _graph()) as graph_cls:
            app = create_app()
            client = TestClient(app)
            created = client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            app.state.task_worker.shutdown(wait=True)
            task = client.get(f"{TASKS}/{created.json()['task_id']}").json()
        assert task["status"] == "completed", task
        assert task["investment_horizon"] is None
        assert task["result"]["investment_horizon"] is None
        assert task["result"]["time_horizon"] is None
        assert _portfolio_passed(graph_cls) is None

    def test_sync_response_echoes_both(self):
        state = {"final_trade_decision": PM_DECISION}
        with patch(GRAPH, _graph(state)) as graph_cls:
            resp = TestClient(create_app()).post(
                "/api/v1/analyze",
                json={"ticker": "AAPL", "trade_date": "2026-06-01", "investment_horizon": "1w"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["investment_horizon"] == "1w"
        assert resp.json()["time_horizon"] == "6-9 months"
        assert _portfolio_passed(graph_cls) == HorizonContext("1w")

    def test_task_service_passes_the_horizon(self):
        tm = TaskManager()
        analysis = MagicMock()
        analysis.run_analysis.return_value.to_dict.return_value = {}
        analysis.run_analysis.return_value.token_usage = None
        request = TaskCreateRequest(
            ticker="AAPL", trade_date="2026-06-01", investment_horizon="1m")
        task, _ = tm.create_task(request)
        TaskService(task_manager=tm, analysis_service=analysis).execute_analysis(
            task.task_id, request)
        assert analysis.run_analysis.call_args.kwargs["investment_horizon"] == "1m"
        assert tm.get_task(task.task_id).status == TaskStatus.COMPLETED


@pytest.mark.unit
class TestDedup:
    def _request(self, horizon=None):
        return TaskCreateRequest(
            ticker="AAPL", trade_date="2026-06-01", investment_horizon=horizon)

    def test_different_horizons_are_different_tasks(self):
        tm = TaskManager()
        none, new_none = tm.create_task(self._request())
        six, new_six = tm.create_task(self._request("6m"))
        twelve, new_twelve = tm.create_task(self._request("12m"))
        assert new_none and new_six and new_twelve
        assert len({none.task_id, six.task_id, twelve.task_id}) == 3

    def test_the_same_horizon_is_a_duplicate(self):
        tm = TaskManager()
        first, _ = tm.create_task(self._request("6m"))
        again, is_new = tm.create_task(self._request("6m"))
        assert not is_new and again.task_id == first.task_id


# ---------------------------------------------------------------------------
# The PM's time horizon
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeHorizon:
    def test_read_from_the_rendered_decision(self):
        assert pm_time_horizon(PM_DECISION) == "6-9 months"

    @pytest.mark.parametrize("text", [None, "", "**Rating**: Hold",
                                      "**Rating**: Hold\n\n**Time Horizon**: not provided"])
    def test_none_when_absent(self, text):
        assert pm_time_horizon(text) is None

    def test_decision_parser_maps_not_provided_to_none(self):
        parsed = parse_portfolio_decision("**Rating**: Buy\n**Time Horizon**: Not provided")
        assert parsed.time_horizon is None


# ---------------------------------------------------------------------------
# trade_date at request time
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTradeDate:
    MODELS = [AnalyzeRequest, TaskCreateRequest, SingleAnalystRequest]

    @staticmethod
    def _tomorrow() -> str:
        return (date.fromisoformat(date_window.get_current_date()) + timedelta(days=1)).isoformat()

    @pytest.mark.parametrize("model", MODELS)
    def test_today_is_accepted(self, model):
        today = date_window.get_current_date()
        assert model(ticker="AAPL", trade_date=today).trade_date == today

    @pytest.mark.parametrize("model", MODELS)
    def test_the_future_is_refused(self, model):
        with pytest.raises(ValidationError, match="future"):
            model(ticker="AAPL", trade_date=self._tomorrow())

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("bad", ["2026-1-5", "2026-02-30", "20260105", "2026/01/05",
                                     " 2026-01-05", ""])
    def test_non_canonical_is_refused(self, model, bad):
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            model(ticker="AAPL", trade_date=bad)

    def test_the_bound_is_the_cores_clock(self, monkeypatch):
        monkeypatch.setattr(date_window, "get_current_date", lambda: "2026-03-01")
        # validators imported the function by name; patch where it is used too.
        from tradingagents.api.schemas import validators

        monkeypatch.setattr(validators, "get_current_date", lambda: "2026-03-01")
        AnalyzeRequest(ticker="AAPL", trade_date="2026-03-01")
        with pytest.raises(ValidationError):
            AnalyzeRequest(ticker="AAPL", trade_date="2026-03-02")

    def test_validator_uses_the_core_function(self):
        from tradingagents.api.schemas import validators

        assert validators.get_current_date is date_window.get_current_date
        assert trading_graph.get_current_date is date_window.get_current_date

    @pytest.mark.parametrize("path", [TASKS, "/api/v1/analyze"])
    @pytest.mark.parametrize("bad", ["2026-1-5", "future"])
    def test_refused_with_422_over_http(self, path, bad):
        trade_date = self._tomorrow() if bad == "future" else bad
        with patch(GRAPH, _graph()) as graph_cls:
            resp = TestClient(create_app()).post(
                path, json={"ticker": "AAPL", "trade_date": trade_date})
        assert resp.status_code == 422, resp.text
        assert "trade_date" in resp.text
        graph_cls.assert_not_called()

    def test_analyst_endpoint_refuses_the_future(self):
        resp = TestClient(create_app()).post(
            "/api/v1/analysts/news", json={"ticker": "AAPL", "trade_date": self._tomorrow()})
        assert resp.status_code == 422, resp.text
