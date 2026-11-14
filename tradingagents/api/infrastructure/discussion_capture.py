"""Capture a run's discussion from LangChain callbacks, without touching the core.

One configure hook is registered at import time
(``register_configure_hook(DISCUSSION_EMITTER, inheritable=True)``): every
callback manager LangChain builds picks up whatever handler
:data:`DISCUSSION_EMITTER` holds in the current context. The analysis service
sets it to a :class:`TaskEventEmitter` *inside the worker thread* around the
graph run (``ThreadPoolExecutor.submit`` does not copy context variables, so
setting it on the submitting thread would capture nothing); LangGraph copies
the context into the threads that run the parallel analysts.

What is read, and what never is:

* a turn is a **top-level graph node**: its chain name is in :data:`MAPPING`,
  equals ``metadata["langgraph_node"]``, and its ``checkpoint_ns`` is empty
  (inner runnables and the analysts' own subgraphs have a namespace). An
  unmapped top-level node becomes a turn of ``agent_id="unknown"``;
  :data:`IGNORED_NODES` (the Memory Log) are no turn at all;
* a turn's text is the one state field its node returns
  (:attr:`Seat.content_path`) -- the same text that ends in the final state.
  Raw LLM output is never used (structured-output nodes answer in JSON tool
  arguments) and LLM callbacks are ignored altogether;
* tool events carry the tool's name only; the seat is the head of the tool
  run's ``checkpoint_ns`` (the analyst whose subgraph called it);
* inputs, prompts, tool arguments and tool outputs are never read.

Callbacks come from several threads at once: each takes the emitter's lock
(then the log's), does O(1) work with no I/O, and never raises.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from tradingagents.api.core.discussion_log import (
    PHASE_CHANGED,
    RUN_STARTED,
    STATUS_RUNNING,
    TASK_STATUS,
    TOOL_CALL,
    TOOL_RESULT,
    TURN_ABORTED,
    TURN_COMPLETED,
    TURN_STARTED,
    DiscussionLog,
)

logger = logging.getLogger(__name__)

# Bump when MAPPING changes in a way a client could see (a seat, role, phase or path).
MAPPING_VERSION = "2026-11-12.1"

PHASES = (
    "analysis",           # 1: the analysts, in parallel
    "research_debate",    # 2: bull vs bear
    "research_decision",  # 3: research manager
    "trading_plan",       # 4: trader
    "risk_debate",        # 5: aggressive / conservative / neutral
    "final_decision",     # 6: portfolio manager
)


@dataclass(frozen=True)
class Seat:
    """Who a graph node is in the discussion, and where its turn's text is."""

    agent_id: str
    role: str
    phase: str | None
    content_path: str | None  # dotted path in the node's output (== the final state field)


MAPPING: dict[str, Seat] = {
    "Market Analyst": Seat("market", "analyst", "analysis", "market_report"),
    "Sentiment Analyst": Seat("sentiment", "analyst", "analysis", "sentiment_report"),
    "News Analyst": Seat("news", "analyst", "analysis", "news_report"),
    "Fundamentals Analyst": Seat("fundamentals", "analyst", "analysis", "fundamentals_report"),
    "Bull Researcher": Seat(
        "bull", "researcher", "research_debate", "investment_debate_state.current_response"),
    "Bear Researcher": Seat(
        "bear", "researcher", "research_debate", "investment_debate_state.current_response"),
    "Research Manager": Seat("research_manager", "manager", "research_decision", "investment_plan"),
    "Trader": Seat("trader", "trader", "trading_plan", "trader_investment_plan"),
    "Aggressive Analyst": Seat(
        "aggressive", "risk_debater", "risk_debate",
        "risk_debate_state.current_aggressive_response"),
    "Conservative Analyst": Seat(
        "conservative", "risk_debater", "risk_debate",
        "risk_debate_state.current_conservative_response"),
    "Neutral Analyst": Seat(
        "neutral", "risk_debater", "risk_debate", "risk_debate_state.current_neutral_response"),
    "Portfolio Manager": Seat(
        "portfolio_manager", "manager", "final_decision", "final_trade_decision"),
}

# Top-level nodes that are no one's turn: the Memory Log settles past decisions
# beside the analysts. Its LLM and tool events are dropped too.
IGNORED_NODES = frozenset({"Memory Log"})

UNKNOWN_SEAT = Seat("unknown", "unknown", None, None)

# A turn's text is capped at 32 KiB of UTF-8 (cut on a character boundary).
MAX_TURN_CONTENT_BYTES = 32 * 1024
# One task's turn text in all: max(1 MiB, expected_turns x 32 KiB), at most
# 4 MiB (see task_content_budget). The final decision's 32 KiB is reserved
# within it, so it is never cut short by the turns before it.
MIN_TASK_CONTENT_BYTES = 1024 * 1024
MAX_TASK_CONTENT_BYTES = 4 * 1024 * 1024
FINAL_AGENT_ID = "portfolio_manager"
# Tool calls kept per task; each has one tool_call and one tool_result event,
# so at most 300 tool events. Later calls are counted in the terminal
# task_status (and the transcript) as ``tools_dropped``.
MAX_TOOL_CALLS = 150

# The configure hook: registered once per process, at import.
DISCUSSION_EMITTER: ContextVar[TaskEventEmitter | None] = ContextVar(
    "tradingagents_api_discussion_emitter", default=None
)
register_configure_hook(DISCUSSION_EMITTER, True)


def expected_turns(
    n_analysts: int, debate_rounds: int, risk_rounds: int, checkpoint_enabled: bool = False
) -> int | None:
    """Turns a run takes: analysts + 2 per debate round + 3 per risk round + 3.

    None when the run may resume from a checkpoint, which skips turns already done.
    """
    if checkpoint_enabled:
        return None
    return n_analysts + 2 * debate_rounds + 3 * risk_rounds + 3


# The core's analyst keys -> their agent ids in the discussion.
ANALYST_AGENT_IDS = {
    "market": "market", "social": "sentiment", "news": "news", "fundamentals": "fundamentals",
}


def analyst_agent_ids(core_keys: Iterable[str]) -> list[str]:
    """The agent ids of the run's analysts, in the order given."""
    return [ANALYST_AGENT_IDS.get(k, k) for k in core_keys]


def task_content_budget(turns: int | None) -> int:
    """A task's turn-text budget: max(1 MiB, turns x 32 KiB), at most 4 MiB.

    ``turns`` is the run's ``expected_turns`` (None when unknown: 1 MiB).
    """
    wanted = (turns or 0) * MAX_TURN_CONTENT_BYTES
    return min(MAX_TASK_CONTENT_BYTES, max(MIN_TASK_CONTENT_BYTES, wanted))


def unmapped_nodes(node_names: Iterable[str]) -> set[str]:
    """Graph nodes neither mapped to a seat nor explicitly ignored."""
    return {n for n in node_names if n not in MAPPING and n not in IGNORED_NODES}


def cap_utf8(text: str, limit: int) -> tuple[str, bool]:
    """``text`` cut to at most ``limit`` UTF-8 bytes on a character boundary.

    A character UTF-8 cannot encode (a lone surrogate) becomes ``?``.
    """
    raw = text.encode("utf-8", "replace")
    if len(raw) <= limit:
        return raw.decode("utf-8"), False
    return raw[: max(0, limit)].decode("utf-8", "ignore"), True


def _dig(value: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _namespace_head(metadata: dict[str, Any]) -> str | None:
    ns = metadata.get("checkpoint_ns")
    if ns:
        return str(ns).split("|", 1)[0].split(":", 1)[0]
    node = metadata.get("langgraph_node")
    return str(node) if node else None


class TaskEventEmitter(BaseCallbackHandler):
    """A run's callback handler: turns LangChain callbacks into discussion events.

    LLM and retriever callbacks are switched off: prompts never reach it.
    (``ignore_agent`` stays off: LangChain gates the tool callbacks on it.)
    """

    ignore_llm = True
    ignore_chat_model = True
    ignore_retriever = True
    ignore_retry = True
    ignore_custom_event = True
    raise_error = False

    def __init__(self, log: DiscussionLog):
        self._log = log
        self._lock = threading.Lock()
        self._closed = False
        self._turn = 0
        self._phase: str | None = None
        # run_id -> (turn, node name, seat) for open top-level node runs.
        self._open: dict[UUID, tuple[int, str, Seat]] = {}
        # run_id -> (agent_id, tool, call_id) for open tool runs.
        self._tools: dict[UUID, tuple[str, str, str]] = {}
        self._tool_calls = 0
        self._tools_dropped = 0
        self._content_bytes = 0
        self._content_budget = task_content_budget(None)
        self._final_done = False

    @property
    def log(self) -> DiscussionLog:
        return self._log

    # -- run lifecycle (called by the analysis service) -------------------------

    def start(
        self,
        *,
        ticker: str,
        trade_date: str,
        analysts: Iterable[str],
        debate_rounds: int,
        risk_rounds: int,
        checkpoint_enabled: bool = False,
    ) -> None:
        """Emit ``run_started`` and ``task_status: running``."""
        analysts = list(analysts)
        turns = expected_turns(len(analysts), debate_rounds, risk_rounds, checkpoint_enabled)
        with self._lock:
            self._content_budget = task_content_budget(turns)
            self._log.start_clock()
            self._log.append(RUN_STARTED, {
                "ticker": ticker,
                "trade_date": trade_date,
                "analysts": analysts,
                "debate_rounds": debate_rounds,
                "risk_rounds": risk_rounds,
                "expected_turns": turns,
                "mapping_version": MAPPING_VERSION,
            })
            self._log.append(TASK_STATUS, {"status": STATUS_RUNNING})

    def close(self) -> None:
        """Abort every turn still open and stop listening. Idempotent.

        Called when the graph run returns or raises; the terminal
        ``task_status`` follows from the task manager.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for turn, _name, seat in sorted(self._open.values(), key=lambda t: t[0]):
                self._log.append(TURN_ABORTED, {"turn": turn, "agent_id": seat.agent_id})
            self._open.clear()
            self._tools.clear()
            if self._tools_dropped:
                self._log.note("tools_dropped", self._tools_dropped)

    # -- callbacks ---------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = kwargs.get("name") or (serialized or {}).get("name")
            md = metadata or {}
            if not name or md.get("checkpoint_ns") or name != md.get("langgraph_node"):
                return
            if name in IGNORED_NODES:
                return
            seat = MAPPING.get(name, UNKNOWN_SEAT)
            with self._lock:
                if self._closed:
                    return
                self._turn += 1
                if seat.phase is not None and seat.phase != self._phase:
                    self._phase = seat.phase
                    self._log.append(PHASE_CHANGED, {"phase": seat.phase})
                self._open[run_id] = (self._turn, name, seat)
                self._log.append(TURN_STARTED, {
                    "turn": self._turn, "agent_id": seat.agent_id,
                    "role": seat.role, "phase": seat.phase,
                })
        except Exception:  # noqa: BLE001 - a capture failure must never fail the run
            logger.debug("discussion capture: on_chain_start failed", exc_info=True)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            with self._lock:
                entry = self._open.pop(run_id, None)
                if entry is None or self._closed:
                    return
                turn, _name, seat = entry
                content: str | None = None
                truncated = False
                if seat.content_path is not None:
                    value = _dig(outputs, seat.content_path)
                    text = value if isinstance(value, str) else ""
                    final = seat.agent_id == FINAL_AGENT_ID
                    # Every turn before the final decision leaves its 32 KiB free.
                    reserve = 0 if final or self._final_done else MAX_TURN_CONTENT_BYTES
                    budget = max(0, self._content_budget - reserve - self._content_bytes)
                    content, truncated = cap_utf8(text, min(MAX_TURN_CONTENT_BYTES, budget))
                    self._content_bytes += len(content.encode("utf-8"))
                    self._final_done = self._final_done or final
                self._log.append(TURN_COMPLETED, {
                    "turn": turn, "agent_id": seat.agent_id,
                    "content": content, "truncated": truncated,
                })
        except Exception:  # noqa: BLE001
            logger.debug("discussion capture: on_chain_end failed", exc_info=True)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            with self._lock:
                entry = self._open.pop(run_id, None)
                if entry is None or self._closed:
                    return
                turn, _name, seat = entry
                self._log.append(TURN_ABORTED, {"turn": turn, "agent_id": seat.agent_id})
        except Exception:  # noqa: BLE001
            logger.debug("discussion capture: on_chain_error failed", exc_info=True)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            tool = str(kwargs.get("name") or (serialized or {}).get("name") or "unknown")
            head = _namespace_head(metadata or {})
            if head in IGNORED_NODES:
                return
            seat = MAPPING.get(head or "", UNKNOWN_SEAT)
            with self._lock:
                if self._closed:
                    return
                if self._tool_calls >= MAX_TOOL_CALLS:
                    self._tools_dropped += 1
                    return
                self._tool_calls += 1
                call_id = str(self._tool_calls)
                self._tools[run_id] = (seat.agent_id, tool, call_id)
                self._log.append(TOOL_CALL, {
                    "agent_id": seat.agent_id, "tool": tool, "call_id": call_id,
                })
        except Exception:  # noqa: BLE001
            logger.debug("discussion capture: on_tool_start failed", exc_info=True)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_done(run_id, ok=True)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_done(run_id, ok=False)

    def _tool_done(self, run_id: UUID, ok: bool) -> None:
        try:
            with self._lock:
                entry = self._tools.pop(run_id, None)
                if entry is None or self._closed:
                    return
                agent_id, tool, call_id = entry
                self._log.append(TOOL_RESULT, {
                    "agent_id": agent_id, "tool": tool, "call_id": call_id, "ok": ok,
                })
        except Exception:  # noqa: BLE001
            logger.debug("discussion capture: tool event failed", exc_info=True)
