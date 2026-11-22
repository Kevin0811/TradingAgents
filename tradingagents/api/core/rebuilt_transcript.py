"""Rebuild a finished run's discussion transcript from the core's ``full_states_log``.

Runs from before live capture (Plan 08, 08-A) have no ``discussion_<task_id>.json``;
their ``<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json``
still holds every turn's text. :func:`build_rebuilt_transcript` folds such a
log into the transcript shape ``GET /analyze/tasks/{id}/transcript`` serves,
with the same seats, roles, phases, per-turn cap and task budget as live
capture, plus a ``rebuilt`` block.

Pure: no I/O, no graph, no model. A log has no tools and no timings, so every
turn has ``tools`` [], ``started_at``/``completed_at`` null and ``aborted`` false.

Debates are split from their ``history`` (each turn is ``"<Speaker> Analyst: <text>"``,
appended with ``"\\n"``). The split is kept only when it is provably exact: the
speakers take turns in the graph's order, the count matches the run's rounds
(when known), the turns joined with ``"\\n"`` give ``history`` back, and each
speaker's turns give back its own ``<side>_history``. Otherwise each side
becomes one turn (its whole side history) and ``rebuilt.split_exact`` is false.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from tradingagents.api.core.discussion_log import SCHEMA_VERSION
from tradingagents.api.infrastructure.discussion_capture import (
    FINAL_AGENT_ID,
    MAPPING,
    MAPPING_VERSION,
    MAX_TURN_CONTENT_BYTES,
    Seat,
    analyst_agent_ids,
    cap_utf8,
    expected_turns,
    task_content_budget,
)

STATUS_COMPLETED = "completed"

# The log fields fingerprinted (sha256 of the raw UTF-8 text). Same names as
# the /analyze result (AnalyzeResponse, AnalysisResult), so a client can prove
# a log belongs to one of its runs.
FINGERPRINT_FIELDS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)

# Analysts in the graph's default order, when the log has no run_settings.
DEFAULT_ANALYSTS = ("market", "sentiment", "news", "fundamentals")

# One seat per agent id, taken from the live MAPPING (never copied).
SEATS: dict[str, Seat] = {}
for _seat in MAPPING.values():
    SEATS.setdefault(_seat.agent_id, _seat)

# A debate's speakers in the graph's order (conditional_logic): agent id, the
# history keys (new name first, then an old upstream name) and the prefixes.
_Speaker = tuple[str, tuple[str, ...], tuple[str, ...]]
RESEARCH_SPEAKERS: tuple[_Speaker, ...] = (
    ("bull", ("bull_history",), ("Bull Analyst: ",)),
    ("bear", ("bear_history",), ("Bear Analyst: ",)),
)
RISK_SPEAKERS: tuple[_Speaker, ...] = (
    ("aggressive", ("aggressive_history", "risky_history"),
     ("Aggressive Analyst: ", "Risky Analyst: ")),
    ("conservative", ("conservative_history", "safe_history"),
     ("Conservative Analyst: ", "Safe Analyst: ")),
    ("neutral", ("neutral_history",), ("Neutral Analyst: ",)),
)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _round_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _settings(state: dict[str, Any]) -> tuple[list[str], int, int] | None:
    """(core analyst keys, debate rounds, risk rounds), or None when unknown."""
    settings = state.get("run_settings")
    if not isinstance(settings, dict):
        return None
    analysts = settings.get("analysts")
    debate = _round_count(settings.get("max_debate_rounds"))
    risk = _round_count(settings.get("max_risk_discuss_rounds"))
    if (not isinstance(analysts, list) or not all(isinstance(a, str) for a in analysts)
            or debate is None or risk is None):
        return None
    return analysts, debate, risk


def _side_history(debate: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _text(debate.get(key))
        if text:
            return text
    return ""


def _split_debate(
    debate: dict[str, Any], speakers: Sequence[_Speaker], rounds: int | None
) -> list[tuple[str, str]] | None:
    """The debate's turns ``(agent_id, text)`` split from ``history``, or None if inexact."""
    history = _text(debate.get("history"))
    if not history:
        return None
    prefixes = [p for _, _, ps in speakers for p in ps]
    pattern = "\n(?=" + "|".join(re.escape(p) for p in prefixes) + ")"
    pieces = re.split(pattern, history)
    lead = ""
    if history.startswith("\n"):  # the agents append "\n" + turn to an empty history
        lead, pieces = "\n", pieces[1:] if pieces and pieces[0] == "" else pieces
    if not pieces or lead + "\n".join(pieces) != history:
        return None
    if rounds is not None and len(pieces) != len(speakers) * rounds:
        return None
    turns: list[tuple[str, str]] = []
    for index, piece in enumerate(pieces):
        agent_id, _keys, own = speakers[index % len(speakers)]
        if not piece.startswith(own):
            return None
        turns.append((agent_id, piece))
    # Each speaker's turns must give back its own side history, when the log has one.
    for agent_id, keys, _prefixes in speakers:
        side = _side_history(debate, keys)
        if not side:
            continue
        own_turns = [text for a, text in turns if a == agent_id]
        if "\n" + "\n".join(own_turns) != side and "\n".join(own_turns) != side:
            return None
    return turns


def _fallback_debate(
    debate: dict[str, Any], speakers: Sequence[_Speaker]
) -> list[tuple[str, str]]:
    """One turn per side: its whole side history, without the leading newline."""
    turns = []
    for agent_id, keys, _prefixes in speakers:
        side = _side_history(debate, keys)
        if side.startswith("\n"):
            side = side[1:]
        if side:
            turns.append((agent_id, side))
    return turns


def _debate_turns(
    state: dict[str, Any], key: str, speakers: Sequence[_Speaker], rounds: int | None
) -> tuple[list[tuple[str, str]], bool]:
    """The debate's turns, and whether they are the exact split of its history."""
    debate = state.get(key)
    if not isinstance(debate, dict):
        return [], True
    split = _split_debate(debate, speakers, rounds)
    if split is not None:
        return split, True
    fallback = _fallback_debate(debate, speakers)
    # An empty debate is no split failure: there was nothing to split.
    exact = not fallback and not _text(debate.get("history"))
    return fallback, exact


def _analyst_turns(state: dict[str, Any], core_keys: list[str] | None) -> list[tuple[str, str]]:
    agent_ids = analyst_agent_ids(core_keys) if core_keys is not None else list(DEFAULT_ANALYSTS)
    turns = []
    for agent_id in agent_ids:
        seat = SEATS.get(agent_id)
        if seat is None or seat.phase != "analysis" or seat.content_path is None:
            continue
        text = _text(state.get(seat.content_path))
        if text:
            turns.append((agent_id, text))
    return turns


def _inferred_rounds(turns: list[tuple[str, str]], speakers: Sequence[_Speaker]) -> int | None:
    count = len(turns)
    return count // len(speakers) if count % len(speakers) == 0 else None


def fingerprints(state: dict[str, Any]) -> dict[str, str]:
    """sha256 hex of each present :data:`FINGERPRINT_FIELDS` text (raw, uncapped UTF-8).

    An old log's ``trader_investment_decision`` counts as ``trader_investment_plan``.
    A field that is missing, empty or not valid UTF-8 (a lone surrogate) has none.
    """
    out = {}
    for field in FINGERPRINT_FIELDS:
        text = state.get(field)
        if field == "trader_investment_plan" and not text:
            text = state.get("trader_investment_decision")  # old upstream key
        if not isinstance(text, str) or not text:
            continue
        try:
            raw = text.encode("utf-8")
        except UnicodeEncodeError:  # a lone surrogate: no UTF-8 text to compare
            continue
        out[field] = hashlib.sha256(raw).hexdigest()
    return out


def _iso(at: datetime | None) -> str | None:
    if at is None:
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return at.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_rebuilt_transcript(state: dict[str, Any], logged_at: datetime | None) -> dict[str, Any]:
    """The transcript of a ``full_states_log`` entry (see the module docstring)."""
    settings = _settings(state)
    core_keys, debate_rounds, risk_rounds = settings if settings else (None, None, None)

    research, research_exact = _debate_turns(
        state, "investment_debate_state", RESEARCH_SPEAKERS, debate_rounds)
    risk, risk_exact = _debate_turns(state, "risk_debate_state", RISK_SPEAKERS, risk_rounds)
    trader_text = _text(state.get("trader_investment_plan")) or _text(
        state.get("trader_investment_decision"))  # old upstream key

    analysts = _analyst_turns(state, core_keys)
    plan: list[tuple[str, str]] = [
        *analysts,
        *research,
        ("research_manager", _text(state.get("investment_plan"))),
        ("trader", trader_text),
        *risk,
        (FINAL_AGENT_ID, _text(state.get("final_trade_decision"))),
    ]
    plan = [(agent_id, text) for agent_id, text in plan if text]

    if settings is not None:
        analyst_ids = analyst_agent_ids(core_keys)
        expected = expected_turns(len(analyst_ids), debate_rounds, risk_rounds)
    else:
        analyst_ids = [agent_id for agent_id, _ in analysts]
        expected = len(plan)
        if research_exact:
            debate_rounds = _inferred_rounds(research, RESEARCH_SPEAKERS)
        if risk_exact:
            risk_rounds = _inferred_rounds(risk, RISK_SPEAKERS)

    # The live cap and budget: 32 KiB a turn, the task budget from expected
    # turns, the final decision's 32 KiB reserved until it has been written.
    budget = task_content_budget(expected)
    used = 0
    final_done = False
    turns = []
    for number, (agent_id, text) in enumerate(plan, start=1):
        seat = SEATS[agent_id]
        final = agent_id == FINAL_AGENT_ID
        reserve = 0 if final or final_done else MAX_TURN_CONTENT_BYTES
        left = max(0, budget - reserve - used)
        content, truncated = cap_utf8(text, min(MAX_TURN_CONTENT_BYTES, left))
        used += len(content.encode("utf-8"))
        final_done = final_done or final
        turns.append({
            "turn": number,
            "agent_id": agent_id,
            "role": seat.role,
            "phase": seat.phase,
            "started_at": None,
            "completed_at": None,
            "aborted": False,
            "content": content,
            "truncated": truncated,
            "tools": [],
        })

    return {
        "v": SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "status": STATUS_COMPLETED,
        "run_started": {
            "ticker": _text(state.get("company_of_interest")),
            "trade_date": _text(state.get("trade_date")),
            "analysts": analyst_ids,
            "debate_rounds": debate_rounds,
            "risk_rounds": risk_rounds,
            "expected_turns": expected,
            "mapping_version": MAPPING_VERSION,
        },
        "turns": turns,
        "tools_dropped": 0,
        "events_dropped": 0,
        "rebuilt": {
            "logged_at": _iso(logged_at),
            "settings_known": settings is not None,
            "split_exact": research_exact and risk_exact,
            "fingerprints": fingerprints(state),
        },
    }
