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
appended with ``"\\n"`` to ``history`` and to the speaker's ``<side>_history``).
The speakers' side histories are boundary oracles: walking the speakers in the
graph's order, a boundary is accepted only where ``history`` and that speaker's
side history both continue with the same ``"\\n" + turn``, which splits exactly
even when a turn quotes another speaker's prefix. Without every side history, a
cut at every prefix is exact only when the run's known round count confirms it.
``rebuilt.split_exact`` is true only when every boundary was confirmed (at 1
round with every side history, one turn per side is exact). Otherwise each side
is one turn, or, with sides missing, each piece of ``history`` goes to the
speaker whose prefix it starts with; ``split_exact`` is false and no text is dropped.

Beside the transcript (not in ``run_started``, which stays equal to live capture's):
``investment_horizon`` is always None, as the core's log keeps no
``portfolio_context``; ``time_horizon`` is the portfolio manager's, read from the
logged ``final_trade_decision`` (None when it gave none).

Empty fields: with ``run_settings`` the run's seats are known, and a seat whose
field is empty keeps a turn with ``""`` content, as live capture records an
empty ``on_chain_end`` (the analysts named in the settings, the research
manager, the trader and the portfolio manager; a debate turn always holds its
speaker's prefix). Without settings, an empty field is skipped, and the
analysts are taken in the default order market, sentiment, news, fundamentals.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from tradingagents.api.core.discussion_log import SCHEMA_VERSION
from tradingagents.api.domain.services.decision_service import pm_time_horizon
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


_Turns = list[tuple[str, str]]


def _starts_turn(text: str, at: int, prefixes: Sequence[str]) -> bool:
    return any(text.startswith("\n" + p, at) for p in prefixes)


def _oracle_split(history: str, sides: list[str], speakers: Sequence[_Speaker]) -> _Turns | None:
    """The turns of ``history``, each boundary confirmed by its speaker's side history.

    Every agent writes ``"\n" + turn`` to both ``history`` and its own side
    history. Walking the speakers in the graph's order, a turn ends at the first
    ``"\n" + <next speaker's prefix>`` (or the end) where the speaker's side
    history continues with ``"\n" + turn`` and then its own next turn (or ends).
    A cut short of the true end leaves the side on the next speaker's prefix,
    never its own, so the first accepted cut is the true one: a turn quoting
    another speaker's prefix is still split exactly. None when the log disagrees.

    Linear time: the cuts are found lazily, and each character of a turn is
    compared with its side once (``matched`` only grows; a mismatch below a
    cut fails every later cut too).
    """
    count = len(speakers)
    cut_patterns = [
        re.compile("\n(?=" + "|".join(re.escape(p) for p in prefixes) + ")")
        for _agent_id, _keys, prefixes in speakers
    ]
    side_at = [0] * count
    turns: _Turns = []
    at = 0
    while at < len(history):
        k = len(turns) % count
        agent_id, _keys, own = speakers[k]
        side, start = sides[k], side_at[k]
        if not _starts_turn(history, at, own) or not _starts_turn(side, start, own):
            return None
        following = cut_patterns[(len(turns) + 1) % count].finditer(history, at + 1)
        matched = 0  # history[at + 1:][:matched] == side[start + 1:][:matched]
        for cut in itertools.chain((m.start() for m in following), (len(history),)):
            length = cut - at - 1
            after = start + 1 + length
            if after > len(side):
                return None  # the side ends inside this cut, and every later one
            if after != len(side) and not _starts_turn(side, after, own):
                continue
            if history[at + 1 + matched:cut] != side[start + 1 + matched:after]:
                return None
            matched = length
            break
        else:
            return None
        turns.append((agent_id, history[at + 1:cut]))
        side_at[k] = after
        at = cut
    if any(side_at[k] != len(side) for k, side in enumerate(sides)):
        return None
    return turns


def _pieces(history: str, speakers: Sequence[_Speaker]) -> list[str] | None:
    """``history`` cut at every ``"\n" + <a speaker's prefix>``; None if they do not rejoin."""
    prefixes = [p for _, _, ps in speakers for p in ps]
    pattern = "\n(?=" + "|".join(re.escape(p) for p in prefixes) + ")"
    pieces = re.split(pattern, history)
    lead = ""
    if history.startswith("\n"):  # the agents append "\n" + turn to an empty history
        lead, pieces = "\n", pieces[1:] if pieces and pieces[0] == "" else pieces
    if not pieces or lead + "\n".join(pieces) != history:
        return None
    return pieces


def _counted_split(
    history: str, speakers: Sequence[_Speaker], rounds: int | None
) -> _Turns | None:
    """The cut at every prefix, confirmed by the known round count, or None.

    Every true boundary is a cut, so as many cuts as the run had turns, each
    on its speaker in the graph's order, leaves no room for a quoted prefix.
    """
    if rounds is None:
        return None
    pieces = _pieces(history, speakers)
    if pieces is None or len(pieces) != len(speakers) * rounds:
        return None
    turns: _Turns = []
    for index, piece in enumerate(pieces):
        agent_id, _keys, own = speakers[index % len(speakers)]
        if not piece.startswith(own):
            return None
        turns.append((agent_id, piece))
    return turns


def _by_prefix(history: str, speakers: Sequence[_Speaker]) -> _Turns:
    """Unverified: each piece of ``history`` to the speaker whose prefix it starts with.

    A piece with no speaker's prefix (text before the first turn) goes to the
    first speaker, so no text is dropped.
    """
    text = history[1:] if history.startswith("\n") else history
    prefixes = [p for _, _, ps in speakers for p in ps]
    pattern = "\n(?=" + "|".join(re.escape(p) for p in prefixes) + ")"
    turns: _Turns = []
    for piece in re.split(pattern, text):
        owner = next((agent_id for agent_id, _keys, own in speakers if piece.startswith(own)),
                     speakers[0][0])
        turns.append((owner, piece))
    return turns


def _split_debate(
    debate: dict[str, Any], speakers: Sequence[_Speaker], rounds: int | None
) -> tuple[_Turns, bool]:
    """The debate's turns ``(agent_id, text)`` and whether every boundary was confirmed.

    In order: one turn per side at 1 round, when ``history`` is the sides one
    after the other (exact); the side histories as boundary oracles (exact);
    the cut at every prefix with the known round count (exact). Otherwise not
    exact: one turn per side when every side is there, else each piece of
    ``history`` to the speaker whose prefix it starts with. Never drops text of
    ``history``: the sides stand in for it only when they hold all of it.
    """
    history = _text(debate.get("history"))
    sides = [_side_history(debate, keys) for _agent_id, keys, _prefixes in speakers]
    if not history and not any(sides):
        return [], True  # nothing to split
    if all(sides):
        if rounds == 1 and "".join(sides) == history:
            return _per_side(speakers, sides), True
        turns = _oracle_split(history, sides, speakers)
        if turns is not None and (rounds is None or len(turns) == len(speakers) * rounds):
            return turns, True
    counted = _counted_split(history, speakers, rounds)
    if counted is not None:
        return counted, True
    if not history:
        return _per_side(speakers, sides), False
    pieces = _by_prefix(history, speakers)
    if all(sides) and all(any(text in side for side in sides) for _agent_id, text in pieces):
        return _per_side(speakers, sides), False
    return pieces, False


def _per_side(speakers: Sequence[_Speaker], sides: list[str]) -> _Turns:
    """One turn per side: its whole side history, without the leading newline."""
    turns = []
    for (agent_id, _keys, _prefixes), side in zip(speakers, sides, strict=True):
        side = side[1:] if side.startswith("\n") else side
        if side:
            turns.append((agent_id, side))
    return turns


def _debate_turns(
    state: dict[str, Any], key: str, speakers: Sequence[_Speaker], rounds: int | None
) -> tuple[_Turns, bool]:
    """The debate's turns, and whether every boundary between them was confirmed."""
    debate = state.get(key)
    if not isinstance(debate, dict):
        return [], True
    return _split_debate(debate, speakers, rounds)


def _analyst_turns(state: dict[str, Any], core_keys: list[str] | None) -> list[tuple[str, str]]:
    """The analysts' turns: the run's, in its order (an empty report kept, as live
    capture records it), or without settings the default order, non-empty only."""
    agent_ids = analyst_agent_ids(core_keys) if core_keys is not None else list(DEFAULT_ANALYSTS)
    turns = []
    for agent_id in agent_ids:
        seat = SEATS.get(agent_id)
        if seat is None or seat.phase != "analysis" or seat.content_path is None:
            continue
        text = _text(state.get(seat.content_path))
        if text or core_keys is not None:
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
    if settings is None:
        # Who spoke is unknown: an empty field is taken as a seat the run never had.
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
        # As the /analyze result names them. The log has no portfolio_context, so
        # the request's horizon is unknown; the PM's is read from its decision.
        "investment_horizon": None,
        "time_horizon": pm_time_horizon(_text(state.get("final_trade_decision"))),
        "rebuilt": {
            "logged_at": _iso(logged_at),
            "settings_known": settings is not None,
            "split_exact": research_exact and risk_exact,
            "fingerprints": fingerprints(state),
        },
    }
