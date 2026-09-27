"""YahooSymbolSource: how the supported-symbols lists are fetched from yfinance.

yfinance is replaced by fakes throughout — no test here touches the network.
The fakes answer the screener the way Yahoo does (at most 250 rows a page,
paged by ``offset``) so pagination, the page cap and the query filters are
exercised against the real ``EquityQuery`` / ``ETFQuery`` objects.
"""

from __future__ import annotations

import pandas as pd
import pytest
from yfinance import EquityQuery, ETFQuery

from tradingagents.api.infrastructure.yahoo_symbol_source import (
    MARKET_EXCHANGES,
    PAGE_RETRIES,
    PAGE_RETRY_BASE_DELAY_SECONDS,
    PAGE_SIZE,
    RETRY_AFTER_MAX_SECONDS,
    US_OTC_EXCHANGES,
    SymbolSourceError,
    YahooSymbolSource,
    http_status,
)


def _exchange_of(query_dict: dict) -> str:
    """Pull the ``exchange`` operand out of a (possibly nested) query dict."""
    if query_dict["operator"] == "EQ" and query_dict["operands"][0] == "exchange":
        return query_dict["operands"][1]
    for operand in query_dict["operands"]:
        if isinstance(operand, dict):
            found = _exchange_of(operand)
            if found:
                return found
    return ""


class FakeScreen:
    """Stands in for ``yfinance.screen``: serves canned quotes page by page."""

    def __init__(self, data: dict[tuple[str, str], list[dict]], report_total: bool = True):
        self.data = data
        self.report_total = report_total
        self.total_override: int | None = None
        self.calls: list[dict] = []

    def __call__(self, query, offset=0, size=25, sortField=None, sortAsc=None):
        kind = "etf" if isinstance(query, ETFQuery) else "equity"
        assert isinstance(query, (EquityQuery, ETFQuery))
        query_dict = query.to_dict()
        exchange = _exchange_of(query_dict)
        self.calls.append(
            {
                "kind": kind,
                "exchange": exchange,
                "offset": offset,
                "size": size,
                "query": query_dict,
            }
        )
        rows = self.data.get((exchange, kind), [])
        result = {"quotes": rows[offset : offset + self._page_rows(size)], "count": size}
        if self.report_total:
            result["total"] = self.total_override if self.total_override is not None else len(rows)
        return result

    def _page_rows(self, size: int) -> int:
        return size


class TrimmingScreen(FakeScreen):
    """A screener that serves fewer rows per page than asked for."""

    def __init__(self, data, page_rows: int, **kwargs):
        super().__init__(data, **kwargs)
        self.page_rows = page_rows

    def _page_rows(self, size: int) -> int:
        return min(size, self.page_rows)


def _quotes(prefix: str, n: int, suffix: str = "", **extra) -> list[dict]:
    return [
        {"symbol": f"{prefix}{i:04d}{suffix}", "longName": f"{prefix} Co {i}", **extra}
        for i in range(n)
    ]


def _source(screen=None, lookup_factory=None, **kwargs) -> tuple[YahooSymbolSource, list]:
    sleeps: list[float] = []
    kwargs.setdefault("page_delay_seconds", 0.5)
    src = YahooSymbolSource(
        screen=screen, lookup_factory=lookup_factory, sleep=sleeps.append, **kwargs
    )
    return src, sleeps


# ---------------------------------------------------------------------------
# Screener: pagination, filters, page cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScreenerPagination:
    def test_paginates_until_an_empty_page_without_a_total(self):
        screen = FakeScreen({("TAI", "equity"): _quotes("", 600, ".TW")}, report_total=False)
        src, sleeps = _source(screen)

        entries = src.fetch("tw")

        tai_calls = [c for c in screen.calls if c["exchange"] == "TAI" and c["kind"] == "equity"]
        # Without a total a short page proves nothing: only an empty one ends it.
        assert [c["offset"] for c in tai_calls] == [0, 250, 500, 600]
        assert all(c["size"] == PAGE_SIZE == 250 for c in tai_calls)
        assert len(entries) == 600
        # A pause before every request but the first.
        assert sleeps == [0.5] * (len(screen.calls) - 1)

    def test_short_pages_do_not_end_the_query_before_the_total(self):
        # Yahoo trims a page (200 rows for size=250): keep paging to the total.
        screen = TrimmingScreen({("TAI", "equity"): _quotes("", 500, ".TW")}, page_rows=200)
        src, _ = _source(screen)

        entries = src.fetch("tw")

        tai_calls = [c for c in screen.calls if c["exchange"] == "TAI" and c["kind"] == "equity"]
        assert [c["offset"] for c in tai_calls] == [0, 200, 400]
        assert len(entries) == 500

    def test_stops_at_total_without_an_extra_empty_request(self):
        screen = FakeScreen({("TAI", "equity"): _quotes("", 500, ".TW")})
        src, _ = _source(screen)

        src.fetch("tw")

        tai_calls = [c for c in screen.calls if c["exchange"] == "TAI" and c["kind"] == "equity"]
        assert [c["offset"] for c in tai_calls] == [0, 250]

    def test_an_empty_page_short_of_the_total_fails_the_fetch(self):
        # Yahoo claims 900 rows but serves 500: a partial list must not be saved.
        screen = FakeScreen({("TAI", "equity"): _quotes("", 500, ".TW")})
        screen.total_override = 900
        src, _ = _source(screen)

        with pytest.raises(SymbolSourceError, match="empty page after 500 of 900 rows"):
            src.fetch("tw")

    def test_hitting_the_page_cap_fails_the_fetch(self):
        screen = FakeScreen({("TAI", "equity"): _quotes("", 2000, ".TW")})
        src, _ = _source(screen, max_pages=3)

        with pytest.raises(SymbolSourceError, match=r"3-page cap after 750 rows of 2000"):
            src.fetch("tw")

        tai_calls = [c for c in screen.calls if c["exchange"] == "TAI" and c["kind"] == "equity"]
        assert len(tai_calls) == 3

    def test_hitting_the_page_cap_without_a_total_fails_too(self):
        screen = FakeScreen({("TAI", "equity"): _quotes("", 2000, ".TW")}, report_total=False)
        src, _ = _source(screen, max_pages=3)

        with pytest.raises(SymbolSourceError, match="page cap"):
            src.fetch("tw")

    def test_equities_require_positive_market_cap_etfs_use_etf_query(self):
        screen = FakeScreen({})
        src, _ = _source(screen)

        with pytest.raises(SymbolSourceError):
            src.fetch("tw")  # nothing came back

        equity = next(c for c in screen.calls if c["kind"] == "equity")
        assert equity["query"] == {
            "operator": "AND",
            "operands": [
                {"operator": "EQ", "operands": ["exchange", "TAI"]},
                {"operator": "GT", "operands": ["intradaymarketcap", 0]},
            ],
        }
        etf = next(c for c in screen.calls if c["kind"] == "etf")
        assert etf["query"] == {"operator": "EQ", "operands": ["exchange", "TAI"]}

    def test_zero_market_cap_rows_are_dropped_client_side_too(self):
        rows = [
            {"symbol": "2330.TW", "longName": "TSMC", "marketCap": 1e12},
            {"symbol": "03000P.TW", "longName": "Some warrant", "marketCap": 0},
        ]
        src, _ = _source(FakeScreen({("TAI", "equity"): rows}))

        assert [e.symbol for e in src.fetch("tw")] == ["2330.TW"]

    @pytest.mark.parametrize("market", ["tw", "us", "jp"])
    def test_every_exchange_of_the_market_is_screened_for_both_types(self, market):
        screen = FakeScreen({(MARKET_EXCHANGES[market][0], "equity"): _quotes("X", 1)})
        src, _ = _source(screen)

        src.fetch(market)

        screened = {(c["exchange"], c["kind"]) for c in screen.calls}
        assert screened == {(ex, k) for ex in MARKET_EXCHANGES[market] for k in ("equity", "etf")}

    def test_us_otc_is_excluded_by_default(self):
        screen = FakeScreen({("NYQ", "equity"): _quotes("A", 1)})
        src, _ = _source(screen)

        src.fetch("us")

        assert not {c["exchange"] for c in screen.calls} & set(US_OTC_EXCHANGES)

    def test_us_otc_can_be_included(self):
        screen = FakeScreen({("NYQ", "equity"): _quotes("A", 1)})
        src, _ = _source(screen, include_otc=True)

        src.fetch("us")

        assert set(US_OTC_EXCHANGES) <= {c["exchange"] for c in screen.calls}

    def test_errors_propagate_so_the_caller_keeps_its_old_list(self):
        def boom(*args, **kwargs):
            raise RuntimeError("429 Too Many Requests")

        src, _ = _source(boom)
        with pytest.raises(RuntimeError, match="429"):
            src.fetch("tw")


# ---------------------------------------------------------------------------
# Mapping quotes to entries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntryMapping:
    def test_tw_equities_and_etfs(self):
        screen = FakeScreen(
            {
                ("TAI", "equity"): [
                    {"symbol": "2330.TW", "longName": "Taiwan Semiconductor", "exchange": "TAI"},
                    {"symbol": "2881A.TW", "shortName": "FUBON FIN PREF A", "exchange": "TAI"},
                ],
                ("TWO", "equity"): [{"symbol": "6488.TWO", "shortName": "GLOBALWAFERS"}],
                ("TAI", "etf"): [
                    {"symbol": "0050.TW", "longName": "Yuanta Taiwan Top 50 ETF"},
                    {"symbol": "00679B.TW", "shortName": "YUANTA US 20+ BOND"},
                ],
                ("TWO", "etf"): [{"symbol": "00635U.TWO", "displayName": "Gold ETF"}],
            }
        )
        src, _ = _source(screen)

        by_symbol = {e.symbol: e for e in src.fetch("tw")}

        assert set(by_symbol) == {
            "2330.TW",
            "2881A.TW",
            "6488.TWO",
            "0050.TW",
            "00679B.TW",
            "00635U.TWO",
        }
        assert by_symbol["2330.TW"].to_dict() == {
            "symbol": "2330.TW",
            "name": "Taiwan Semiconductor",
            "exchange": "TAI",
            "type": "equity",
            "market": "tw",
        }
        # Falls back to shortName/displayName, and to the queried exchange.
        assert by_symbol["6488.TWO"].name == "GLOBALWAFERS"
        assert by_symbol["6488.TWO"].exchange == "TWO"
        assert by_symbol["00635U.TWO"].name == "Gold ETF"
        assert by_symbol["0050.TW"].type == "etf"

    def test_entries_are_deduplicated_and_sorted(self):
        rows = [{"symbol": "b"}, {"symbol": "A"}, {"symbol": "B"}, {"symbol": ""}]
        src, _ = _source(FakeScreen({("NYQ", "equity"): rows}))

        assert [e.symbol for e in src.fetch("us")] == ["A", "B"]

    def test_source_label_names_yfinance(self):
        import yfinance

        src, _ = _source()
        assert src.source_label("tw") == f"yfinance {yfinance.__version__} screener"
        assert src.source_label("fx") == f"yfinance {yfinance.__version__} lookup"

    def test_defaults_to_yfinance_screen(self, monkeypatch):
        import yfinance

        screen = FakeScreen({("JPX", "equity"): [{"symbol": "7203.T", "longName": "Toyota"}]})
        monkeypatch.setattr(yfinance, "screen", screen)
        src = YahooSymbolSource(page_delay_seconds=0, sleep=lambda _: None)

        assert [e.symbol for e in src.fetch("jp")] == ["7203.T"]
        assert screen.calls


# ---------------------------------------------------------------------------
# Lookup: crypto and fx
# ---------------------------------------------------------------------------


class FakeLookup:
    """Stands in for ``yfinance.Lookup``: returns a symbol-indexed DataFrame."""

    instances: list[FakeLookup] = []

    def __init__(self, rows: dict[str, list[dict]]):
        self.rows = rows
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, query):
        self.query = query
        return self

    def _frame(self, kind: str, count: int) -> pd.DataFrame:
        self.calls.append((self.query, kind, count))
        rows = self.rows.get(kind, [])
        return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()

    def get_cryptocurrency(self, count=25):
        return self._frame("crypto", count)

    def get_currency(self, count=25):
        return self._frame("currency", count)


@pytest.mark.unit
class TestLookupMarkets:
    def test_crypto_keeps_usd_pairs_and_adds_core_seeds(self):
        lookup = FakeLookup(
            {
                "crypto": [
                    {"symbol": "BTC-USD", "shortName": "Bitcoin USD", "exchange": "CCC"},
                    {"symbol": "PEPE24478-USD", "shortName": "Pepe USD"},
                    {"symbol": "BTC-EUR", "shortName": "Bitcoin EUR"},
                    {"symbol": "ETHUSD", "shortName": "not Yahoo form"},
                ]
            }
        )
        src, _ = _source(lookup_factory=lookup)

        by_symbol = {e.symbol: e for e in src.fetch("crypto")}

        assert "BTC-EUR" not in by_symbol and "ETHUSD" not in by_symbol
        assert by_symbol["BTC-USD"].to_dict() == {
            "symbol": "BTC-USD",
            "name": "Bitcoin USD",
            "exchange": "CCC",
            "type": "crypto",
            "market": "crypto",
        }
        assert "PEPE24478-USD" in by_symbol
        # Core-supported bases are always present once the fetch succeeded.
        assert {"ETH-USD", "SOL-USD", "DOGE-USD"} <= set(by_symbol)
        assert lookup.calls[0][1] == "crypto"

    def test_fx_maps_every_usd_pair_to_the_short_form(self):
        lookup = FakeLookup(
            {
                "currency": [
                    {"symbol": "TWD=X", "shortName": "USD/TWD"},
                    {"symbol": "USDJPY=X", "shortName": "USD/JPY"},
                    {"symbol": "EURUSD=X", "shortName": "EUR/USD"},
                    {"symbol": "EURJPY=X", "shortName": "EUR/JPY"},  # cross: dropped
                    {"symbol": "GC=F", "shortName": "Gold"},  # not a currency
                ]
            }
        )
        src, _ = _source(lookup_factory=lookup)

        by_symbol = {e.symbol: e for e in src.fetch("fx")}

        assert {"TWD=X", "JPY=X", "EUR=X"} <= set(by_symbol)
        assert not any(s.startswith("USD") or len(s) != 5 for s in by_symbol)
        assert by_symbol["TWD=X"].to_dict() == {
            "symbol": "TWD=X",
            "name": "USD/TWD",
            "exchange": "CCY",
            "type": "currency",
            "market": "fx",
        }
        # Seeds: every currency the core maps, plus TWD.
        assert {"GBP=X", "KRW=X", "CNY=X"} <= set(by_symbol)

    def test_empty_lookup_is_a_failure_not_a_seed_only_list(self):
        src, _ = _source(lookup_factory=FakeLookup({}))
        with pytest.raises(SymbolSourceError):
            src.fetch("fx")
        with pytest.raises(SymbolSourceError):
            src.fetch("crypto")


# ---------------------------------------------------------------------------
# Per-request retries on 5xx / 429
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


def _http_error(status: int, headers: dict | None = None):
    """The exception yfinance's ``raise_for_status`` raises (curl_cffi's HTTPError)."""
    from curl_cffi.requests.exceptions import HTTPError

    return HTTPError(f"HTTP Error {status}: ", 0, _Response(status, headers))


class FlakyScreen(FakeScreen):
    """Fails the request at ``(exchange, kind, offset)`` with ``errors``, in order."""

    def __init__(self, data, fail_at: tuple[str, str, int], errors: list[Exception]):
        super().__init__(data)
        self.fail_at = fail_at
        self.errors = list(errors)

    def __call__(self, query, offset=0, size=25, sortField=None, sortAsc=None):
        kind = "etf" if isinstance(query, ETFQuery) else "equity"
        if (_exchange_of(query.to_dict()), kind, offset) == self.fail_at and self.errors:
            self.calls.append({"exchange": self.fail_at[0], "kind": kind, "offset": offset})
            raise self.errors.pop(0)
        return super().__call__(query, offset, size, sortField, sortAsc)


def _page_calls(screen, exchange="TAI", kind="equity"):
    return [c["offset"] for c in screen.calls if c["exchange"] == exchange and c["kind"] == kind]


@pytest.mark.unit
class TestRetries:
    def test_constants(self):
        assert (PAGE_RETRIES, PAGE_RETRY_BASE_DELAY_SECONDS) == (3, 2.0)

    def test_a_500_mid_query_is_retried_and_the_fetch_completes(self, caplog):
        screen = FlakyScreen(
            {("TAI", "equity"): _quotes("", 600, ".TW")},
            fail_at=("TAI", "equity", 250),
            errors=[_http_error(500), _http_error(503)],
        )
        src, sleeps = _source(screen, page_delay_seconds=0)

        entries = src.fetch("tw")

        assert len(entries) == 600
        assert _page_calls(screen) == [0, 250, 250, 250, 500]
        assert sleeps == [2.0, 5.0]  # backoff before retries 1 and 2
        assert "screener tw/TAI/equity offset=250 failed with HTTP 500" in caplog.text
        assert "retry 1 of 3 in 2.0s" in caplog.text
        assert "failed with HTTP 503" in caplog.text and "retry 2 of 3 in 5.0s" in caplog.text

    def test_a_page_that_keeps_failing_fails_the_market_after_three_retries(self):
        screen = FlakyScreen(
            {("TAI", "equity"): _quotes("", 600, ".TW")},
            fail_at=("TAI", "equity", 250),
            errors=[_http_error(500)] * 10,
        )
        src, sleeps = _source(screen, page_delay_seconds=0)

        with pytest.raises(Exception, match="HTTP Error 500"):
            src.fetch("tw")

        assert _page_calls(screen) == [0, 250, 250, 250, 250]  # 1 try + 3 retries
        assert sleeps == [2.0, 5.0, 10.0]

    def test_429_is_retried_including_yfinance_rate_limit_errors(self):
        from yfinance.exceptions import YFRateLimitError

        screen = FlakyScreen(
            {("TAI", "equity"): _quotes("", 10, ".TW")},
            fail_at=("TAI", "equity", 0),
            errors=[_http_error(429), YFRateLimitError()],
        )
        src, sleeps = _source(screen, page_delay_seconds=0)

        assert len(src.fetch("tw")) == 10
        assert sleeps == [2.0, 5.0]

    @pytest.mark.parametrize(
        "error", [_http_error(404), _http_error(400), _http_error(401), ValueError("bad json")]
    )
    def test_other_errors_are_not_retried(self, error):
        screen = FlakyScreen(
            {("TAI", "equity"): _quotes("", 10, ".TW")},
            fail_at=("TAI", "equity", 0),
            errors=[error],
        )
        src, sleeps = _source(screen, page_delay_seconds=0)

        with pytest.raises(type(error)):
            src.fetch("tw")
        assert _page_calls(screen) == [0]
        assert sleeps == []

    @pytest.mark.parametrize(
        ("retry_after", "slept"),
        [
            ("7", 7.0),
            ("0", 0.0),
            ("600", RETRY_AFTER_MAX_SECONDS),  # capped
            ("Wed, 21 Oct 2026 07:28:00 GMT", 2.0),  # date form: fall back to backoff
        ],
    )
    def test_retry_after_is_honoured(self, retry_after, slept):
        screen = FlakyScreen(
            {("TAI", "equity"): _quotes("", 10, ".TW")},
            fail_at=("TAI", "equity", 0),
            errors=[_http_error(503, {"Retry-After": retry_after})],
        )
        src, sleeps = _source(screen, page_delay_seconds=0)

        src.fetch("tw")
        assert sleeps == [slept]

    def test_lookup_errors_are_retried_too(self, caplog):
        from yfinance.exceptions import YFDataException

        lookup = FakeLookup({"currency": [{"symbol": "TWD=X", "shortName": "USD/TWD"}]})
        failures = [
            YFDataException(
                "USD: 'lookup' fetch returned error: {'code': 'Internal Server Error', "
                "'description': 'Server caught an exception'}"
            )
        ]

        def factory(query):
            if failures:
                raise failures.pop(0)
            return lookup(query)

        src, sleeps = _source(lookup_factory=factory, page_delay_seconds=0)

        assert "TWD=X" in {e.symbol for e in src.fetch("fx")}
        assert sleeps == [2.0]
        assert "lookup fx count=1000 failed with HTTP 500" in caplog.text

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (_http_error(502), 502),
            (RuntimeError("HTTP Error 500: "), 500),
            (RuntimeError("503 Server Error: Service Unavailable for url"), 503),
            (RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"), 503),
            (RuntimeError("Too Many Requests. Rate limited."), 429),
            (RuntimeError("no data"), None),
        ],
    )
    def test_http_status(self, error, status):
        assert http_status(error) == status
