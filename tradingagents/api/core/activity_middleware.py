"""ASGI middleware that reports Yahoo-backed requests to the ActivityMonitor.

The supported-symbols refresher only runs while TradingAgents is idle. Besides
analysis tasks (counted from the TaskManager), the requests that hit Yahoo
Finance themselves count as activity: ``/data/*`` (market data), the
per-analyst ``/analysts/*`` endpoints and the synchronous ``POST /analyze``,
which runs a whole analysis outside the TaskManager. The middleware only
counts them in and out; it never touches the request or response.
"""

from __future__ import annotations

from collections.abc import Callable

from tradingagents.api.domain.services.refresh_activity import ActivityMonitor


class YahooActivityMiddleware:
    """Counts in-flight Yahoo-backed requests and when the last one ended."""

    def __init__(self, app, monitor: ActivityMonitor, is_tracked: Callable[[str, str], bool]):
        self.app = app
        self.monitor = monitor
        self.is_tracked = is_tracked

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not self.is_tracked(
            scope.get("method", ""), scope.get("path", "")
        ):
            await self.app(scope, receive, send)
            return
        self.monitor.request_started()
        try:
            await self.app(scope, receive, send)
        finally:
            self.monitor.request_finished()


def yahoo_backed_paths(api_prefix: str) -> Callable[[str, str], bool]:
    """Predicate for the requests under ``api_prefix`` (``/api/v1``) that use Yahoo."""
    prefixes = (f"{api_prefix}/data/", f"{api_prefix}/analysts/")
    analyze = f"{api_prefix}/analyze"

    def is_tracked(method: str, path: str) -> bool:
        if path.startswith(prefixes):
            return True
        return method == "POST" and path.rstrip("/") == analyze

    return is_tracked
