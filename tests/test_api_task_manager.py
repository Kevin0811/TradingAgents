"""Tests for the TradingAgents API TaskManager.

Covers:
- Task creation and deduplication
- Task lifecycle (PENDING → QUEUED → PROCESSING → COMPLETED/FAILED)
- Task lookup, listing, deletion
- TTL cleanup and capacity enforcement
- Counting properties (active_task_count, processing_count, total_task_count)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(ticker="AAPL", trade_date="2026-06-01", asset_type="stock"):
    return TaskCreateRequest(
        ticker=ticker,
        trade_date=trade_date,
        asset_type=asset_type,
    )


# ---------------------------------------------------------------------------
# Task creation and deduplication
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskCreation:
    """Task creation must return a new task with correct defaults."""

    def test_create_task_returns_new_task(self):
        mgr = TaskManager()
        req = _make_request()
        task, is_new = mgr.create_task(req)

        assert is_new is True
        assert task.status == TaskStatus.PENDING
        assert task.ticker == "AAPL"
        assert task.trade_date == "2026-06-01"
        assert task.asset_type == "stock"
        assert task.task_id is not None
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.completed_at is None
        assert task.error is None
        assert task.result is None

    def test_create_task_deduplicates_active_pending(self):
        mgr = TaskManager()
        req = _make_request()
        first, is_new1 = mgr.create_task(req)
        second, is_new2 = mgr.create_task(req)

        assert is_new1 is True
        assert is_new2 is False
        assert first.task_id == second.task_id

    def test_create_task_deduplicates_active_processing(self):
        mgr = TaskManager()
        req = _make_request()
        first, _ = mgr.create_task(req)
        mgr.update_task(first.task_id, status=TaskStatus.PROCESSING)

        second, is_new2 = mgr.create_task(req)
        assert is_new2 is False
        assert first.task_id == second.task_id

    def test_create_task_deduplicates_active_queued(self):
        mgr = TaskManager()
        req = _make_request()
        first, _ = mgr.create_task(req)
        mgr.update_task(first.task_id, status=TaskStatus.QUEUED)

        second, is_new2 = mgr.create_task(req)
        assert is_new2 is False
        assert first.task_id == second.task_id

    def test_create_task_allows_new_after_completed(self):
        mgr = TaskManager()
        req = _make_request()
        first, _ = mgr.create_task(req)
        mgr.update_task(first.task_id, status=TaskStatus.COMPLETED)

        second, is_new2 = mgr.create_task(req)
        assert is_new2 is True
        assert first.task_id != second.task_id

    def test_create_task_allows_new_after_failed(self):
        mgr = TaskManager()
        req = _make_request()
        first, _ = mgr.create_task(req)
        mgr.update_task(first.task_id, status=TaskStatus.FAILED)

        second, is_new2 = mgr.create_task(req)
        assert is_new2 is True
        assert first.task_id != second.task_id

    def test_create_task_different_ticker_is_new(self):
        mgr = TaskManager()
        req1 = _make_request(ticker="AAPL")
        req2 = _make_request(ticker="MSFT")

        first, _ = mgr.create_task(req1)
        second, is_new2 = mgr.create_task(req2)

        assert is_new2 is True
        assert first.task_id != second.task_id

    def test_create_task_different_date_is_new(self):
        mgr = TaskManager()
        req1 = _make_request(trade_date="2026-06-01")
        req2 = _make_request(trade_date="2026-06-02")

        first, _ = mgr.create_task(req1)
        second, is_new2 = mgr.create_task(req2)

        assert is_new2 is True
        assert first.task_id != second.task_id

    def test_create_task_different_asset_type_is_new(self):
        mgr = TaskManager()
        req1 = _make_request(asset_type="stock")
        req2 = _make_request(asset_type="crypto")

        first, _ = mgr.create_task(req1)
        second, is_new2 = mgr.create_task(req2)

        assert is_new2 is True
        assert first.task_id != second.task_id


# ---------------------------------------------------------------------------
# Task capacity enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskCapacity:
    """Task queue must enforce max_tasks limit."""

    def test_raises_when_queue_full(self):
        mgr = TaskManager(max_tasks=2)
        mgr.create_task(_make_request(ticker="A"))
        mgr.create_task(_make_request(ticker="B"))

        with pytest.raises(ValueError, match="Task queue is full"):
            mgr.create_task(_make_request(ticker="C"))

    def test_cleanup_frees_capacity(self):
        mgr = TaskManager(max_tasks=2, ttl_minutes=0)
        req_a = _make_request(ticker="A")
        req_b = _make_request(ticker="B")

        task_a, _ = mgr.create_task(req_a)
        task_b, _ = mgr.create_task(req_b)

        # Mark both as completed so cleanup can remove them
        mgr.update_task(task_a.task_id, status=TaskStatus.COMPLETED)
        mgr.update_task(task_b.task_id, status=TaskStatus.COMPLETED)

        # Force completed_at to be old enough for TTL
        old = datetime.now() - timedelta(minutes=1)
        mgr.update_task(task_a.task_id, completed_at=old)
        mgr.update_task(task_b.task_id, completed_at=old)

        # Cleanup should remove both
        removed = mgr.cleanup()
        assert removed == 2

        # Now we can create a new task
        task_c, is_new = mgr.create_task(_make_request(ticker="C"))
        assert is_new is True


# ---------------------------------------------------------------------------
# Task lookup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskLookup:
    """get_task() and find_active_task() must work correctly."""

    def test_get_task_returns_task(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        found = mgr.get_task(task.task_id)
        assert found is not None
        assert found.task_id == task.task_id

    def test_get_task_returns_none_for_missing(self):
        mgr = TaskManager()
        assert mgr.get_task("nonexistent") is None

    def test_find_active_task_returns_none_when_no_tasks(self):
        mgr = TaskManager()
        assert mgr.find_active_task("AAPL", "2026-06-01", "stock") is None

    def test_find_active_task_returns_none_for_completed(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.COMPLETED)
        assert mgr.find_active_task("AAPL", "2026-06-01", "stock") is None


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskListing:
    """list_tasks() must filter and sort correctly."""

    def test_list_all_tasks(self):
        mgr = TaskManager()
        mgr.create_task(_make_request(ticker="AAPL"))
        mgr.create_task(_make_request(ticker="MSFT"))
        tasks = mgr.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filter_by_status(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="AAPL"))
        t2, _ = mgr.create_task(_make_request(ticker="MSFT"))
        mgr.update_task(t2.task_id, status=TaskStatus.COMPLETED)

        pending = mgr.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"

        completed = mgr.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].ticker == "MSFT"

    def test_list_tasks_filter_by_ticker(self):
        mgr = TaskManager()
        mgr.create_task(_make_request(ticker="AAPL"))
        mgr.create_task(_make_request(ticker="AAPL", trade_date="2026-06-02"))
        mgr.create_task(_make_request(ticker="MSFT"))

        aapl_tasks = mgr.list_tasks(ticker="AAPL")
        assert len(aapl_tasks) == 2
        assert all(t.ticker == "AAPL" for t in aapl_tasks)

    def test_list_tasks_sorted_newest_first(self):
        import time
        mgr = TaskManager()
        mgr.create_task(_make_request(ticker="A"))
        time.sleep(0.01)  # Ensure different created_at timestamps
        mgr.create_task(_make_request(ticker="B"))
        tasks = mgr.list_tasks()
        # Newest first (B was created after A)
        assert tasks[0].ticker == "B"
        assert tasks[1].ticker == "A"


# ---------------------------------------------------------------------------
# Task update
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskUpdate:
    """update_task() must modify fields and handle completion timestamps."""

    def test_update_task_status(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.PROCESSING)
        updated = mgr.get_task(task.task_id)
        assert updated.status == TaskStatus.PROCESSING

    def test_update_task_sets_completed_at_on_completion(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        assert task.completed_at is None
        mgr.update_task(task.task_id, status=TaskStatus.COMPLETED)
        updated = mgr.get_task(task.task_id)
        assert updated.completed_at is not None

    def test_update_task_sets_completed_at_on_failure(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.FAILED, error="boom")
        updated = mgr.get_task(task.task_id)
        assert updated.completed_at is not None
        assert updated.error == "boom"

    def test_update_task_does_not_set_completed_at_for_non_terminal(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.QUEUED)
        updated = mgr.get_task(task.task_id)
        assert updated.completed_at is None

    def test_update_task_returns_false_for_missing(self):
        mgr = TaskManager()
        assert mgr.update_task("nonexistent", status=TaskStatus.COMPLETED) is False

    def test_update_task_updates_updated_at(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        original_updated = task.updated_at
        import time
        time.sleep(0.01)
        mgr.update_task(task.task_id, status=TaskStatus.PROCESSING)
        updated = mgr.get_task(task.task_id)
        assert updated.updated_at > original_updated


# ---------------------------------------------------------------------------
# Task deletion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskDeletion:
    """delete_task() must remove tasks and return False for missing."""

    def test_delete_task_removes_it(self):
        mgr = TaskManager()
        task, _ = mgr.create_task(_make_request())
        assert mgr.delete_task(task.task_id) is True
        assert mgr.get_task(task.task_id) is None

    def test_delete_task_returns_false_for_missing(self):
        mgr = TaskManager()
        assert mgr.delete_task("nonexistent") is False


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskCleanup:
    """cleanup() must remove expired tasks."""

    def test_cleanup_removes_expired_completed(self):
        mgr = TaskManager(ttl_minutes=1)
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.COMPLETED)
        # Force completed_at to be old
        old = datetime.now() - timedelta(minutes=2)
        mgr.update_task(task.task_id, completed_at=old)

        removed = mgr.cleanup()
        assert removed == 1
        assert mgr.get_task(task.task_id) is None

    def test_cleanup_keeps_recent_completed(self):
        mgr = TaskManager(ttl_minutes=60)
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.COMPLETED)

        removed = mgr.cleanup()
        assert removed == 0
        assert mgr.get_task(task.task_id) is not None

    def test_cleanup_removes_stale_pending(self):
        mgr = TaskManager(ttl_minutes=1)
        task, _ = mgr.create_task(_make_request())
        # Force created_at to be old
        old = datetime.now() - timedelta(minutes=2)
        mgr.update_task(task.task_id, created_at=old)

        removed = mgr.cleanup()
        assert removed == 1
        assert mgr.get_task(task.task_id) is None

    def test_cleanup_does_not_remove_failed_recent(self):
        mgr = TaskManager(ttl_minutes=60)
        task, _ = mgr.create_task(_make_request())
        mgr.update_task(task.task_id, status=TaskStatus.FAILED)

        removed = mgr.cleanup()
        assert removed == 0
        assert mgr.get_task(task.task_id) is not None


# ---------------------------------------------------------------------------
# Counting properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskCounts:
    """Counting properties must reflect current state."""

    def test_active_task_count_includes_pending_processing_queued(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="A"))
        t2, _ = mgr.create_task(_make_request(ticker="B"))
        t3, _ = mgr.create_task(_make_request(ticker="C"))
        mgr.update_task(t2.task_id, status=TaskStatus.PROCESSING)
        mgr.update_task(t3.task_id, status=TaskStatus.QUEUED)

        assert mgr.active_task_count == 3

    def test_active_task_count_excludes_completed(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="A"))
        t2, _ = mgr.create_task(_make_request(ticker="B"))
        mgr.update_task(t2.task_id, status=TaskStatus.COMPLETED)

        assert mgr.active_task_count == 1

    def test_active_task_count_excludes_failed(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="A"))
        t2, _ = mgr.create_task(_make_request(ticker="B"))
        mgr.update_task(t2.task_id, status=TaskStatus.FAILED)

        assert mgr.active_task_count == 1

    def test_processing_count_only_includes_processing(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="A"))
        t2, _ = mgr.create_task(_make_request(ticker="B"))
        t3, _ = mgr.create_task(_make_request(ticker="C"))
        mgr.update_task(t2.task_id, status=TaskStatus.PROCESSING)
        mgr.update_task(t3.task_id, status=TaskStatus.QUEUED)

        assert mgr.processing_count == 1

    def test_total_task_count_includes_all(self):
        mgr = TaskManager()
        mgr.create_task(_make_request(ticker="A"))
        mgr.create_task(_make_request(ticker="B"))
        mgr.create_task(_make_request(ticker="C"))

        assert mgr.total_task_count == 3

    def test_counts_update_after_deletion(self):
        mgr = TaskManager()
        t1, _ = mgr.create_task(_make_request(ticker="A"))
        t2, _ = mgr.create_task(_make_request(ticker="B"))
        assert mgr.total_task_count == 2

        mgr.delete_task(t1.task_id)
        assert mgr.total_task_count == 1