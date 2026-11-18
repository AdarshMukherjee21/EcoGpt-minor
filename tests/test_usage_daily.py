"""usage.py daily rollup regression tests.

Register rows via the real record() path so the day rollup is exercised
against the live schema, not by hand-written SQL inserts.
"""
from __future__ import annotations

from proxy import usage as usage_log


def _rows(db_path):
    usage_log.record(db_path, model="m-a", route="anthropic", stream=False, status=200,
                     usage={"input_tokens": 10, "output_tokens": 2}, cost="x-cost: $0.50",
                     duration_ms=100)
    usage_log.record(db_path, model="m-b", route="responses", stream=True, status=200,
                     usage={"input_tokens": 20, "output_tokens": 4}, cost="x-cost: $1.25",
                     duration_ms=100)
    # no cost header on this one
    usage_log.record(db_path, model="m-a", route="anthropic", stream=False, status=400,
                     usage=None, cost=None, duration_ms=50)


def test_daily_sums_costs_across_rows_same_day(tmp_path):
    """Regression for the N+1 rewrite: costs aggregate per day over ALL rows."""
    db = tmp_path / "usage.db"
    usage_log.init_db(db)
    _rows(db)
    days = usage_log.daily(db, 30)
    assert len(days) == 1
    day = days[0]
    assert day["requests"] == 3
    assert day["cost"] == 1.75
    assert day["input_tokens"] == 30
    assert day["output_tokens"] == 6
    assert day["compressed"] == 0


def test_daily_days_window_returns_most_recent(tmp_path):
    db = tmp_path / "usage.db"
    usage_log.init_db(db)
    _rows(db)
    # The /api/usage/daily endpoint clamps days >= 1; the window is the most
    # recent N days, however many exist.
    days = usage_log.daily(db, 1)
    assert len(days) == 1
    assert days[0]["requests"] == 3
