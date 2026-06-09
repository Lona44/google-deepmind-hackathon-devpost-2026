"""Unit tests for the cost tracker.

Spec Section 4.2: per-experiment cap aborts mid-cascade; per-session cap
hard-stops the CLI between experiments. Both env-overridable.
"""

from __future__ import annotations

import pytest

from judge_panel.cost_tracker import (
    CostCapExceededError,
    PerExperimentTracker,
    PerSessionTracker,
)


class TestPerExperimentTracker:
    def test_records_costs(self):
        t = PerExperimentTracker(max_cost_usd=0.50)
        t.charge("auditor", 0.0058)
        t.charge("detector", 0.0152)
        assert t.total_cost_usd == pytest.approx(0.021)

    def test_aborts_when_cap_exceeded(self):
        t = PerExperimentTracker(max_cost_usd=0.05)
        t.charge("auditor", 0.02)
        t.charge("detector", 0.02)
        with pytest.raises(CostCapExceededError) as exc_info:
            t.charge("critic", 0.02)
        assert "0.06" in str(exc_info.value)

    def test_default_cap_from_env(self, monkeypatch):
        monkeypatch.setenv("JUDGE_PANEL_MAX_COST_PER_RUN", "1.25")
        t = PerExperimentTracker.from_env()
        assert t.max_cost_usd == 1.25

    def test_default_cap_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("JUDGE_PANEL_MAX_COST_PER_RUN", raising=False)
        t = PerExperimentTracker.from_env()
        assert t.max_cost_usd == 0.50

    def test_per_role_breakdown(self):
        t = PerExperimentTracker(max_cost_usd=1.0)
        t.charge("auditor", 0.01)
        t.charge("auditor", 0.005)  # e.g. retry
        t.charge("detector", 0.02)
        assert t.per_role_costs == {"auditor": 0.015, "detector": 0.02}


class TestPerSessionTracker:
    def test_accumulates_across_experiments(self):
        s = PerSessionTracker(max_cost_usd=5.0)
        s.record_experiment("run-1", 0.05)
        s.record_experiment("run-2", 0.05)
        assert s.total_cost_usd == 0.10
        assert s.experiment_count == 2

    def test_remaining_budget(self):
        s = PerSessionTracker(max_cost_usd=1.0)
        s.record_experiment("run-1", 0.30)
        assert s.remaining_budget_usd == pytest.approx(0.70)

    def test_would_exceed_check(self):
        s = PerSessionTracker(max_cost_usd=1.0)
        s.record_experiment("run-1", 0.95)
        assert s.would_exceed(0.10) is True
        assert s.would_exceed(0.04) is False

    def test_default_cap_from_env(self, monkeypatch):
        monkeypatch.setenv("JUDGE_PANEL_MAX_SESSION_COST", "12.50")
        s = PerSessionTracker.from_env()
        assert s.max_cost_usd == 12.50

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("JUDGE_PANEL_MAX_SESSION_COST", raising=False)
        s = PerSessionTracker.from_env()
        assert s.max_cost_usd == 5.0
