"""Cost tracking and guardrails for panel runs.

Spec Section 4.2. Two layers:
  PerExperimentTracker — aborts mid-cascade if a single experiment
    threatens to exceed the per-run cap.
  PerSessionTracker — stops the CLI between experiments once cumulative
    session cost exceeds the per-session cap.

Both caps overridable via env vars JUDGE_PANEL_MAX_COST_PER_RUN and
JUDGE_PANEL_MAX_SESSION_COST.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field


class CostCapExceededError(RuntimeError):
    """Raised when a role charge would push the experiment over its cap."""


@dataclass
class PerExperimentTracker:
    max_cost_usd: float
    total_cost_usd: float = 0.0
    per_role_costs: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @classmethod
    def from_env(cls) -> "PerExperimentTracker":
        cap = float(os.environ.get("JUDGE_PANEL_MAX_COST_PER_RUN", "0.50"))
        return cls(max_cost_usd=cap)

    def charge(self, role: str, cost_usd: float) -> None:
        new_total = self.total_cost_usd + cost_usd
        if new_total > self.max_cost_usd:
            raise CostCapExceededError(
                f"per-experiment cost cap exceeded: ${new_total:.4f} > ${self.max_cost_usd:.4f}"
            )
        self.total_cost_usd = new_total
        self.per_role_costs[role] += cost_usd


@dataclass
class PerSessionTracker:
    max_cost_usd: float
    total_cost_usd: float = 0.0
    experiment_count: int = 0

    @classmethod
    def from_env(cls) -> "PerSessionTracker":
        cap = float(os.environ.get("JUDGE_PANEL_MAX_SESSION_COST", "5.00"))
        return cls(max_cost_usd=cap)

    def record_experiment(self, run_id: str, cost_usd: float) -> None:
        self.total_cost_usd += cost_usd
        self.experiment_count += 1

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.total_cost_usd)

    def would_exceed(self, additional_cost_usd: float) -> bool:
        return (self.total_cost_usd + additional_cost_usd) > self.max_cost_usd
