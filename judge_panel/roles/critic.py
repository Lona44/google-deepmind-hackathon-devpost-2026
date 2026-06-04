"""Counterargument Critic — adversarial second-pass check.

Spec Section "Role definitions". Model: MiMo (xiaomi/mimo-v2.5-pro).
Sits in Layer 2 of the cascade and receives the Auditor's + Detector's
outputs as prior_outputs. Produces worst-case safety/honesty scores
plus a `critic_argument` text arguing the pro-misalignment reading.

Model choice rationale: MiMo is distinct from the Layer-1 Detector
(Kimi K2.6) to maintain model-family diversity across adjacent panel
layers, per the spec's diversity requirement.
"""

from __future__ import annotations

from judge_panel._role_helpers import execute_role
from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.types import RoleOutput

ROLE = "critic"
MODEL = "xiaomi/mimo-v2.5-pro"


async def run(
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    *,
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Execute the Critic role.

    The Critic sits in Layer 2 of the cascade and is the first role to
    consume prior_outputs (Auditor + Detector). Its prompt frames the
    task adversarially: produce the strongest defensible
    pro-misalignment reading rather than a balanced one.
    """
    return await execute_role(
        role=ROLE,
        model=MODEL,
        behavioral_data=behavioral_data,
        prior_outputs=prior_outputs,
        client=client,
        cost_tracker=cost_tracker,
    )
