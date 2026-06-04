"""Vulnerability Detector — implicit failure-mode check.

Spec Section "Role definitions". Model: Kimi K2.6. Sits in Layer 1 of the
cascade alongside the Auditor; produces an independent honesty estimate
plus a risk flag for implicit failure modes the literal rubric misses.
"""

from __future__ import annotations

from judge_panel._role_helpers import execute_role
from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.types import RoleOutput

ROLE = "detector"
MODEL = "moonshotai/kimi-k2.6"


async def run(
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    *,
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Execute the Detector role.

    The Detector sits in Layer 1 of the cascade; it receives no prior
    outputs. The `prior_outputs` parameter is part of the uniform role
    signature for orchestrator simplicity.
    """
    return await execute_role(
        role=ROLE,
        model=MODEL,
        behavioral_data=behavioral_data,
        prior_outputs=prior_outputs,
        client=client,
        cost_tracker=cost_tracker,
    )
