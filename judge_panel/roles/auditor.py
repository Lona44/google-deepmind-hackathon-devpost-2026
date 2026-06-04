"""Security Criterion Auditor — literal rubric compliance check.

Spec Section "Role definitions". Model: MiMo V2.5 Pro.
"""

from __future__ import annotations

from judge_panel._role_helpers import execute_role
from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.types import RoleOutput

ROLE = "auditor"
MODEL = "xiaomi/mimo-v2.5-pro"


async def run(
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    *,
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Execute the Auditor role.

    The Auditor sits in Layer 1 of the cascade; it receives no prior outputs.
    The `prior_outputs` parameter is part of the uniform role signature for
    orchestrator simplicity.
    """
    return await execute_role(
        role=ROLE,
        model=MODEL,
        behavioral_data=behavioral_data,
        prior_outputs=prior_outputs,
        client=client,
        cost_tracker=cost_tracker,
    )
