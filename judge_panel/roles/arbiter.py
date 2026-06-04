"""Holistic Arbiter — Layer 3 synthesizer producing the final verdict.

Spec Section "Role definitions" + Section 3.4 (Critic-failure handling).
Model: Kimi K2.6 (moonshotai/kimi-k2.6). Sits in Layer 3 of the cascade
and receives Auditor + Detector + Critic outputs as prior_outputs.
Produces the final `safety_score`, `honesty_score`, and the role-level
`misalignment_level` classification — Arbiter is the only role that
populates that field.

Model choice rationale: Kimi K2.6 maintains adjacent-layer model-family
diversity. Layer 2 (Critic) is MiMo; using a different family for the
synthesis layer reduces shared failure modes between the dissent step
and the verdict step.
"""

from __future__ import annotations

from judge_panel._role_helpers import execute_role
from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.types import RoleOutput

ROLE = "arbiter"
MODEL = "moonshotai/kimi-k2.6"


async def run(
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    *,
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Execute the Arbiter role.

    Synthesizes the prior judges' analyses (Auditor + Detector + Critic)
    into the final panel verdict, including the misalignment_level
    classification. The prompt handles failed-prior cases explicitly per
    spec Section 3.4.
    """
    return await execute_role(
        role=ROLE,
        model=MODEL,
        behavioral_data=behavioral_data,
        prior_outputs=prior_outputs,
        client=client,
        cost_tracker=cost_tracker,
    )
