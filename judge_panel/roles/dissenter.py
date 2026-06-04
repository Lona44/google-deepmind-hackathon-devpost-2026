"""Dissenting Reviewer — Layer 4 constitutional check on the Arbiter's verdict.

Spec Section "Role definitions". Model: MiMo (xiaomi/mimo-v2.5-pro). Sits in
Layer 4 of the cascade and receives ONLY the Arbiter's output as a prior
(the orchestrator also forwards behavioral_data so the Dissenter can verify
the Arbiter's evidence citations, but the prompt directs the Dissenter to
focus on the Arbiter's verdict — not to re-judge the agent).

Model choice rationale: MiMo is deliberately a different model family from
the Arbiter's Kimi K2.6. This is the explicit independence assumption of
the constitutional check — using the same family would make the meta-check
correlated with the verdict it is reviewing, defeating the purpose.

Unlike every other role, the Dissenter does NOT populate `safety_score`,
`honesty_score`, `misalignment_level`, or `evidence`. Its output is a
binary `dissent_flag` plus a `dissent_reason` when the flag is raised. The
RoleOutput.model_validator only enforces non-empty evidence when a
numerical score is set, so an empty evidence list is valid here.
"""

from __future__ import annotations

from judge_panel._role_helpers import execute_role
from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.types import RoleOutput

ROLE = "dissenter"
MODEL = "xiaomi/mimo-v2.5-pro"


async def run(
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    *,
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Execute the Dissenter role.

    Runs the constitutional check on the Arbiter's verdict. `prior_outputs`
    is expected to contain just the Arbiter's RoleOutput; the orchestrator
    is responsible for that contract. Returns a RoleOutput with
    `dissent_flag` set and no numerical scores.
    """
    return await execute_role(
        role=ROLE,
        model=MODEL,
        behavioral_data=behavioral_data,
        prior_outputs=prior_outputs,
        client=client,
        cost_tracker=cost_tracker,
    )
