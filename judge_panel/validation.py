"""Semantic validation of RoleOutput against the source behavioral_data.

The load-bearing rule: every Evidence.quote MUST be a verbatim substring of
the actual step text. Hallucinated citations are killed at parse time so
they never reach the published methodology.

Strip semantics (per spec): leading/trailing whitespace on the quote IS
stripped. Internal whitespace is NOT normalised — a quote that collapses
two spaces into one is still a hallucination.
"""

from __future__ import annotations

import json

from judge_panel.types import Evidence, RoleOutput


class SemanticValidationError(ValueError):
    """Raised when RoleOutput content fails a semantic check (not a schema check)."""


def _step_searchable_text(step: dict) -> str:
    """Concatenate everything in a step that a quote could plausibly cite.

    Includes reasoning, tool_calls (serialised), and result.
    """
    parts: list[str] = []
    if "reasoning" in step:
        parts.append(str(step["reasoning"]))
    if "tool_calls" in step:
        # Serialise tool_calls as JSON so that any nested arg string is searchable.
        parts.append(json.dumps(step["tool_calls"]))
    if "result" in step:
        parts.append(str(step["result"]))
    return "\n".join(parts)


def _validate_one_evidence(ev: Evidence, behavioral_data: dict) -> None:
    steps = behavioral_data.get("steps", [])
    if ev.step_id >= len(steps):
        raise SemanticValidationError(
            f"step_id {ev.step_id} out of bounds (have {len(steps)} steps)"
        )
    step = steps[ev.step_id]
    haystack = _step_searchable_text(step)
    needle = ev.quote.strip()
    if needle not in haystack:
        raise SemanticValidationError(
            f"quote not found verbatim in step {ev.step_id}: {ev.quote!r}"
        )


def validate_role_output(role_output: RoleOutput, behavioral_data: dict) -> None:
    """Run all semantic checks on a RoleOutput. Raises on any failure.

    Schema-level checks (score ranges, mandatory evidence, etc.) are already
    enforced by the pydantic model. This function adds the data-dependent
    checks that pydantic can't make in isolation.
    """
    for ev in role_output.evidence:
        _validate_one_evidence(ev, behavioral_data)
