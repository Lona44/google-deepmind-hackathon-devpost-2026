"""Pydantic types for the multi-judge panel.

This module is the typed contract surface. Every other module in judge_panel/
imports from here. Spec: docs/superpowers/specs/2026-06-04-judge-panel-design.md
Section 2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Evidence(BaseModel):
    """A verbatim citation from behavioral_data supporting a role's claim.

    The verbatim-quote check (substring match against the actual step text)
    is performed by judge_panel.validation.validate_evidence_quotes() — NOT
    in this schema. This class only enforces structural shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: int = Field(ge=0, description="0-indexed step in behavioral_data.steps")
    quote: str = Field(min_length=1, description="Verbatim text from the trace")
    interpretation: str = Field(min_length=1, description="What this evidence shows")

    @field_validator("quote", "interpretation")
    @classmethod
    def _not_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace content")
        return value
