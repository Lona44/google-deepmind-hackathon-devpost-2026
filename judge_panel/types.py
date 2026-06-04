"""Pydantic types for the multi-judge panel.

This module is the typed contract surface. Every other module in judge_panel/
imports from here. Spec: docs/superpowers/specs/2026-06-04-judge-panel-design.md
Section 2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


RoleName = Literal["auditor", "detector", "critic", "arbiter", "dissenter"]
MisalignmentLevel = Literal["L0", "L1", "L2", "L3", "L4"]


class ErrorDetail(BaseModel):
    """Structured error attached to a RoleOutput when a role call fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1, description="Short error class identifier")
    message: str = Field(min_length=1)


class RoleOutput(BaseModel):
    """One role's contribution to a panel verdict.

    See spec Section 2.2 for the table of which roles populate which fields.
    Schema-level rule enforced here: if a numerical score is given, at least
    one Evidence citation is required. Verbatim-quote semantic validation
    is performed separately by judge_panel.validation.
    """

    model_config = ConfigDict(extra="forbid")

    role: RoleName
    model: str = Field(min_length=1)
    prompt_sha: str = Field(min_length=1)

    safety_score: int | None = Field(default=None, ge=0, le=5)
    honesty_score: int | None = Field(default=None, ge=1, le=5)
    misalignment_level: MisalignmentLevel | None = None

    reasoning: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    detector_risk_flag: Literal["low", "medium", "high"] | None = None
    critic_argument: str | None = None
    dissent_flag: bool | None = None
    dissent_reason: str | None = None

    duration_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(ge=0.0)

    raw_response: dict
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def _evidence_required_when_score_given(self) -> "RoleOutput":
        has_numerical = self.safety_score is not None or self.honesty_score is not None
        if has_numerical and not self.evidence:
            raise ValueError(
                "evidence must be non-empty when safety_score or honesty_score is set"
            )
        return self
