"""Shared helpers used by every role module.

Keeps roles/<role>.py files minimal — each role only owns its model choice,
prompt name, and per-role field assignments (safety_score, etc.). Parsing,
validation, cost recording, error wrapping are here.
"""

from __future__ import annotations

import json
import re

from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterCallResult, OpenRouterClient
from judge_panel.prompt_renderer import compute_prompt_sha, render_prompt
from judge_panel.retry import ValidationRetryError, with_retries
from judge_panel.types import ErrorDetail, Evidence, RoleName, RoleOutput
from judge_panel.validation import SemanticValidationError, validate_role_output


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def strip_markdown_fence(content: str) -> str:
    """If `content` is wrapped in a ```json``` fence, return the inner JSON.
    Otherwise return content unchanged."""
    match = _FENCE_RE.search(content)
    if match:
        return match.group(1)
    return content.strip()


def parse_role_payload(content: str) -> dict:
    """Parse the model's response into a dict. Raises ValidationRetryError on
    invalid JSON so the retry helper can re-issue the call with a correction."""
    try:
        return json.loads(strip_markdown_fence(content))
    except json.JSONDecodeError as exc:
        raise ValidationRetryError(f"model output was not valid JSON: {exc}") from exc


def build_role_output(
    *,
    role: RoleName,
    model: str,
    prompt_sha: str,
    parsed_payload: dict,
    api_result: OpenRouterCallResult,
) -> RoleOutput:
    """Convert the model's parsed JSON payload plus the API call telemetry
    into a typed RoleOutput. Raises pydantic ValidationError on bad shape."""
    return RoleOutput(
        role=role,
        model=model,
        prompt_sha=prompt_sha,
        safety_score=parsed_payload.get("safety_score"),
        honesty_score=parsed_payload.get("honesty_score"),
        misalignment_level=parsed_payload.get("misalignment_level"),
        reasoning=parsed_payload.get("reasoning", ""),
        evidence=[Evidence(**e) for e in parsed_payload.get("evidence", [])],
        confidence=parsed_payload.get("confidence", 0.0),
        detector_risk_flag=parsed_payload.get("detector_risk_flag"),
        critic_argument=parsed_payload.get("critic_argument"),
        dissent_flag=parsed_payload.get("dissent_flag"),
        dissent_reason=parsed_payload.get("dissent_reason"),
        duration_ms=api_result.duration_ms,
        input_tokens=api_result.input_tokens,
        cached_input_tokens=api_result.cached_input_tokens,
        output_tokens=api_result.output_tokens,
        cost_usd=api_result.cost_usd,
        raw_response=api_result.raw_response,
    )


def make_failed_role_output(
    *,
    role: RoleName,
    model: str,
    prompt_sha: str,
    error_kind: str,
    error_message: str,
    api_result: OpenRouterCallResult | None = None,
) -> RoleOutput:
    """Build a RoleOutput representing a failed call.

    Score fields are None; reasoning explains the failure; error is set.
    """
    return RoleOutput(
        role=role,
        model=model,
        prompt_sha=prompt_sha,
        reasoning=f"role failed: {error_message}",
        evidence=[],
        confidence=0.0,
        duration_ms=api_result.duration_ms if api_result else 0,
        input_tokens=api_result.input_tokens if api_result else 0,
        cached_input_tokens=api_result.cached_input_tokens if api_result else 0,
        output_tokens=api_result.output_tokens if api_result else 0,
        cost_usd=api_result.cost_usd if api_result else 0.0,
        raw_response=api_result.raw_response if api_result else {},
        error=ErrorDetail(kind=error_kind, message=error_message),
    )


async def execute_role(
    *,
    role: RoleName,
    model: str,
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
    client: OpenRouterClient,
    cost_tracker: PerExperimentTracker,
) -> RoleOutput:
    """Generic role execution.

    Renders the prompt, makes the OpenRouter call (with retry), parses the
    payload, runs semantic validation, records cost. Returns either a
    successful RoleOutput or a failed one with error set.
    """
    prompt_sha = compute_prompt_sha(role)
    parts = render_prompt(role, behavioral_data=behavioral_data, prior_outputs=prior_outputs)
    last_api_result: OpenRouterCallResult | None = None

    async def attempt() -> RoleOutput:
        nonlocal last_api_result
        api_result = await client.call(
            model=model, cached_prefix=parts.cached_prefix, fresh_suffix=parts.fresh_suffix
        )
        last_api_result = api_result
        cost_tracker.charge(role, api_result.cost_usd)
        try:
            payload = parse_role_payload(api_result.content)
            role_output = build_role_output(
                role=role, model=model, prompt_sha=prompt_sha,
                parsed_payload=payload, api_result=api_result,
            )
        except ValidationRetryError:
            raise
        except Exception as exc:  # pydantic ValidationError or other shape issues
            raise ValidationRetryError(f"could not build RoleOutput: {exc}") from exc

        # Semantic validation — verbatim quote check etc.
        try:
            validate_role_output(role_output, behavioral_data)
        except SemanticValidationError as exc:
            raise ValidationRetryError(f"semantic check failed: {exc}") from exc

        return role_output

    try:
        return await with_retries(attempt, policy="validation", backoff_base=0.0)
    except ValidationRetryError as exc:
        return make_failed_role_output(
            role=role, model=model, prompt_sha=prompt_sha,
            error_kind="validation_failed",
            error_message=str(exc),
            api_result=last_api_result,
        )
