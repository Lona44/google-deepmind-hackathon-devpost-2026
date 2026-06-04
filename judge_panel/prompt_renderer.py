"""Prompt rendering with explicit cache-prefix / fresh-suffix split.

Spec Section 3.2: stable content (system prompt, rubric, schema, few-shot)
goes in the cached_prefix region. Run-specific content (behavioral_data,
prior role outputs) goes in the fresh_suffix region. OpenRouter's prompt
cache hits on the prefix; the suffix is always billed at full input rate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from judge_panel.types import RoleName, RoleOutput

PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"

CACHE_BREAKPOINT = (
    "\n\n---\n"
    "## Behavioral data for this run\n"
    "---\n"
)


@dataclass(frozen=True)
class PromptParts:
    """Two regions of a fully-rendered prompt."""

    cached_prefix: str  # stable across all runs of this role
    fresh_suffix: str   # run-specific

    @property
    def full(self) -> str:
        return self.cached_prefix + self.fresh_suffix


def load_prompt(role: str) -> str:
    """Load the markdown prompt for a role. Raises FileNotFoundError if absent."""
    path = PROMPT_ROOT / f"{role}.md"
    return path.read_text()


def compute_prompt_sha(role: str) -> str:
    """SHA-1 of the role's prompt markdown file. Used in RoleOutput.prompt_sha."""
    text = load_prompt(role)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _serialise_prior(prior: list[RoleOutput]) -> str:
    if not prior:
        return ""
    blocks = []
    for p in prior:
        # Compact JSON to keep tokens down; pydantic .model_dump_json() output is canonical.
        blocks.append(f"### Prior {p.role} output\n```json\n{p.model_dump_json(indent=2)}\n```")
    return "\n\n## Prior role outputs\n\n" + "\n\n".join(blocks)


def render_prompt(
    role: RoleName,
    *,
    behavioral_data: dict,
    prior_outputs: list[RoleOutput],
) -> PromptParts:
    """Render a role prompt split into cached + fresh regions.

    cached_prefix: role definition + rubric + schema + few-shot (from the
        markdown file, unchanged across runs).
    fresh_suffix: the explicit CACHE_BREAKPOINT marker + behavioral_data +
        prior role outputs serialised as JSON.
    """
    prompt_text = load_prompt(role)
    cached_prefix = prompt_text + CACHE_BREAKPOINT

    behavioral_block = json.dumps(behavioral_data, indent=2)
    prior_block = _serialise_prior(prior_outputs)
    fresh_suffix = behavioral_block + prior_block

    return PromptParts(cached_prefix=cached_prefix, fresh_suffix=fresh_suffix)
