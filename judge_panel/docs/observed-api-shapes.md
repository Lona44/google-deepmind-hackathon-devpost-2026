# Observed OpenRouter response shapes for MiMo V2.5 Pro + Kimi K2.6

Recorded: 2026-06-04
Recordings: `scripts/exploration-recordings/`
- `xiaomi-mimo-v2.5-pro-20260604T005250.json`
- `moonshotai-kimi-k2.6-20260604T005250.json`
- `_summary-20260604T005250.json`

Single-call exploration with a representative arithmetic-safety-judgement prompt
(`temperature=0`, `max_tokens=2000`). Both calls returned HTTP 200 with
`finish_reason="stop"`. Total live cost: **$0.00474** (MiMo $0.000824, Kimi $0.003912).

## Question 1: Does the model wrap JSON in ` ```json ``` ` fences?

**MiMo V2.5 Pro:** No. Raw `choices[0].message.content` begins with `{` directly,
no markdown fences and no leading prose.

**Kimi K2.6:** No, but the content has **one leading whitespace character**
before the opening `{` (e.g. `" {\n  \"safety_score\": 5,..."`). No markdown fences.

**Decision:** Role parsers must `strip()` the content before `json.loads()`.
Also strip leading/trailing ``` ```json ``` ``` fences as a defence in depth,
because fence behaviour appears non-deterministic across OpenRouter providers
and may change between routes/versions. Both observed responses parse cleanly
as JSON after `.strip()` followed by fence-strip.

## Question 2: Are reasoning traces returned separately or inline?

**MiMo V2.5 Pro:** Reasoning is exposed in **two separate fields** on
`choices[0].message`:
- `reasoning` — string, the full chain-of-thought as plain text
- `reasoning_details` — list of structured reasoning blocks
  (each `{"type": "reasoning.text", "text": "..."}`)

The `content` field contains ONLY the final JSON verdict. No `<thinking>` tags
appear inline in `content`.

**Kimi K2.6:** Same shape — `reasoning` and `reasoning_details` are present as
sibling fields on `choices[0].message`. `content` contains only the JSON verdict.

**Both** also expose `completion_tokens_details.reasoning_tokens`
(MiMo: 837, Kimi: 1317) so reasoning-token cost is observable separately from
output tokens.

**Decision:** Parsers read `choices[0].message.content` for the JSON verdict.
The `reasoning` field (and `reasoning_details`) is captured into `raw_response`
for audit but is **not** parsed as part of the verdict. The pydantic
`OpenRouterMessage` type must declare `reasoning: str | None = None` and
`reasoning_details: list[dict] | None = None` as optional, since other
OpenRouter-routed providers may omit them.

## Question 3: Does OpenRouter report cached input tokens?

**Observed `usage` keys for MiMo:**
`prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`, `is_byok`,
`prompt_tokens_details`, `cost_details`, `completion_tokens_details`.

`prompt_tokens_details` contents:
```json
{"cached_tokens": 384, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}
```
(384 of 387 prompt tokens hit upstream cache on first call — Xiaomi appears
to apply automatic prefix caching.)

**Observed `usage` keys for Kimi K2.6:** Identical set as above.
`prompt_tokens_details`:
```json
{"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}
```
(No cache hit on the cold first call; cache_write also 0, suggesting
Moonshot/Kimi either doesn't cache prefixes or doesn't report writes via this
field.)

**Decision:** Read `usage.prompt_tokens_details.cached_tokens` for cached input
counting and `usage.prompt_tokens_details.cache_write_tokens` for cache writes.
Both fields are guaranteed present for these two providers; treat them as
optional with default `0` in pydantic so other future providers don't break.
Persist `usage.cost` (per-call USD reported by OpenRouter) into RoleOutput
metrics so the calibration suite can sum provider-attributed cost directly
without recomputing from token counts.

## Question 4: Do both models honor an explicit JSON schema in the prompt?

**MiMo V2.5 Pro:** Yes. Returned exactly the three required keys
(`safety_score`, `reasoning`, `quote`), correct types (int, string, string),
no extra keys. The quoted text is verbatim from the input.

**Kimi K2.6:** Yes. Same — three required keys, correct types, verbatim quote.
Only quirk: one leading space before `{` (see Question 1).

**Decision:** Prompt-based schema is sufficient for both target models. Do NOT
use `response_format={"type":"json_object"}` since OpenRouter routing means
different upstream providers may not support it (and the prompt-based path
worked on first attempt for both). The validation layer in
`judge_panel.parsers` will still defensively `pydantic.ValidationError`-trap
on parse so a future model regression surfaces a clean retry path rather than
a stack trace.

## Question 5: What's the round-trip latency at our typical prompt size?

**MiMo V2.5 Pro:** **22,222 ms** (~22.2 s) for 387-token prompt → 944 completion
tokens (of which 837 were internal reasoning tokens).

**Kimi K2.6:** **17,478 ms** (~17.5 s) for 143-token prompt → 1,091 completion
tokens (of which 1,317 reasoning tokens — note reasoning_tokens > completion
indicates reasoning is billed separately and not subtracted from completion).

Both responses include a substantial reasoning-trace generation phase, which
dominates wall-clock time. Real Layer 2/3/4 prompts (with longer behavioral
context) will be slower; budget assumes 20–40 s per role call.

**Decision:** Layer 1 runs Auditor + Detector in parallel via `asyncio.gather`.
Other layers run sequentially. Total budget per panel run: ~30 s wall-clock
per role, ~2 min total for 4 sequential roles. Set per-call `httpx` timeout to
**90 s** to give 2-3x headroom over observed p50, and configure the role
orchestrator to retry once with exponential backoff on `httpx.ReadTimeout`.
