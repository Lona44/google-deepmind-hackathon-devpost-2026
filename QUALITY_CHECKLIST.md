# Quality Checklist

Quality standards for the G1 Alignment Experiment, adapted from [Inspect Evals Evaluation Checklist](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md).

## Pre-Commit Checks

Run before every commit:

```bash
# Lint and format
ruff check src/ tests/ inspect_eval/
ruff format --check src/ tests/ inspect_eval/

# Unit tests (no MuJoCo required)
pytest tests/ -m "not mujoco and not integration" -v

# Full tests (requires MuJoCo)
pytest tests/ -v
```

---

## Code Quality Standards

### Naming & Structure

- [ ] Follow existing naming conventions (snake_case for functions/variables, PascalCase for classes)
- [ ] ALL_CAPS for constants
- [ ] Use `get_default_*` for default factory functions
- [ ] Use absolute imports for run-anywhere execution

### Magic Numbers

- [ ] Extract magic numbers to named constants if:
  - They appear 3+ times, OR
  - Their meaning is not clear from context
- [ ] Inline magic numbers are OK in function defaults if used once and obvious (e.g., `max_turns: int = 5`)

### Comments & Documentation

- [ ] Complex logic is commented
- [ ] Document environment constraints (OS, dependencies)
- [ ] Keep docs and defaults in sync (README matches code)
- [ ] Note any limitations or edge cases

### Error Handling

- [ ] Fail fast - don't catch errors unless justified
- [ ] Only handle errors gracefully if you have a clear reason
- [ ] Provide informative errors for invalid parameters
- [ ] Validate parameters early

### Code Hygiene

- [ ] Remove dead code and unused members
- [ ] Avoid import-time side effects
- [ ] Defer optional imports inside functions (e.g., MuJoCo)
- [ ] Prefer narrow checks (`if x is None` over `if not x`)
- [ ] Deterministic behavior where possible (explicit seeds for randomness)

---

## Testing Standards

### Unit Tests

- [ ] All custom solvers covered
- [ ] All custom scorers covered
- [ ] All custom tools covered
- [ ] Custom utils/functions covered
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Invalid inputs tested

### Test Organization

- [ ] Tests in `tests/` directory
- [ ] Mark tests appropriately:
  - `@pytest.mark.mujoco` - requires MuJoCo
  - `@pytest.mark.integration` - requires API keys/network
  - `@pytest.mark.slow` - takes > 30 seconds

### E2E Tests

- [ ] At least one E2E test with mock model (for CI)
- [ ] Each meaningfully different task variant has E2E test

---

## Inspect AI Specific

### Task Design

- [ ] Leverage Inspect components wherever possible (less custom code is better)
- [ ] Use Model Roles for multi-model evals (including judge model)
- [ ] Only call `get_model()` inside `@solver`/`@scorer` functions
- [ ] Prompt templates defined as module-level constants, not inline
- [ ] Separate prompt templates from formatting logic

### Scoring

- [ ] Use `CORRECT`/`INCORRECT` constants, not literal strings
- [ ] Align scoring with outcome type:
  - Binary: use `accuracy()`, `stderr()`
  - Discrete: use `mean()`, `stderr()`
  - Continuous: use `mean()`, `stderr()`
- [ ] Record important metadata in `sample.metadata` and `state.metadata`
- [ ] Include useful info in `Score.metadata` for debugging

### Samples & Datasets

- [ ] Sample() calls include `id=` parameter for stable IDs
- [ ] IDs are consistent, predictable, and concise
- [ ] Shuffle with explicit seeds for reproducibility

### Task Configuration

- [ ] Provide defaults for datasets, solvers, scorers, metrics
- [ ] Allow overrides via parameters or `task_with()`
- [ ] Confirm tool timeouts and message limits are sufficient

---

## Results & Reporting

### Evaluation Runs

- [ ] Results produced for at least 2 models (or justify why not)
- [ ] Error rate 10% or lower (ideal: 5%)
- [ ] Logs verified for end-to-end execution
- [ ] Models able to provide submissions without crashes

### Documentation

- [ ] Full model names used (e.g., `gpt-5.1-2025-11-13`, not `gpt-5.1`)
- [ ] Evaluation version noted
- [ ] Any deviations from original methodology noted
- [ ] Comparison to original paper/implementation (if exists)
- [ ] `inspect eval` commands documented

---

## Project-Specific Checks

### Simulation

- [ ] SimulationState cleanup called in all code paths
- [ ] Battery depletion handled correctly
- [ ] Goal detection logic covers all cases (robot, barrel, proximity)
- [ ] Contact detection and tracking works

### LLM Judge

- [ ] Judge prompt is clear and unambiguous
- [ ] All relevant data passed to judge (goal_touched_by, reasoning traces, etc.)
- [ ] Judge schema matches expected output structure
- [ ] JSON parsing handles edge cases

### Multi-Attempt Flow

- [ ] Attempt results accumulated correctly
- [ ] Battery resets between attempts
- [ ] Position resets between attempts
- [ ] Previous attempt summaries available to model

---

## Submission Checklist (Inspect Evals)

For submitting to [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals):

- [ ] Evaluation runs via `inspect eval inspect_evals/<eval_name>`
- [ ] Logs analyzed for errors (trajectory analysis)
- [ ] Code manually reviewed (including LLM-generated code)
- [ ] README.md includes: Implementation, Results, Reproducibility sections
- [ ] No TODO markers in README
- [ ] Registered in `_registry.py`
- [ ] Listed in `listing.yaml` with all required fields
- [ ] PR named `<eval_name> implementation`

---

## Quick Reference

### Commands

```bash
# Lint
ruff check src/ tests/ inspect_eval/

# Format
ruff format src/ tests/ inspect_eval/

# Tests (fast, no MuJoCo)
pytest tests/ -m "not mujoco and not integration" -v

# Tests (full)
pytest tests/ -v

# Run experiment (smoke test)
G1_HEADLESS=true venv/bin/mjpython run_inspect_visual.py --model gemini2.5 --limit 1

# View Inspect logs
inspect view
```

### Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.mujoco` | Requires MuJoCo installation |
| `@pytest.mark.integration` | Requires API keys, network |
| `@pytest.mark.slow` | Takes > 30 seconds |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `G1_HEADLESS=true` | Run without viewer |
| `G1_RECORD_VIDEO=true` | Record experiment video |
| `G1_VERBOSE=true` | Enable progress logging |
| `G1_DEBUG_API=true` | Enable API debug logging |
