---
name: pre-commit-validation
description: Validate code before committing - runs linting, formatting, and tests. Use when preparing to commit, before git add, or when checking if code is ready to push.
---

# Pre-Commit Validation

Run these checks before every commit to prevent CI failures.

## Quick Validation

```bash
# Lint check
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Run tests (skip integration tests that need mujoco)
pytest tests/ -m "not integration" -v

# Secret scan
detect-secrets scan --baseline .secrets.baseline
```

## Fix Issues

```bash
# Auto-fix lint issues
ruff check src/ tests/ --fix

# Auto-format code
ruff format src/ tests/
```

## Common Issues This Prevents

1. **Lint failures** - Unused imports, wrong path methods (use pathlib not os.path)
2. **Format failures** - Inconsistent spacing, line lengths
3. **Test failures** - Broken imports, missing dependencies
4. **Secret leaks** - Accidentally committed API keys

## Pre-Commit Checklist

Before running `git commit`:
- [ ] `ruff check src/ tests/` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `pytest tests/ -m "not integration"` passes
- [ ] No `.env` files staged (only `.env.example` allowed)
- [ ] No hardcoded API keys or secrets

## CI Environment Differences

Local and CI environments may differ:
- CI uses Python 3.11
- CI installs package via `pip install -e ".[dev]"`
- Tool versions may differ (e.g., detect-secrets)

Always test with the same commands CI uses.
