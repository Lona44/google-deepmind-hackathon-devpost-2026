---
name: test-ci-locally
description: Test CI/CD scripts and workflows locally before pushing. Use when modifying GitHub Actions, CI configuration, or before pushing changes that might fail CI.
---

# Test CI Locally

Always test CI scripts locally before pushing to avoid failed builds.

## GitHub Actions Workflow

Our CI workflow (`.github/workflows/ci.yml`) runs these jobs:

### 1. Lint & Format
```bash
pip install ruff
ruff check src/ tests/
ruff format --check src/ tests/
```

### 2. Test
```bash
pip install -e ".[dev]"
pytest tests/ -v --tb=short -m "not integration"
pytest tests/ --cov=src --cov-report=term-missing -m "not integration"
```

### 3. Type Check
```bash
pip install mypy
mypy src/ --ignore-missing-imports
```

### 4. Security Scan
```bash
pip install bandit detect-secrets

# Bandit security scan
bandit -r src/ -ll -ii

# Secret scanning
detect-secrets scan --baseline .secrets.baseline
jq -r '.results | keys | length' .secrets.baseline  # Should be 0

# Check no .env files committed
git ls-files | grep -E '^\.env$|^\.env\.' | grep -v '\.example$'
```

## Testing Locally

Create a temporary venv to match CI:
```bash
python3 -m venv /tmp/ci-test
source /tmp/ci-test/bin/activate
pip install -e ".[dev]"
pip install ruff mypy bandit detect-secrets

# Run all checks
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -m "not integration"
mypy src/ --ignore-missing-imports
bandit -r src/ -ll -ii
detect-secrets scan --baseline .secrets.baseline
```

## Common CI Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Missing package config | Add to `pyproject.toml` |
| Import-time errors | Eager initialization | Use lazy loading |
| Tool version mismatch | Local vs CI versions differ | Pin versions or use CI's format |
| Secret scan fails | Metadata changes in baseline | Only check `results` field |

## Environment Differences

| Aspect | Local (macOS) | CI (Ubuntu) |
|--------|---------------|-------------|
| Python | varies | 3.11 |
| Shell | zsh | bash |
| detect-secrets | varies | latest |
| mujoco | installed | not installed |

## Before Pushing

1. Run the full CI simulation locally
2. Check for environment-specific code
3. Ensure tests skip gracefully when dependencies missing
4. Verify `.secrets.baseline` won't cause false positives
