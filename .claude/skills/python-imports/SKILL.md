---
name: python-imports
description: Python import best practices for avoiding CI failures. Use when writing imports, dealing with optional dependencies, or fixing import-time errors.
---

# Python Import Best Practices

Avoid import-time failures that break CI and tests.

## Core Principle

**Don't run code at import time** - especially code that:
- Requires environment variables
- Connects to external services
- Imports optional dependencies

## Lazy Initialization Pattern

### Bad: Eager validation
```python
# This fails at import time if GEMINI_API_KEY missing
import os
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found")  # Crashes on import!
```

### Good: Lazy validation
```python
import os

_api_key = os.getenv("GEMINI_API_KEY")

def _get_api_key() -> str:
    """Get API key, raising if not configured."""
    if not _api_key:
        raise ValueError("GEMINI_API_KEY not found")
    return _api_key

class Client:
    def __init__(self):
        self.key = _get_api_key()  # Only fails when actually used
```

## Optional Dependencies

### Bad: Direct import
```python
# Fails if mujoco not installed
import mujoco
from mujoco import viewer
```

### Good: Guarded import
```python
try:
    import mujoco
    from mujoco import viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco = None
    viewer = None

def run_simulation():
    if not HAS_MUJOCO:
        raise RuntimeError("mujoco required: pip install mujoco")
    # ... use mujoco
```

### Good: pytest skip
```python
import pytest

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not available")

@pytest.mark.integration
def test_simulation():
    # Only runs if mujoco available
    pass
```

## Package __init__.py

### Bad: Import everything
```python
# src/__init__.py
from .gemini_client import GeminiNavigator  # Requires google-genai
from .robot import RobotController           # Requires mujoco
from .simulation import SimulationRunner     # Requires both
```

### Good: Minimal exports
```python
# src/__init__.py
# Only export what has no heavy dependencies
from .config import (
    PROJECT_ROOT,
    ScenarioConfig,
    load_scenario,
)

__all__ = ["PROJECT_ROOT", "ScenarioConfig", "load_scenario"]
```

## This Project's Patterns

We use these patterns:
- `src/__init__.py` - Only exports config (no external deps)
- `src/gemini_client.py` - Lazy API key validation via `_get_api_key()`
- `tests/test_g1_sensors.py` - Uses `pytest.importorskip("mujoco")`
- Integration tests marked with `@pytest.mark.integration`

## CI Implications

| Pattern | CI Behavior |
|---------|-------------|
| Eager import | Test collection fails |
| Lazy import | Tests skip gracefully |
| `importorskip` | Test skipped with message |
| `@pytest.mark.integration` | Excluded via `-m "not integration"` |
