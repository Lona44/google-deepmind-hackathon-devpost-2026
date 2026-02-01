# G1 Alignment Experiment - Makefile
# Common commands for local development and Modal deployment

.PHONY: help local modal-serve modal-deploy test lint format clean

# Default target
help:
	@echo "G1 Alignment Experiment"
	@echo ""
	@echo "Local Development:"
	@echo "  make local           Run single experiment locally"
	@echo "  make local-batch     Run batch of 3 experiments locally"
	@echo "  make test            Run unit tests"
	@echo "  make lint            Run linter"
	@echo "  make format          Format code"
	@echo ""
	@echo "Modal Deployment:"
	@echo "  make modal-serve     Start Modal dev server (hot reload)"
	@echo "  make modal-deploy    Deploy to Modal (production)"
	@echo "  make modal-run       Run single experiment on Modal"
	@echo ""
	@echo "Variables:"
	@echo "  SCENARIO=barrels_lo  Scenario (barrels_lo, barrels_mi, barrels_mh, barrels_hi)"
	@echo "  MODEL=robotics       Model (robotics, gemini2.5, claude, opus, gpt5)"
	@echo "  NUM_RUNS=1           Number of runs for batch"

# =============================================================================
# Local Development
# =============================================================================

SCENARIO ?= barrels_lo
MODEL ?= robotics
NUM_RUNS ?= 3

# Run single experiment locally
local:
	./venv/bin/mjpython local_runner.py --scenario $(SCENARIO) --model $(MODEL)

# Run batch locally
local-batch:
	./venv/bin/mjpython local_runner.py --scenario $(SCENARIO) --model $(MODEL) --num-runs $(NUM_RUNS)

# Run with existing runner (for comparison)
run:
	./venv/bin/mjpython run_inspect_visual.py --scenario $(SCENARIO) --model $(MODEL) --headless

# Run unit tests (skip MuJoCo and integration)
test:
	./venv/bin/pytest tests/ -v --tb=short -m "not integration and not mujoco"

# Run all tests including MuJoCo (requires MuJoCo installed)
test-all:
	./venv/bin/pytest tests/ -v --tb=short -m "not integration"

# Lint code
lint:
	./venv/bin/ruff check src/ tests/ inspect_eval/ modal/

# Format code
format:
	./venv/bin/ruff format src/ tests/ inspect_eval/ modal/

# Check formatting without changes
format-check:
	./venv/bin/ruff format --check src/ tests/ inspect_eval/ modal/

# =============================================================================
# Modal Deployment
# =============================================================================

# Start Modal dev server (hot reload)
modal-serve:
	modal serve modal/app.py

# Deploy to Modal (production)
modal-deploy:
	modal deploy modal/app.py

# Run single experiment on Modal
modal-run:
	modal run modal/app.py::run_experiment --scenario $(SCENARIO) --model $(MODEL)

# Create Modal secrets (run once)
modal-setup:
	@echo "Create Modal secrets with:"
	@echo "  modal secret create g1-api-keys \\"
	@echo "    GOOGLE_API_KEY=your_key \\"
	@echo "    ANTHROPIC_API_KEY=your_key \\"
	@echo "    OPENAI_API_KEY=your_key \\"
	@echo "    G1_AUTH_PASSWORD=your_password"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Pre-commit Validation
# =============================================================================

validate: lint format-check test
	@echo "All checks passed!"
