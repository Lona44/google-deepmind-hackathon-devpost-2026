"""
Integration tests for G1 Alignment Experiment.

These tests require:
- MuJoCo installation
- GEMINI_API_KEY environment variable
- Significant time to run (uses real API calls)

Run manually with: pytest tests/integration/ -v -m integration
Or via CI with: gh workflow run ci.yml -f run_integration=true
"""
