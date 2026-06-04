# Calibration validation (Tier C — live API)

This directory contains the live-API calibration validator that compares
the multi-judge panel's verdicts against the existing Gemini 3 Pro
baselines for the same recorded G1 experiments.

## What this produces

Each run writes to `results/<YYYY-MM-DD>/`:

- `verdict-<run_id>.json` — full panel verdict for each input run
- `comparison-<run_id>.json` — CalibrationResult (panel vs Gemini)
- `REPORT.md` — aggregate inter-rater agreement metrics + per-run table

`REPORT.md` is the citable artefact for the AISI research note (APP-48).

## How to run

```bash
cd /Users/m44/Desktop/Projects/G1-Alignment/embodied-ai-alignment

# Confirm the OpenRouter key is loaded and has budget
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; assert os.getenv('OPENROUTER_API_KEY'), 'set OPENROUTER_API_KEY in .env'"

# Run calibration (live API — expect ~$0.15-0.20)
python3 tests/judge_panel/calibration/run_calibration.py
```

## Cost

Expect ~$0.05 per behavioral_data input. The default input set is the
2 recorded G1 experiments + 1 synthetic edge case = 3 runs = ~$0.15. The
per-session cost cap defaults to $5 (env-overridable).

## Reproducibility

- Temperature 0 for all role calls
- prompt_sha recorded in every RoleOutput
- Results never overwritten — each `results/<date>/` directory is its own
  citable artefact

## Acknowledged limits

See spec Section 5.7. The calibration is an agreement metric, not a truth
metric — if the published Gemini baseline was wrong, the panel agreeing
with it doesn't mean the panel is correct.
