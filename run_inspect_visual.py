"""Run Inspect eval with MuJoCo viewer via mjpython."""

from inspect_ai import eval
from inspect_ai.model import GenerateConfig

# Configuration
MODEL = "google/gemini-2.5-pro"
REASONING_EFFORT = "high"  # options: none, minimal, low, medium, high, xhigh

# Run the native task with visual and reasoning
results = eval(
    "inspect_eval/tasks.py@g1_native",
    model=MODEL,
    config=GenerateConfig(reasoning_effort=REASONING_EFFORT),
    metadata={
        "reasoning_effort": REASONING_EFFORT,
        "model": MODEL,
    },
    limit=1,
)

print("\n=== Eval Complete ===")
for result in results:
    print(f"Status: {result.status}")
    for sample in result.samples:
        print(f"Score: {sample.scores}")
