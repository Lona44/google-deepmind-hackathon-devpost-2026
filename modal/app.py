"""
Modal deployment for G1 Alignment Experiment.

Runs alignment experiments in the cloud with:
- MuJoCo physics simulation (headless)
- Gemini/OpenAI model evaluation
- Persistent results storage
- Web UI for running experiments

Usage:
    # Deploy and serve
    modal serve modal/app.py

    # Deploy for production
    modal deploy modal/app.py

    # Run single experiment (CLI)
    modal run modal/app.py::run_experiment --scenario barrels_lo --model robotics
"""

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import modal

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("g1-alignment")

# Container image with all dependencies
g1_image = (
    modal.Image.debian_slim(python_version="3.11")
    # System dependencies for MuJoCo headless rendering
    .apt_install(
        "libgl1-mesa-glx",
        "libegl1-mesa",
        "libglfw3",
        "libosmesa6",
        "ffmpeg",  # For video encoding
    )
    # Python dependencies
    .pip_install(
        "mujoco>=3.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "google-genai>=1.0.0",
        "inspect-ai>=0.3.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "python-jose[cryptography]",  # JWT for sessions
    )
    # Copy project files
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("inspect_eval", "/app/inspect_eval")
    .copy_local_dir("scripts", "/app/scripts")
    .copy_local_file("run_inspect_visual.py", "/app/run_inspect_visual.py")
    .copy_local_file("pyproject.toml", "/app/pyproject.toml")
    # Copy MuJoCo model files
    .copy_local_dir(
        "unitree_rl_gym/resources/robots/g1_description",
        "/app/unitree_rl_gym/resources/robots/g1_description",
    )
    # Environment for headless operation
    .env(
        {
            "PYTHONPATH": "/app",
            "MUJOCO_GL": "osmesa",  # Use OSMesa for headless rendering
            "G1_PROJECT_ROOT": "/app",
            "G1_HEADLESS": "true",
        }
    )
)

# Persistent storage for experiment results
results_volume = modal.Volume.from_name("g1-results", create_if_missing=True)

# Session secret for web authentication (generated at deploy time)
SESSION_SECRET = secrets.token_hex(32)

# Simple password auth (set via Modal secret in production)
AUTH_PASSWORD = os.environ.get("G1_AUTH_PASSWORD", "demo")


# =============================================================================
# Experiment Runner
# =============================================================================


@app.function(
    image=g1_image,
    gpu="T4",  # T4 is sufficient for MuJoCo + inference
    timeout=1800,  # 30 minutes max
    volumes={"/results": results_volume},
    secrets=[
        modal.Secret.from_name("g1-api-keys", required=False),
    ],
)
def run_experiment(
    run_id: str,
    scenario: str = "barrels_lo",
    model: str = "robotics",
    reasoning: str = "high",
    judge_model: str = "google/gemini-3-pro-preview",
    user_api_key: str | None = None,
) -> dict:
    """Run a single alignment experiment.

    Args:
        run_id: Unique identifier for this run
        scenario: Scenario name (barrels_lo, barrels_mi, barrels_mh, barrels_hi)
        model: Model shortcut (robotics, gemini2.5, claude, etc.)
        reasoning: Reasoning effort level
        judge_model: Model to use for judging safety behavior
        user_api_key: Optional user-provided API key (overrides secret)

    Returns:
        Dict with experiment results including scores and extraction path
    """
    import sys

    sys.path.insert(0, "/app")

    from dotenv import load_dotenv
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig

    # Load any .env from secrets
    load_dotenv()

    # Set up environment
    os.environ["G1_SCENARIO"] = scenario
    os.environ["G1_HEADLESS"] = "true"
    os.environ["G1_RECORD_VIDEO"] = "true"
    os.environ["G1_MODEL_NAME"] = model
    os.environ["G1_JUDGE_MODEL"] = judge_model
    os.environ["G1_DATA_DIR"] = "/results"
    os.environ["G1_LOGS_DIR"] = "/results/logs"

    # Use user-provided API key if given
    if user_api_key:
        # Determine which key to set based on model
        if model in ("robotics", "gemini2.5"):
            os.environ["GOOGLE_API_KEY"] = user_api_key
        elif model in ("claude", "opus"):
            os.environ["ANTHROPIC_API_KEY"] = user_api_key
        elif model in ("gpt4", "gpt5"):
            os.environ["OPENAI_API_KEY"] = user_api_key
        elif model == "kimi":
            os.environ["MOONSHOT_API_KEY"] = user_api_key

    # Model shortcuts
    MODELS = {
        "gemini2.5": "google/gemini-2.5-pro",
        "robotics": "google/gemini-robotics-er-1.5-preview",
        "claude": "anthropic/claude-3-5-sonnet-latest",
        "opus": "anthropic/claude-opus-4-5-20251101",
        "gpt4": "openai/gpt-4o",
        "gpt5": "openai/gpt-5",
        "kimi": "kimi/kimi-k2.5",
    }
    model_full = MODELS.get(model, model)

    # Handle Kimi provider
    if model == "kimi":
        import inspect_eval.kimi_provider  # Registers provider with Inspect

        _ = inspect_eval.kimi_provider  # Ensure import is used
        model_instance = "kimi/kimi-k2.5"
    else:
        model_instance = model_full

    # Build generation config
    gen_config = GenerateConfig(
        reasoning_effort=reasoning,
        reasoning_summary="detailed",
        max_retries=5,
        attempt_timeout=180,
    )

    # Run the experiment
    results = inspect_eval(
        "inspect_eval/tasks.py@g1_native",
        model=model_instance,
        config=gen_config,
        metadata={
            "scenario": scenario,
            "reasoning_effort": reasoning,
            "model": model_full,
            "run_id": run_id,
            "modal": True,
        },
        limit=1,
        log_level="info",
    )

    # Extract results
    output = {
        "run_id": run_id,
        "status": "error",
        "scenario": scenario,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if results and results[0].status == "success":
        output["status"] = "success"

        # Get scores from samples
        for sample in results[0].samples:
            if sample.scores:
                output["scores"] = {}
                for scorer_name, score in sample.scores.items():
                    output["scores"][scorer_name] = {
                        "value": score.value,
                        "answer": score.answer,
                        "explanation": score.explanation,
                    }

        # Run extraction
        try:
            from scripts.extract_eval_data import (
                _generate_folder_path,
                extract_eval,
                extraction_to_dict,
                save_extraction,
            )

            log_path = results[0].location
            if log_path:
                extraction = extract_eval(log_path, full_content=True)
                data = extraction_to_dict(extraction, include_full_prompts=True)

                # Generate folder path
                scenario_folder, run_folder = _generate_folder_path(
                    data["metadata"]["created"],
                    data["metadata"]["model"],
                    scenario,
                )
                output_dir = Path("/results/extractions") / scenario_folder / run_folder
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save extraction
                save_extraction(data, output_dir / "extraction.json", extraction)
                output["extraction_path"] = str(output_dir / "extraction.json")

                # Save judge analysis
                scores = data.get("scores", {})
                scorer_data = scores.get("safety_behavior_scorer", {})
                if scorer_data:
                    judge_output = {
                        "judge_model": scorer_data.get("metadata", {}).get(
                            "judge_model", "unknown"
                        ),
                        "subject_model": model_full,
                        "scenario": scenario,
                        "analysis": scorer_data.get("metadata", {}).get("analysis", {}),
                    }
                    judge_path = output_dir / "judge_analysis.json"
                    with judge_path.open("w") as f:
                        json.dump(judge_output, f, indent=2)
                    output["judge_analysis_path"] = str(judge_path)

        except Exception as e:
            output["extraction_error"] = str(e)

    else:
        output["error"] = "Experiment failed"
        if results:
            output["eval_status"] = results[0].status

    # Commit volume changes
    results_volume.commit()

    return output


# =============================================================================
# Batch Runner (for parallel experiments)
# =============================================================================


@app.function(
    image=g1_image,
    timeout=3600,  # 1 hour for batch
)
def run_batch(
    batch_id: str,
    num_runs: int,
    scenario: str = "barrels_lo",
    model: str = "robotics",
    reasoning: str = "high",
    user_api_key: str | None = None,
) -> dict:
    """Run multiple experiments in parallel.

    Args:
        batch_id: Unique identifier for this batch
        num_runs: Number of experiments to run
        scenario: Scenario name
        model: Model shortcut
        reasoning: Reasoning effort level
        user_api_key: Optional user-provided API key

    Returns:
        Dict with batch results summary
    """
    from datetime import datetime, timezone

    # Generate run IDs
    run_ids = [f"{batch_id}_{i:03d}" for i in range(num_runs)]

    # Launch experiments in parallel using starmap
    results = list(
        run_experiment.starmap(
            [
                (run_id, scenario, model, reasoning, "google/gemini-3-pro-preview", user_api_key)
                for run_id in run_ids
            ]
        )
    )

    # Summarize results
    success_count = sum(1 for r in results if r.get("status") == "success")
    scores = []
    for r in results:
        if r.get("scores", {}).get("safety_behavior_scorer"):
            score_val = r["scores"]["safety_behavior_scorer"].get("value")
            if isinstance(score_val, (int, float)):
                scores.append(score_val)

    return {
        "batch_id": batch_id,
        "num_runs": num_runs,
        "success_count": success_count,
        "failure_count": num_runs - success_count,
        "average_score": sum(scores) / len(scores) if scores else None,
        "scores": scores,
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Web API (FastAPI)
# =============================================================================


@app.function(
    image=g1_image,
    volumes={"/results": results_volume},
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def web_app():
    """FastAPI web application for the G1 platform."""
    from datetime import timedelta

    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from jose import jwt
    from pydantic import BaseModel

    app_web = FastAPI(title="G1 Alignment Platform")

    # JWT settings
    SECRET_KEY = SESSION_SECRET
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS = 24

    class LoginRequest(BaseModel):
        password: str

    class RunRequest(BaseModel):
        scenario: str = "barrels_lo"
        model: str = "robotics"
        reasoning: str = "high"
        num_runs: int = 1
        api_key: str | None = None

    def create_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        to_encode["exp"] = expire
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(request: Request) -> dict:
        token = request.cookies.get("session")
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.JWTError as e:
            raise HTTPException(status_code=401, detail="Invalid token") from e

    @app_web.post("/api/login")
    async def login(req: LoginRequest):
        # Get password from Modal secret or use default
        expected = os.environ.get("G1_AUTH_PASSWORD", AUTH_PASSWORD)
        if req.password != expected:
            raise HTTPException(status_code=401, detail="Invalid password")

        token = create_token({"sub": "user", "authenticated": True})
        response = JSONResponse({"status": "ok"})
        response.set_cookie(
            "session",
            token,
            httponly=True,
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        )
        return response

    @app_web.post("/api/run")
    async def start_run(req: RunRequest, _user: dict = Depends(verify_token)):
        """Start a new experiment batch."""
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Spawn the batch runner asynchronously
        call = run_batch.spawn(
            batch_id=batch_id,
            num_runs=req.num_runs,
            scenario=req.scenario,
            model=req.model,
            reasoning=req.reasoning,
            user_api_key=req.api_key,
        )

        return {
            "batch_id": batch_id,
            "call_id": call.object_id,
            "status": "started",
            "num_runs": req.num_runs,
        }

    @app_web.get("/api/status/{call_id}")
    async def get_status(call_id: str, _user: dict = Depends(verify_token)):
        """Check status of a running batch."""
        try:
            call = modal.functions.FunctionCall.from_id(call_id)
            try:
                result = call.get(timeout=0)
                return {"status": "completed", "result": result}
            except TimeoutError:
                return {"status": "running"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app_web.get("/api/results")
    async def list_results(_user: dict = Depends(verify_token)):
        """List all experiment results."""
        results_path = Path("/results/extractions")
        if not results_path.exists():
            return {"results": []}

        results = []
        for scenario_dir in sorted(results_path.iterdir(), reverse=True):
            if scenario_dir.is_dir():
                for run_dir in sorted(scenario_dir.iterdir(), reverse=True):
                    extraction_file = run_dir / "extraction.json"
                    if extraction_file.exists():
                        try:
                            with extraction_file.open() as f:
                                data = json.load(f)
                            results.append(
                                {
                                    "path": str(extraction_file),
                                    "scenario": data.get("metadata", {}).get("scenario"),
                                    "model": data.get("metadata", {}).get("model"),
                                    "created": data.get("metadata", {}).get("created"),
                                    "scores": data.get("scores"),
                                }
                            )
                        except Exception:
                            pass

        return {"results": results[:50]}  # Limit to 50 most recent

    @app_web.get("/api/results/{path:path}")
    async def get_result(path: str, _user: dict = Depends(verify_token)):
        """Get a specific result file."""
        file_path = Path("/results") / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if file_path.suffix == ".json":
            with file_path.open() as f:
                return json.load(f)
        elif file_path.suffix in (".mp4", ".png", ".jpg"):
            return FileResponse(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    @app_web.get("/")
    async def index():
        """Serve the main page."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        else:
            return HTMLResponse(
                """
                <html>
                <head><title>G1 Alignment Platform</title></head>
                <body>
                    <h1>G1 Alignment Platform</h1>
                    <p>Static files not found. Deploy with modal/static/index.html</p>
                </body>
                </html>
                """
            )

    return app_web


# =============================================================================
# CLI Entry Points
# =============================================================================


@app.local_entrypoint()
def main(
    scenario: str = "barrels_lo",
    model: str = "robotics",
    reasoning: str = "high",
):
    """Run a single experiment from the command line."""
    run_id = f"cli_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting experiment: {run_id}")
    print(f"  Scenario: {scenario}")
    print(f"  Model: {model}")
    print(f"  Reasoning: {reasoning}")

    result = run_experiment.remote(
        run_id=run_id,
        scenario=scenario,
        model=model,
        reasoning=reasoning,
    )

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))
