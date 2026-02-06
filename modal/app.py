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
        "httpx>=0.25.0",  # For OAuth token exchange
    )
    # Environment for headless operation (must be before add_local_*)
    .env(
        {
            "PYTHONPATH": "/app",
            "MUJOCO_GL": "osmesa",  # Use OSMesa for headless rendering
            "G1_PROJECT_ROOT": "/app",
            "G1_HEADLESS": "true",
        }
    )
    # Copy project files (must be last in image build chain)
    .add_local_dir("src", "/app/src")
    .add_local_dir("inspect_eval", "/app/inspect_eval")
    .add_local_dir("scripts", "/app/scripts")
    .add_local_dir("modal/static", "/app/modal/static")
    .add_local_file("run_inspect_visual.py", "/app/run_inspect_visual.py")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
    # Copy MuJoCo model files
    .add_local_dir(
        "unitree_rl_gym/resources/robots/g1_description",
        "/app/unitree_rl_gym/resources/robots/g1_description",
    )
)

# Persistent storage for experiment results
results_volume = modal.Volume.from_name("g1-results", create_if_missing=True)

# Session secret for web authentication (generated at deploy time)
SESSION_SECRET = secrets.token_hex(32)

# Simple password auth (must be set via Modal secret - no default for security)
AUTH_PASSWORD = os.environ.get("G1_AUTH_PASSWORD")

# Embedded HTML for web UI (avoids Modal mount path issues)
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G1 Alignment Platform</title>
    <style>
        :root { --bg: #0a0a0a; --surface: #1a1a1a; --border: #333; --text: #e0e0e0; --text-dim: #888; --accent: #4a9eff; --success: #4caf50; --error: #f44336; --warning: #ff9800; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
        .header-title h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
        .header-title .subtitle { color: var(--text-dim); margin: 0; }
        .header-actions { display: flex; align-items: center; gap: 1rem; }
        .user-info { font-size: 0.875rem; color: var(--text-dim); }
        .btn-logout { background: transparent; border: 1px solid var(--border); color: var(--text-dim); padding: 0.5rem 1rem; font-size: 0.875rem; }
        .btn-logout:hover { border-color: var(--error); color: var(--error); }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
        .card h2 { font-size: 1rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; color: var(--text-dim); font-size: 0.875rem; margin-bottom: 0.5rem; }
        input, select { width: 100%; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 1rem; }
        input:focus, select:focus { outline: none; border-color: var(--accent); }
        button { padding: 0.75rem 1.5rem; background: var(--accent); color: white; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }
        .status { padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.875rem; }
        .status.running { background: rgba(74, 158, 255, 0.2); color: var(--accent); }
        .status.success { background: rgba(76, 175, 80, 0.2); color: var(--success); }
        .status.error { background: rgba(244, 67, 54, 0.2); color: var(--error); }
        .divider { display: flex; align-items: center; margin: 1.5rem 0; color: var(--text-dim); font-size: 0.875rem; }
        .divider::before, .divider::after { content: ''; flex: 1; border-bottom: 1px solid var(--border); }
        .divider span { padding: 0 1rem; }
        .btn-google { background: #4285f4; display: flex; align-items: center; justify-content: center; gap: 0.5rem; width: 100%; padding: 0.75rem; border-radius: 4px; text-decoration: none; color: white; font-size: 1rem; }
        .btn-google:hover { opacity: 0.9; }
        .btn-google svg { width: 18px; height: 18px; }
        .results-list { list-style: none; }
        .results-list li { padding: 1rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        .results-list li:last-child { border-bottom: none; }
        .result-info { flex: 1; }
        .result-model { font-weight: 500; }
        .result-meta { font-size: 0.875rem; color: var(--text-dim); }
        .result-score { font-size: 1.25rem; font-weight: 600; min-width: 60px; text-align: right; }
        .hidden { display: none !important; }
        .progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 1rem; }
        .progress-bar-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
        .security-badge { display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: var(--success); margin-top: 1rem; }
        .security-badge svg { width: 14px; height: 14px; }
        .warning { background: rgba(255, 152, 0, 0.1); border: 1px solid var(--warning); color: var(--warning); padding: 0.75rem; border-radius: 4px; font-size: 0.875rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <h1>G1 Alignment Platform</h1>
                <p class="subtitle">AI Alignment Research with Simulated Robotics</p>
            </div>
            <div id="header-actions" class="header-actions hidden">
                <span id="user-info" class="user-info"></span>
                <button id="logout-btn" class="btn-logout">Logout</button>
            </div>
        </div>
        <div id="login-section" class="card">
            <h2>Authentication</h2>
            <div id="domain-warning" class="warning hidden">Access restricted to authorized domains only.</div>
            <div id="google-login" class="hidden">
                <a href="/auth/google" class="btn-google">
                    <svg viewBox="0 0 24 24"><path fill="white" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="white" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="white" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="white" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                    Sign in with Google
                </a>
            </div>
            <div id="password-divider" class="divider hidden"><span>or</span></div>
            <form id="login-form" class="hidden">
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" placeholder="Enter password" required>
                </div>
                <button type="submit">Login with Password</button>
            </form>
            <p id="login-error" class="hidden" style="color: var(--error); margin-top: 1rem;"></p>
            <div class="security-badge">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>
                Protected by rate limiting &amp; HTTPS
            </div>
        </div>
        <div id="main-section" class="hidden">
            <div class="card">
                <h2>Run Experiment</h2>
                <form id="run-form">
                    <div class="row">
                        <div class="form-group">
                            <label for="scenario">Scenario</label>
                            <select id="scenario">
                                <option value="barrels_lo">barrels_lo (Baseline)</option>
                                <option value="barrels_mi">barrels_mi (Institutional)</option>
                                <option value="barrels_mh">barrels_mh (HP Dose)</option>
                                <option value="barrels_hi">barrels_hi (Max Pressure)</option>
                                <option value="barrels_corrupt">barrels_corrupt (Telemetry Corruption)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="model">Model</label>
                            <select id="model">
                                <option value="robotics">Gemini Robotics ER</option>
                                <option value="gemini2.5">Gemini 2.5 Pro</option>
                                <option value="claude">Claude 3.5 Sonnet</option>
                                <option value="gpt5">GPT-5</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="num-runs">Runs</label>
                            <select id="num-runs">
                                <option value="1">1</option>
                                <option value="3">3</option>
                                <option value="5">5</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" id="run-btn">Start Experiment</button>
                </form>
            </div>
            <div id="status-card" class="card hidden">
                <h2>Status</h2>
                <div id="status-text" class="status running">Starting...</div>
                <div class="progress-bar"><div id="progress-fill" class="progress-bar-fill" style="width: 0%"></div></div>
            </div>
            <div class="card">
                <h2>Recent Results</h2>
                <ul id="results-list" class="results-list"><li><span class="result-meta">Loading...</span></li></ul>
            </div>
        </div>
    </div>
    <script>
        let currentCallId = null;
        const loginSection = document.getElementById('login-section');
        const mainSection = document.getElementById('main-section');
        const headerActions = document.getElementById('header-actions');
        const userInfo = document.getElementById('user-info');
        const logoutBtn = document.getElementById('logout-btn');
        const loginForm = document.getElementById('login-form');
        const loginError = document.getElementById('login-error');
        const googleLogin = document.getElementById('google-login');
        const passwordDivider = document.getElementById('password-divider');
        const domainWarning = document.getElementById('domain-warning');
        const runForm = document.getElementById('run-form');
        const runBtn = document.getElementById('run-btn');
        const statusCard = document.getElementById('status-card');
        const statusText = document.getElementById('status-text');
        const progressFill = document.getElementById('progress-fill');
        const resultsList = document.getElementById('results-list');

        async function checkAuth() {
            try {
                const res = await fetch('/api/results');
                if (res.ok) { showMain(); loadResults(); }
            } catch (e) {}
        }

        logoutBtn.addEventListener('click', async () => {
            await fetch('/api/logout', { method: 'POST' });
            headerActions.classList.add('hidden');
            mainSection.classList.add('hidden');
            loginSection.classList.remove('hidden');
            checkAuthMethods();
        });

        async function checkAuthMethods() {
            try {
                const res = await fetch('/api/auth-methods');
                if (res.ok) {
                    const methods = await res.json();
                    // Show/hide auth options based on server config
                    if (methods.google) {
                        googleLogin.classList.remove('hidden');
                    } else {
                        googleLogin.classList.add('hidden');
                    }
                    if (methods.password) {
                        loginForm.classList.remove('hidden');
                        if (methods.google) {
                            passwordDivider.classList.remove('hidden');
                        }
                    } else {
                        loginForm.classList.add('hidden');
                        passwordDivider.classList.add('hidden');
                    }
                    if (methods.domain_restricted) {
                        domainWarning.classList.remove('hidden');
                    }
                }
            } catch (e) {}
        }

        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const password = document.getElementById('password').value;
            loginError.classList.add('hidden');
            try {
                const res = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }) });
                if (res.ok) { showMain(); loadResults(); }
                else {
                    const data = await res.json();
                    loginError.textContent = data.detail || 'Invalid password';
                    loginError.classList.remove('hidden');
                }
            } catch (e) { loginError.textContent = 'Connection error'; loginError.classList.remove('hidden'); }
        });

        let csrfToken = null;

        async function fetchCsrfToken() {
            try {
                const res = await fetch('/api/csrf-token');
                if (res.ok) {
                    const data = await res.json();
                    csrfToken = data.csrf_token;
                }
            } catch (e) {}
        }

        function showMain() {
            loginSection.classList.add('hidden');
            mainSection.classList.remove('hidden');
            headerActions.classList.remove('hidden');
            fetchCsrfToken();  // Get CSRF token when logged in
        }

        async function loadResults() {
            try {
                const res = await fetch('/api/results');
                if (!res.ok) return;
                const data = await res.json();
                if (data.results.length === 0) { resultsList.innerHTML = '<li><span class="result-meta">No results yet</span></li>'; return; }
                resultsList.innerHTML = data.results.map(r => {
                    const score = r.scores?.safety_behavior_scorer?.value;
                    const scoreDisplay = score != null ? (score * 100).toFixed(0) + '%' : '--';
                    return `<li><div class="result-info"><div class="result-model">${r.model || 'Unknown'}</div><div class="result-meta">${r.scenario}</div></div><div class="result-score">${scoreDisplay}</div></li>`;
                }).join('');
            } catch (e) {}
        }

        runForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const scenario = document.getElementById('scenario').value;
            const model = document.getElementById('model').value;
            const numRuns = parseInt(document.getElementById('num-runs').value);
            runBtn.disabled = true;
            statusCard.classList.remove('hidden');
            statusText.textContent = 'Starting...';
            statusText.className = 'status running';
            progressFill.style.width = '10%';
            try {
                const headers = { 'Content-Type': 'application/json' };
                if (csrfToken) headers['X-CSRF-Token'] = csrfToken;
                const res = await fetch('/api/run', { method: 'POST', headers, body: JSON.stringify({ scenario, model, num_runs: numRuns, reasoning: 'high' }) });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || 'Failed');
                }
                const data = await res.json();
                currentCallId = data.call_id;
                progressFill.style.width = '20%';
                statusText.textContent = 'Running ' + numRuns + ' experiment(s)...';
                pollStatus();
            } catch (e) { statusText.textContent = 'Error: ' + e.message; statusText.className = 'status error'; runBtn.disabled = false; }
        });

        async function pollStatus() {
            if (!currentCallId) return;
            try {
                const res = await fetch('/api/status/' + currentCallId);
                const data = await res.json();
                if (data.status === 'completed') {
                    progressFill.style.width = '100%';
                    statusText.textContent = 'Completed';
                    statusText.className = 'status success';
                    runBtn.disabled = false;
                    currentCallId = null;
                    loadResults();
                } else if (data.status === 'running') { setTimeout(pollStatus, 3000); }
                else { statusText.textContent = 'Error'; statusText.className = 'status error'; runBtn.disabled = false; currentCallId = null; }
            } catch (e) { setTimeout(pollStatus, 5000); }
        }

        checkAuth();
        checkAuthMethods();
    </script>
</body>
</html>"""


# =============================================================================
# Experiment Runner
# =============================================================================


@app.function(
    image=g1_image,
    gpu="T4",  # T4 is sufficient for MuJoCo + inference
    timeout=1800,  # 30 minutes max
    volumes={"/results": results_volume},
    secrets=[
        modal.Secret.from_name("g1-api-keys"),
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
    secrets=[modal.Secret.from_name("g1-api-keys")],  # Access env vars including G1_ALLOWED_IPS
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web_app():
    """FastAPI web application for the G1 platform with OWASP protections."""
    import hashlib
    import logging
    import re
    import time
    import urllib.parse
    from collections import defaultdict
    from datetime import timedelta
    from typing import Literal

    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from jose import jwt
    from pydantic import BaseModel, Field, field_validator
    from starlette.middleware.base import BaseHTTPMiddleware

    # ==========================================================================
    # Security Configuration
    # ==========================================================================

    # JWT settings
    SECRET_KEY = SESSION_SECRET
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS = 24

    # Rate limiting settings (A07: Identification and Authentication Failures)
    RATE_LIMIT_ATTEMPTS = 5  # Max attempts per window
    RATE_LIMIT_WINDOW = 300  # 5 minutes in seconds
    RATE_LIMIT_LOCKOUT = 900  # 15 minutes lockout after exceeding
    login_attempts: dict[str, list[float]] = defaultdict(list)

    # Google OAuth settings (all must be set via Modal secrets)
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    G1_APP_URL = os.environ.get("G1_APP_URL", "")  # e.g., https://your-app.modal.run
    OAUTH_REDIRECT_URI = f"{G1_APP_URL}/auth/google/callback" if G1_APP_URL else ""

    # Security mode: "oauth_only" disables password auth entirely
    AUTH_MODE = os.environ.get("G1_AUTH_MODE", "both")  # "oauth_only", "password_only", "both"

    # Allowed email domains for OAuth (empty = allow all)
    ALLOWED_EMAIL_DOMAINS = os.environ.get("G1_ALLOWED_DOMAINS", "").split(",")
    ALLOWED_EMAIL_DOMAINS = [d.strip() for d in ALLOWED_EMAIL_DOMAINS if d.strip()]

    # Allowed specific emails for OAuth (takes precedence over domain check)
    ALLOWED_EMAILS = os.environ.get("G1_ALLOWED_EMAILS", "").split(",")
    ALLOWED_EMAILS = [e.strip().lower() for e in ALLOWED_EMAILS if e.strip()]

    # IP Allowlisting (empty = allow all)
    ALLOWED_IPS = os.environ.get("G1_ALLOWED_IPS", "").split(",")
    ALLOWED_IPS = [ip.strip() for ip in ALLOWED_IPS if ip.strip()]

    # ==========================================================================
    # Audit Logging (A09: Security Logging and Monitoring Failures)
    # ==========================================================================

    audit_log = logging.getLogger("g1.security.audit")
    audit_log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | AUDIT | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    audit_log.addHandler(handler)

    def log_security_event(event: str, ip: str, details: str = ""):
        """Log security-relevant events for monitoring."""
        audit_log.info(f"{event} | IP={ip} | {details}")

    # ==========================================================================
    # Request Models with Validation (A03: Injection Prevention)
    # ==========================================================================

    class LoginRequest(BaseModel):
        password: str = Field(..., min_length=1, max_length=128)

    class RunRequest(BaseModel):
        scenario: Literal["barrels_lo", "barrels_mi", "barrels_mh", "barrels_hi", "barrels_corrupt"] = "barrels_lo"
        model: Literal["robotics", "gemini2.5", "claude", "opus", "gpt4", "gpt5", "kimi"] = (
            "robotics"
        )
        reasoning: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "high"
        num_runs: int = Field(default=1, ge=1, le=10)  # Max 10 runs per batch
        api_key: str | None = Field(default=None, max_length=256)

        @field_validator("api_key")
        @classmethod
        def validate_api_key(cls, v: str | None) -> str | None:
            if v is None:
                return v
            # Basic sanitization - only allow expected characters
            if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
                raise ValueError("Invalid API key format")
            return v

    # ==========================================================================
    # FastAPI App with Security Middleware
    # ==========================================================================

    app_web = FastAPI(
        title="G1 Alignment Platform",
        docs_url=None,  # Disable Swagger UI in production (A05: Security Misconfiguration)
        redoc_url=None,  # Disable ReDoc in production
        openapi_url=None,  # Disable OpenAPI schema
    )

    # CORS Middleware (A05: Security Misconfiguration)
    cors_origins = [G1_APP_URL] if G1_APP_URL else []
    app_web.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,  # Strict origin from env var
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # Only needed methods
        allow_headers=["Content-Type", "X-CSRF-Token"],  # Include CSRF header
        max_age=3600,
    )

    # Security Headers Middleware (A05: Security Misconfiguration)
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response: Response = await call_next(request)

            # A05: Prevent clickjacking
            response.headers["X-Frame-Options"] = "DENY"

            # A05: Prevent MIME sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"

            # A05: XSS Protection (legacy browsers)
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # A05: HSTS - force HTTPS (1 year, include subdomains)
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            # A05: Referrer policy - don't leak URLs
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # A05: Permissions policy - disable unnecessary features
            response.headers["Permissions-Policy"] = (
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "magnetometer=(), microphone=(), payment=(), usb=()"
            )

            # Cross-origin isolation headers
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

            # Prevent Adobe Flash/PDF cross-domain access
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

            # A03: Content Security Policy (prevents XSS)
            # Note: 'unsafe-inline' needed for embedded styles/scripts
            # In production, use nonces or hashes instead
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "form-action 'self' https://accounts.google.com; "
                "base-uri 'self';"
            )

            return response

    app_web.add_middleware(SecurityHeadersMiddleware)

    # IP Allowlist Middleware (if configured)
    if ALLOWED_IPS:

        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Get client IP
                forwarded = request.headers.get("x-forwarded-for")
                if forwarded:
                    client_ip = forwarded.split(",")[0].strip()
                else:
                    client_ip = request.client.host if request.client else "unknown"

                if client_ip not in ALLOWED_IPS:
                    log_security_event("IP_BLOCKED", client_ip, f"Not in allowlist: {ALLOWED_IPS}")
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Access denied. Your IP is not authorized."},
                    )

                return await call_next(request)

        app_web.add_middleware(IPAllowlistMiddleware)
        audit_log.info(f"IP allowlist enabled: {ALLOWED_IPS}")

    # ==========================================================================
    # Helper Functions
    # ==========================================================================

    def get_client_ip(request: Request) -> str:
        """Get client IP, considering proxies."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check_rate_limit(ip: str) -> tuple[bool, int]:
        """Check if IP is rate limited. Returns (is_allowed, seconds_until_unlock)."""
        now = time.time()
        attempts = login_attempts[ip]

        # Clean old attempts outside the window
        attempts[:] = [t for t in attempts if now - t < RATE_LIMIT_WINDOW]

        if len(attempts) >= RATE_LIMIT_ATTEMPTS:
            # Check if still in lockout period
            oldest_in_window = min(attempts) if attempts else now
            lockout_end = oldest_in_window + RATE_LIMIT_LOCKOUT
            if now < lockout_end:
                return False, int(lockout_end - now)

        return True, 0

    def record_attempt(ip: str):
        """Record a login attempt."""
        login_attempts[ip].append(time.time())

    def create_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        to_encode["exp"] = expire
        # Add issued-at for token freshness checks
        to_encode["iat"] = datetime.now(timezone.utc)
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

    def normalize_email(email: str) -> str:
        """Normalize email to canonical form (handles Gmail aliases, Unicode)."""
        import unicodedata

        # NFKC normalization converts lookalike Unicode chars to ASCII equivalents
        # e.g., Roman numeral small L -> "l", fullwidth @ -> "@"
        email_normalized = unicodedata.normalize("NFKC", email)
        email_lower = email_normalized.lower().strip()

        if "@" not in email_lower:
            return email_lower

        local, domain = email_lower.rsplit("@", 1)

        # Gmail-specific normalization (dots are ignored, plus addressing stripped)
        if domain in ("gmail.com", "googlemail.com"):
            # Remove dots from local part
            local = local.replace(".", "")
            # Remove plus addressing (everything after +)
            if "+" in local:
                local = local.split("+")[0]

        return f"{local}@{domain}"

    def is_email_allowed(email: str) -> bool:
        """Check if email is allowed (specific emails take precedence over domains)."""
        email_normalized = normalize_email(email)

        # If specific emails are configured, ONLY those emails are allowed
        if ALLOWED_EMAILS:
            # Normalize the allowlist entries too for comparison
            allowed_normalized = [normalize_email(e) for e in ALLOWED_EMAILS]
            return email_normalized in allowed_normalized

        # If only domains are configured, check domain
        if ALLOWED_EMAIL_DOMAINS:
            domain = email_normalized.split("@")[-1]
            return domain in [d.lower() for d in ALLOWED_EMAIL_DOMAINS]

        # No restrictions configured
        return True

    # ==========================================================================
    # CSRF Protection (A08: Software and Data Integrity)
    # ==========================================================================

    def generate_csrf_token() -> str:
        """Generate a new CSRF token."""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()

    def verify_csrf_token(request: Request, token: str | None) -> bool:
        """Verify CSRF token matches the one in cookie."""
        cookie_token = request.cookies.get("csrf_token")
        if not cookie_token or not token:
            return False
        return secrets.compare_digest(cookie_token, token)

    @app_web.get("/api/csrf-token")
    async def get_csrf_token(response: Response):
        """Get a CSRF token for forms."""
        token = generate_csrf_token()
        response.set_cookie(
            "csrf_token",
            token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=3600,  # 1 hour
        )
        return {"csrf_token": token}

    def require_csrf(request: Request) -> None:
        """Dependency to validate CSRF token from X-CSRF-Token header."""
        csrf_token = request.headers.get("X-CSRF-Token")

        if not verify_csrf_token(request, csrf_token):
            client_ip = get_client_ip(request)
            log_security_event("CSRF_FAILED", client_ip, f"path={request.url.path}")
            raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # ==========================================================================
    # Authentication Endpoints
    # ==========================================================================

    @app_web.post("/api/login")
    async def login(request: Request, req: LoginRequest):
        client_ip = get_client_ip(request)

        # Check if password auth is disabled
        if AUTH_MODE == "oauth_only":
            log_security_event(
                "LOGIN_BLOCKED", client_ip, "Password auth disabled (oauth_only mode)"
            )
            raise HTTPException(
                status_code=403,
                detail="Password login is disabled. Please use Google Sign-In.",
            )

        # Rate limiting (A07: Brute force protection)
        allowed, wait_time = check_rate_limit(client_ip)
        if not allowed:
            log_security_event("RATE_LIMITED", client_ip, f"Lockout for {wait_time}s")
            raise HTTPException(
                status_code=429,
                detail=f"Too many login attempts. Try again in {wait_time} seconds.",
            )

        # Record this attempt before verification
        record_attempt(client_ip)

        # Verify password (constant-time comparison to prevent timing attacks)
        expected = os.environ.get("G1_AUTH_PASSWORD", AUTH_PASSWORD)
        # Use secrets.compare_digest for timing-safe comparison
        if not secrets.compare_digest(req.password.encode(), expected.encode()):
            remaining = max(0, RATE_LIMIT_ATTEMPTS - len(login_attempts[client_ip]))
            log_security_event("LOGIN_FAILED", client_ip, f"{remaining} attempts remaining")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid password. {remaining} attempts remaining.",
            )

        # Success - clear rate limit attempts
        login_attempts[client_ip] = []
        log_security_event("LOGIN_SUCCESS", client_ip, "method=password")

        token = create_token({"sub": "password_user", "auth_method": "password"})
        response = JSONResponse({"status": "ok"})
        response.set_cookie(
            "session",
            token,
            httponly=True,
            secure=True,  # HTTPS only (A02: Cryptographic Failures)
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        )
        return response

    @app_web.get("/auth/google")
    async def google_auth(request: Request):
        """Redirect to Google OAuth."""
        client_ip = get_client_ip(request)

        if not GOOGLE_CLIENT_ID:
            log_security_event("OAUTH_ERROR", client_ip, "Google OAuth not configured")
            raise HTTPException(status_code=503, detail="Google OAuth not configured")

        # Generate state for CSRF protection (A08: Software and Data Integrity)
        state = hashlib.sha256(secrets.token_bytes(32)).hexdigest()

        log_security_event("OAUTH_STARTED", client_ip, "provider=google")

        params = {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": "email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
        response = RedirectResponse(url)
        response.set_cookie("oauth_state", state, httponly=True, secure=True, max_age=600)
        return response

    @app_web.get("/auth/google/callback")
    async def google_callback(request: Request, code: str = "", state: str = ""):
        """Handle Google OAuth callback."""
        import httpx

        client_ip = get_client_ip(request)

        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            log_security_event("OAUTH_ERROR", client_ip, "OAuth not configured")
            raise HTTPException(status_code=503, detail="Google OAuth not configured")

        # Verify state for CSRF protection (A08: Software and Data Integrity)
        saved_state = request.cookies.get("oauth_state")
        if not saved_state or not secrets.compare_digest(saved_state, state):
            log_security_event("OAUTH_CSRF_FAILED", client_ip, "State mismatch")
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": OAUTH_REDIRECT_URI,
                },
            )

            if token_response.status_code != 200:
                log_security_event("OAUTH_TOKEN_FAILED", client_ip, "Code exchange failed")
                raise HTTPException(status_code=400, detail="Failed to exchange code")

            tokens = token_response.json()
            access_token = tokens.get("access_token")

            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if user_response.status_code != 200:
                log_security_event("OAUTH_USERINFO_FAILED", client_ip, "Failed to get user info")
                raise HTTPException(status_code=400, detail="Failed to get user info")

            user_info = user_response.json()

        email = user_info.get("email", "")

        # Check email restriction (A01: Broken Access Control)
        if not is_email_allowed(email):
            log_security_event("OAUTH_EMAIL_BLOCKED", client_ip, f"email={email}")
            raise HTTPException(status_code=403, detail="Access denied")

        log_security_event("LOGIN_SUCCESS", client_ip, f"method=google, email={email}")

        # Create session token
        token = create_token(
            {
                "sub": email,
                "name": user_info.get("name"),
                "auth_method": "google",
            }
        )

        response = RedirectResponse("/")
        response.set_cookie(
            "session",
            token,
            httponly=True,
            secure=True,  # HTTPS only (A02: Cryptographic Failures)
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        )
        response.delete_cookie("oauth_state")
        return response

    @app_web.get("/api/auth-methods")
    async def auth_methods():
        """Return available authentication methods."""
        password_enabled = AUTH_MODE in ("password_only", "both")
        google_enabled = (
            AUTH_MODE in ("oauth_only", "both") and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
        )
        return {
            "password": password_enabled,
            "google": bool(google_enabled),
            "domain_restricted": bool(ALLOWED_EMAIL_DOMAINS),
            "email_restricted": bool(ALLOWED_EMAILS),
        }

    @app_web.post("/api/logout")
    async def logout(request: Request):
        """Log out and clear session."""
        client_ip = get_client_ip(request)
        log_security_event("LOGOUT", client_ip, "Session cleared")
        response = JSONResponse({"status": "ok"})
        response.delete_cookie("session")
        return response

    # ==========================================================================
    # Protected API Endpoints
    # ==========================================================================

    @app_web.post("/api/run")
    async def start_run(
        request: Request,
        req: RunRequest,
        user: dict = Depends(verify_token),
        _csrf: None = Depends(require_csrf),
    ):
        """Start a new experiment batch (CSRF protected)."""
        client_ip = get_client_ip(request)
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        log_security_event(
            "EXPERIMENT_STARTED",
            client_ip,
            f"user={user.get('sub')} batch={batch_id} scenario={req.scenario} model={req.model}",
        )

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
        # Embedded HTML to avoid Modal mount path issues
        return HTMLResponse(INDEX_HTML)

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
