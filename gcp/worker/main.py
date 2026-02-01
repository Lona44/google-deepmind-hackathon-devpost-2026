"""
G1 Experiment Worker for GCP.

Processes experiment jobs from Cloud Tasks/Pub/Sub, runs MuJoCo simulations
with AI, and saves results to Cloud Storage.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from google.cloud import firestore, storage

# Add project root to path for imports
sys.path.insert(0, "/app")

from trajectory_recorder import TrajectoryRecorder


def get_firestore_client():
    """Get Firestore client."""
    return firestore.Client()


def get_storage_client():
    """Get Cloud Storage client."""
    return storage.Client()


def get_user_api_key(user_id: str, vendor: str) -> str | None:
    """
    Retrieve decrypted API key for a user.

    In production, keys are encrypted with KMS.
    """
    import base64

    db = get_firestore_client()
    key_ref = db.collection("users").document(user_id).collection("api_keys").document(vendor)
    key_doc = key_ref.get()

    if not key_doc.exists:
        return None

    key_data = key_doc.to_dict()
    encrypted = key_data.get("encrypted_key")

    if not encrypted:
        return None

    # TODO: Replace with KMS decryption in production
    return base64.b64decode(encrypted.encode()).decode()


def update_experiment_status(
    user_id: str,
    experiment_id: str,
    status: str,
    error: str | None = None,
    results_url: str | None = None,
):
    """Update experiment status in Firestore."""
    db = get_firestore_client()
    exp_ref = (
        db.collection("users").document(user_id).collection("experiments").document(experiment_id)
    )

    update_data = {
        "status": status,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    if status == "running":
        update_data["started_at"] = firestore.SERVER_TIMESTAMP
    elif status in ("completed", "failed"):
        update_data["completed_at"] = firestore.SERVER_TIMESTAMP

    if error:
        update_data["error"] = error
    if results_url:
        update_data["results_url"] = results_url

    exp_ref.update(update_data)


def upload_results(
    user_id: str,
    experiment_id: str,
    trajectory_path: Path,
    video_path: Path | None = None,
    debrief_path: Path | None = None,
) -> str:
    """
    Upload experiment results to Cloud Storage.

    Returns the base URL for results.
    """
    client = get_storage_client()
    bucket = client.bucket("g1-results")

    base_path = f"{user_id}/{experiment_id}"

    # Upload trajectory
    blob = bucket.blob(f"{base_path}/trajectory.json")
    blob.upload_from_filename(str(trajectory_path))

    # Upload video if exists
    if video_path and video_path.exists():
        blob = bucket.blob(f"{base_path}/video.mp4")
        blob.upload_from_filename(str(video_path))

    # Upload debrief if exists
    if debrief_path and debrief_path.exists():
        blob = bucket.blob(f"{base_path}/debrief.json")
        blob.upload_from_filename(str(debrief_path))

    return f"gs://g1-results/{base_path}"


def run_experiment(
    experiment_id: str,
    user_id: str,
    scenario: str,
    model: str,
    reasoning: str,
) -> dict:
    """
    Run a single experiment with trajectory recording.

    This is a simplified version - the full implementation would use
    the existing simulation code from src/.
    """
    import mujoco
    import numpy as np

    # Set up environment
    api_key = get_user_api_key(user_id, "gemini")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # Update status to running
    update_experiment_status(user_id, experiment_id, "running")

    # Initialize trajectory recorder
    recorder = TrajectoryRecorder(
        experiment_id=experiment_id,
        model="g1/g1_12dof.xml",
        target_fps=30,
    )
    recorder.set_metadata(
        {
            "scenario": scenario,
            "model": model,
            "reasoning": reasoning,
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
        }
    )

    try:
        # Import simulation components
        from src.config import load_scenario
        from src.simulation_state import SimulationState

        # Load scenario
        scenario_config = load_scenario(scenario)

        # Initialize simulation
        sim = SimulationState(scenario=scenario_config)
        sim.initialize()

        recorder.start()
        recorder.record_event(
            "simulation_initialized",
            {
                "scenario": scenario,
                "goal": list(scenario_config.goal),
            },
        )

        # Run simulation loop
        # This is simplified - real implementation would integrate with
        # the Inspect AI eval framework

        attempt = 1
        max_attempts = 5
        sim_time = 0.0
        dt = 0.02  # 50Hz simulation

        while attempt <= max_attempts and sim_time < 600:  # 10 min max
            # Get current state
            robot_pos = sim.robot.get_position(sim.data)
            robot_heading = sim.robot.get_heading(sim.data)
            battery = sim.battery.charge_percent

            # Record frame
            recorder.record_frame(
                sim_time=sim_time,
                qpos=list(sim.data.qpos),
                qvel=list(sim.data.qvel),
                robot_position=(robot_pos[0], robot_pos[1]),
                robot_heading=robot_heading,
                battery=battery,
                attempt=attempt,
            )

            # Check for goal reached
            goal_dist = np.sqrt(
                (robot_pos[0] - scenario_config.goal[0]) ** 2
                + (robot_pos[1] - scenario_config.goal[1]) ** 2
            )

            if goal_dist < 0.5:
                recorder.record_event(
                    "goal_reached",
                    {
                        "position": [robot_pos[0], robot_pos[1]],
                        "distance": goal_dist,
                    },
                )
                break

            # Check for battery depleted
            if battery <= 0:
                recorder.record_event(
                    "battery_depleted",
                    {
                        "position": [robot_pos[0], robot_pos[1]],
                        "attempt": attempt,
                    },
                )
                attempt += 1
                if attempt <= max_attempts:
                    recorder.record_event("attempt_start", {"attempt": attempt})
                    sim.battery.reset()

            # Step simulation
            mujoco.mj_step(sim.model, sim.data)
            sim_time += dt

        recorder.stop()

        # Save results to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save trajectory
            trajectory_path = tmpdir / "trajectory.json"
            recorder.save(trajectory_path)

            # Upload to Cloud Storage
            results_url = upload_results(
                user_id=user_id,
                experiment_id=experiment_id,
                trajectory_path=trajectory_path,
            )

            # Update status
            update_experiment_status(
                user_id=user_id,
                experiment_id=experiment_id,
                status="completed",
                results_url=results_url,
            )

        return {
            "status": "completed",
            "frames": recorder.frame_count,
            "duration": recorder.duration,
            "results_url": results_url,
        }

    except Exception as e:
        recorder.stop()
        update_experiment_status(
            user_id=user_id,
            experiment_id=experiment_id,
            status="failed",
            error=str(e),
        )
        raise


def process_job(job_data: dict):
    """Process a single experiment job."""
    print(f"Processing job: {job_data}")

    result = run_experiment(
        experiment_id=job_data["experiment_id"],
        user_id=job_data["user_id"],
        scenario=job_data.get("scenario", "barrels_lo"),
        model=job_data.get("model", "robotics"),
        reasoning=job_data.get("reasoning", "high"),
    )

    print(f"Job completed: {result}")
    return result


def main():
    """
    Main entry point for worker.

    Can be run as:
    1. Cloud Run Job - receives job data from environment
    2. Pub/Sub push - receives job data from HTTP request
    3. CLI - receives job data from command line
    """
    import argparse

    parser = argparse.ArgumentParser(description="G1 Experiment Worker")
    parser.add_argument("--job", type=str, help="Job data as JSON")
    args = parser.parse_args()

    if args.job:
        # CLI mode
        job_data = json.loads(args.job)
        process_job(job_data)
    elif os.environ.get("CLOUD_RUN_JOB"):
        # Cloud Run Job mode - get job from environment
        job_data = json.loads(os.environ.get("JOB_DATA", "{}"))
        process_job(job_data)
    else:
        print("No job data provided. Use --job or set JOB_DATA environment variable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
