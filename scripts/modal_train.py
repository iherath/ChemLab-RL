"""Bounded, single-GPU ChemLab experiments on Modal.

Authentication is supplied by the CLI, never bundled with source. Dataset and
checkpoint volumes are separate from the model download cache. The local
entrypoint refuses a workspace other than ish70 before uploading or computing.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "chemlab-vla-pilot"
DATA_VOLUME = "chemlab-rl-data"
RUN_VOLUME = "chemlab-rl-runs"
CACHE_VOLUME = "chemlab-rl-model-cache"
REMOTE_ROOT = "/workspace/chemlab"
OGPO_PROFILES = {
    "engineering": dict(group_size=2, flow_steps=2, critic_warmup_updates=1,
                        controls=20, updates=4, runner_seconds=600, total_seconds=1200, checkpoint_every=1),
    "learning": dict(group_size=4, flow_steps=10, critic_warmup_updates=100,
                     controls=2000, updates=1000, runner_seconds=3600, total_seconds=3900, checkpoint_every=100),
}

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)


def exclude_local(path: Path) -> bool:
    return any(part in {".git", ".venv", "__pycache__", ".pytest_cache", "work", "recordings"}
               for part in path.parts) or path.suffix in {".pyc", ".log"}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libosmesa6", "ffmpeg", "git")
    .pip_install(
        "lerobot[smolvla]==0.7.0", "h5py>=3.11", "numpy>=2,<2.3",
        "mujoco==3.16.0", "gymnasium==1.4.0", "trimesh==5.1.1",
        "imageio[ffmpeg]", "pillow", "pytest",
    )
    .env({
        "HF_HOME": "/model-cache/huggingface",
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": ":".join([
            REMOTE_ROOT, REMOTE_ROOT + "/chem_rl_sim",
            REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim",
        ]),
    })
    .add_local_dir(ROOT / "chemlab_rl", REMOTE_ROOT + "/chemlab_rl", ignore=exclude_local)
    .add_local_dir(ROOT / "scripts", REMOTE_ROOT + "/scripts", ignore=exclude_local)
    .add_local_dir(ROOT / "chem_rl_sim/chem_rl_sim", REMOTE_ROOT + "/chem_rl_sim/chem_rl_sim", ignore=exclude_local)
    .add_local_dir(ROOT / "chem_rl_sim/third_party/urgantry_sim", REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim", ignore=exclude_local)
)


@app.function(
    image=image, gpu="A100-80GB", cpu=8, memory=32768,
    timeout=6 * 3600, startup_timeout=1200, max_containers=1,
    retries=0, scaledown_window=2,
    volumes={"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume},
)
def execute(run_id: str, command: list[str], deadline: float) -> dict:
    return _execute_command(run_id, command, deadline)


@app.function(
    image=image, gpu="A100-80GB", cpu=8, memory=32768,
    timeout=1200, startup_timeout=120, max_containers=1,
    retries=0, scaledown_window=2,
    volumes={"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume},
)
def execute_ogpo_gradient(run_id: str, command: list[str], deadline: float) -> dict:
    """One GPU, no retries, and a hard 20-minute function timeout."""
    if command[:2] != ["-m", "scripts.train_ogpo_vadar"] or "--gradient-smoke" not in command:
        raise ValueError("This bounded function accepts only the OGPO gradient integration check")
    return _execute_command(run_id, command, min(deadline, time.time() + 1170))


@app.function(
    image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768),
    timeout=1200, startup_timeout=120, max_containers=1, retries=0, scaledown_window=2,
    volumes={"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume},
)
def execute_ogpo_engineering(run_id: str, request: dict, deadline: float) -> dict:
    return _execute_ogpo_pilot(run_id, request, deadline, "engineering")


@app.function(
    image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768),
    timeout=3900, startup_timeout=120, max_containers=1, retries=0, scaledown_window=2,
    volumes={"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume},
)
def execute_ogpo_learning(run_id: str, request: dict, deadline: float) -> dict:
    return _execute_ogpo_pilot(run_id, request, deadline, "learning")


@app.function(
    image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768),
    timeout=7200, startup_timeout=120, max_containers=1, retries=0, scaledown_window=2,
    volumes={"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume},
)
def execute_evaluation(run_id: str, command: list[str], deadline: float) -> dict:
    return _execute_evaluation(run_id, command, deadline)


def _execute_evaluation(run_id: str, command: list[str], deadline: float) -> dict:
    if command[:2] != ["-m", "scripts.evaluate_vla"]:
        raise ValueError("The evaluation function accepts only the frozen policy evaluator")
    for flag, maximum in (("--max-steps", 5000), ("--max-seconds", 7200)):
        if flag not in command or not 0 < float(command[command.index(flag) + 1]) <= maximum:
            raise ValueError("Evaluation command exceeds the bounded full-task limits")
    hard_deadline = min(deadline, time.time() + 7200)
    if hard_deadline - time.time() <= 15:
        raise TimeoutError("Evaluation budget expired before startup")
    if (Path("/runs") / run_id).exists():
        raise FileExistsError("Use a new evaluation run id to preserve previous measurements")
    return _execute_command(run_id, command, hard_deadline - 15, termination_grace_seconds=5,
                            launcher_context={"mode": "frozen_policy_evaluation", "hard_deadline_epoch": hard_deadline,
                                              "max_total_seconds": 7200, "gpu_count": 1, "retries": 0})


def _evaluation_completion(directory: Path, *, exit_code: int, timed_out: bool) -> dict:
    """A successful process exit alone does not establish a finished episode."""
    path = directory / "evaluation.json"
    try:
        summary = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError) as error:
        return dict(status="incomplete", requested_episodes=1, reported_episodes=0, completed_episodes=0,
                    termination_reasons=[], summary_path=str(path), summary_read_error=type(error).__name__)
    episodes = summary.get("episodes", [])
    complete = (exit_code == 0 and not timed_out and len(episodes) == 1
                and episodes[0].get("completed_evaluation") is True
                and summary.get("completed_episodes") == 1)
    return dict(status="completed" if complete else "incomplete", requested_episodes=1,
                reported_episodes=len(episodes), completed_episodes=summary.get("completed_episodes", 0),
                termination_reasons=[row.get("termination_reason") for row in episodes],
                summary_path=str(path) if path.is_file() else None)


def _execute_ogpo_pilot(run_id: str, request: dict, deadline: float, mode: str) -> dict:
    if request.get("mode") != mode or request.get("run_id") != run_id:
        raise ValueError("OGPO function/profile identity mismatch")
    command = _ogpo_command(request)
    hard_deadline = min(deadline, time.time() + OGPO_PROFILES[mode]["total_seconds"])
    if hard_deadline - time.time() <= 15:
        raise TimeoutError("OGPO absolute budget expired before model startup")
    from scripts.collect_granular_demos import source_identity
    if source_identity()[0] != request["preflight"]["source_version"]:
        raise ValueError("Remote physics source changed after preflight")
    manifest_path = Path(request["preflight"]["checkpoint_path"]) / "data_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != request["preflight"]["manifest_sha256"]:
        raise ValueError("BC manifest changed after remote-volume preflight")
    if not Path(request["preflight"]["dataset_path"]).is_file():
        raise FileNotFoundError("BC dataset disappeared after remote-volume preflight")
    if (Path("/runs") / run_id).exists():
        raise FileExistsError("Use a new OGPO run id; this wrapper does not overwrite or resume")
    context = {**request, "hard_deadline_epoch": hard_deadline,
               "replay_initialization": "demonstration_seeded_off_policy" if request["demo_replay"] else "online_only",
               "grain_count": 200, "max_episode_steps": 5000,
               "limits": OGPO_PROFILES[mode], "gpu_count": 1, "retries": 0}
    # Reserve time for termination and volume commits inside the absolute cap.
    return _execute_command(run_id, command, hard_deadline - 15,
                            launcher_context=context, termination_grace_seconds=5)


def _execute_command(run_id: str, command: list[str], deadline: float, *,
                     launcher_context: dict | None = None, termination_grace_seconds: int = 45) -> dict:
    """Enforce an absolute deadline across restarts and persist all subprocess logs."""
    import hashlib
    import subprocess
    import sys
    import threading
    import torch

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("Invalid run identifier")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Experiment deadline already expired; refusing another attempt")
    if len(command) < 2 or command[0] != "-m" or command[1] not in {
        "chemlab_rl.vla_bc", "scripts.train_vla_bc", "scripts.evaluate_vla", "scripts.train_ogpo_vadar",
    }:
        raise ValueError("Only project BC, OGPO, and evaluation modules may be launched")
    for option in ("--dataset", "--demo-replay"):
        if option not in command:
            continue
        dataset_index = command.index(option) + 1
        if dataset_index >= len(command):
            raise ValueError("Missing dataset path")
        dataset_path = Path(command[dataset_index])
        if dataset_path.parent != Path("/datasets") or not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset is absent from the mounted data volume: {dataset_path}")
    out = Path("/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metadata = {
        "run_id": run_id, "started_utc_epoch": start,
        "deadline_utc_epoch": deadline, "command": command,
        "gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__),
        "purpose": ("pretrained VLA OGPO gradient integration check" if "--gradient-smoke" in command
                    else "experimental OGPO-VaDar fine tuning" if "scripts.train_ogpo_vadar" in command
                    else "frozen policy evaluation" if "scripts.evaluate_vla" in command
                    else "pretrained VLA behavioral cloning pilot"),
    }
    if launcher_context is not None:
        metadata["launcher"] = launcher_context
    if "scripts.train_ogpo_vadar" in command:
        metadata["source_sha256"] = {
            name: hashlib.sha256((Path(REMOTE_ROOT) / name).read_bytes()).hexdigest()
            for name in ("scripts/train_ogpo_vadar.py", "chemlab_rl/ogpo_training.py",
                         "chemlab_rl/smolvla_ogpo.py", "chemlab_rl/vla_bc.py")
        }
    (out / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_volume.commit()
    proc = subprocess.Popen(
        [sys.executable, *command], cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, "CHEMLAB_DEADLINE_UTC": str(deadline)},
    )
    timed_out = threading.Event()

    def end_at_deadline() -> None:
        timed_out.set()
        proc.terminate()
        try:
            proc.wait(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()

    timer = threading.Timer(max(1.0, deadline - time.time()), end_at_deadline)
    timer.daemon = True
    timer.start()
    try:
        with (out / "console.log").open("a", buffering=1) as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                # A checkpoint announcement follows closed/flushed output files.
                # Make these durable without waiting for the training process.
                try:
                    if json.loads(line).get("event") == "checkpoint":
                        run_volume.commit()
                except (json.JSONDecodeError, AttributeError):
                    pass
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.terminate()
        cache_volume.commit()
        run_volume.commit()
    result = {**metadata, "exit_code": code, "elapsed_seconds": time.time() - start,
              "timed_out": timed_out.is_set()}
    if "scripts.evaluate_vla" in command:
        result["evaluation"] = _evaluation_completion(out, exit_code=code, timed_out=timed_out.is_set())
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    run_volume.commit()
    if code:
        raise RuntimeError(f"Training exited {code}; see {RUN_VOLUME}/{run_id}/console.log")
    return result


def verify_workspace() -> None:
    """Use CLI identity output without printing or reading credential values."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "modal", "profile", "list", "--json"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    profiles = json.loads(result.stdout)
    selected = os.environ.get("MODAL_PROFILE")
    active = [p for p in profiles if p["name"] == selected] if selected else [p for p in profiles if p["active"]]
    if len(active) != 1 or active[0]["workspace"] != "ish70":
        raise RuntimeError("Refusing compute/upload: select the authenticated ish70 Modal profile")
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        raise RuntimeError("Use the verified ish70 config profile, without token environment overrides")


def _dataset_basename(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:h5|hdf5)", value):
        raise ValueError("Remote dataset must be a plain .h5/.hdf5 basename")
    return value


def _require_remote_dataset(name: str) -> None:
    """Check volume metadata without downloading data or allocating a GPU."""
    from modal.volume import FileEntryType
    name = _dataset_basename(name)
    entries = data_volume.listdir("/" + name)
    if not any(entry.path.lstrip("/") == name and entry.type == FileEntryType.FILE for entry in entries):
        raise FileNotFoundError(f"{DATA_VOLUME}/{name}")


def _ogpo_command(request: dict) -> list[str]:
    """Build one of two fixed profiles; callers cannot raise individual budgets."""
    mode = request["mode"]
    if mode not in OGPO_PROFILES:
        raise ValueError("OGPO mode must be engineering or explicitly learning")
    for key in ("checkpoint_run", "run_id"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", request[key]):
            raise ValueError("Invalid OGPO checkpoint/run identifier")
    step, seed, target = request["checkpoint_step"], request["seed"], request["target"]
    if (type(step) is not int or step <= 0 or type(seed) is not int or not 0 <= seed < 2**31
            or not isinstance(target, (int, float)) or not math.isfinite(target) or not .1 <= target <= 3.4
            or type(request["demo_replay"]) is not bool):
        raise ValueError("Require a trained checkpoint, 31-bit seed, valid 200-grain target, and explicit replay flag")
    dataset = _dataset_basename(request["dataset"])
    config = OGPO_PROFILES[mode]
    command = ["-m", "scripts.train_ogpo_vadar", "--checkpoint",
               f"/runs/{request['checkpoint_run']}/checkpoint_{step:06d}",
               "--output", f"/runs/{request['run_id']}", "--device", "cuda", "--seed", str(seed),
               "--grain-count", "200", "--powder-ids", "0", "1", "2", "3", "4",
               "--target-g", str(target), "--max-episode-steps", "5000",
               "--batch-size", "2", "--group-size", str(config["group_size"]),
               "--actor-microbatch-size", "2", "--flow-steps", str(config["flow_steps"]),
               "--critic-warmup-updates", str(config["critic_warmup_updates"]),
               "--discount", "0.999", "--r-step", "0", "--r-done", "5", "--value-cap", "15",
               "--bc-anchor-weight", "0", "--max-environment-steps", str(config["controls"]),
               "--max-updates", str(config["updates"]), "--max-seconds", str(config["runner_seconds"]),
               "--checkpoint-every", str(config["checkpoint_every"])]
    if request["demo_replay"]:
        command.extend(["--demo-replay", "/datasets/" + dataset])
    return command


def _preflight_ogpo(request: dict) -> dict:
    """Read only small manifest/volume metadata before allocating a GPU."""
    from modal.types import FileEntryType
    from chemlab_rl.vla_bc import manifest_state_contract
    from scripts.modal_collect_granular import _source_identity

    _ogpo_command(request)
    directory = f"/{request['checkpoint_run']}/checkpoint_{request['checkpoint_step']:06d}"
    entries = {entry.path.lstrip("/"): entry for entry in run_volume.listdir(directory)
               if entry.type == FileEntryType.FILE}
    required = ("config.json", "model.safetensors", "data_manifest.json", "policy_preprocessor.json",
                "policy_postprocessor.json", "run_config.json", "training_state.pt")
    missing = [name for name in required if directory.lstrip("/") + "/" + name not in entries
               or entries[directory.lstrip("/") + "/" + name].size <= 0]
    if missing:
        raise FileNotFoundError(f"Incomplete BC checkpoint {directory}: {missing}")
    manifest_entry = entries[directory.lstrip("/") + "/data_manifest.json"]
    if manifest_entry.size > 2 * 1024 * 1024:
        raise ValueError("BC manifest exceeds the bounded metadata-read limit")
    content = bytearray()
    for chunk in run_volume.read_file(directory + "/data_manifest.json"):
        content.extend(chunk)
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("BC manifest exceeded its metadata-read limit")
    manifest = json.loads(content)
    if (manifest_state_contract(manifest) != (True, 15) or manifest.get("legacy")
            or manifest.get("use") != "synthetic_coarse_grain_control_data"
            or manifest.get("grain_count") != 200 or manifest.get("heldout_powder_ids") != [5]
            or manifest.get("heldout_episodes") or manifest.get("normalization_fitted_on") != "train_episodes_only"
            or len(manifest.get("train_episodes", [])) not in (5, 30)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("dataset_sha256", ""))):
        raise ValueError("OGPO requires a trained physical 15-D BC5/BC30 checkpoint with the declared split")
    version, _ = _source_identity()
    if manifest.get("source_version") != version:
        raise ValueError("BC data source differs from the current physical collector/environment")
    _require_remote_dataset(request["dataset"])
    if any(entry.path.strip("/") == request["run_id"] for entry in run_volume.listdir("/")):
        raise FileExistsError("Use a new OGPO run id; this wrapper does not overwrite or resume")
    return dict(checkpoint_path="/runs" + directory, manifest_sha256=hashlib.sha256(content).hexdigest(),
                dataset_path="/datasets/" + request["dataset"], expected_dataset_sha256=manifest["dataset_sha256"],
                source_version=version, state_dim=15, grain_count=200,
                training_episodes=len(manifest["train_episodes"]), checkpoint_files=list(required))


@app.local_entrypoint()
def ogpo_pilot(checkpoint_run: str, checkpoint_step: int, dataset: str, run_id: str,
               mode: str = "engineering", demo_replay: bool = False, seed: int = 0,
               target: float = .5) -> None:
    """Opt-in engineering/learning profiles; replay seeding remains separate."""
    request = dict(checkpoint_run=checkpoint_run, checkpoint_step=checkpoint_step, dataset=dataset,
                   run_id=run_id, mode=mode, demo_replay=demo_replay, seed=seed, target=target)
    _ogpo_command(request)
    verify_workspace()
    request["preflight"] = _preflight_ogpo(request)
    local = ROOT / "docs/pilot" / run_id
    local.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + OGPO_PROFILES[mode]["total_seconds"]
    launch = {**request, "started_epoch": started, "hard_deadline_epoch": deadline,
              "replay_initialization": "demonstration_seeded_off_policy" if demo_replay else "online_only",
              "limits": OGPO_PROFILES[mode], "status": "submitted"}
    (local / "wrapper_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps({"event": "ogpo_pilot_preflight", **launch}), flush=True)
    try:
        function = execute_ogpo_engineering if mode == "engineering" else execute_ogpo_learning
        result = function.remote(run_id, request, deadline)
        (local / "wrapper_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except Exception as error:
        (local / "wrapper_failure.json").write_text(json.dumps(
            {"status": "failed", "error_type": type(error).__name__, "elapsed_seconds": time.time() - started,
             "remote_artifacts": f"{RUN_VOLUME}/{run_id}", "retries": 0}, indent=2) + "\n")
        raise


@app.local_entrypoint()
def main(
    dataset: str = "data/teleop/scrape_vial_bc.hdf5",
    run_id: str = "smolvla-diagnostic-001",
    steps: int = 100,
    batch_size: int = 8,
    max_minutes: int = 20,
    allow_legacy_diagnostic: bool = False,
    upload: bool = False,
    remote_dataset: str = "",
) -> None:
    """An upload must be explicitly requested; default run is capped at 20 minutes."""
    verify_workspace()
    if not 5 <= max_minutes <= 300:
        raise ValueError("Each pilot must be between 5 and 300 minutes, including load/save margin")
    if not 1 <= steps <= 20000 or not 1 <= batch_size <= 64:
        raise ValueError("Steps/batch size outside the bounded pilot range")
    if remote_dataset:
        if upload:
            raise ValueError("--remote-dataset and --upload are mutually exclusive")
        name = _dataset_basename(remote_dataset)
    else:
        path = Path(dataset).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = _dataset_basename(path.name)
        if upload:
            with data_volume.batch_upload() as batch:
                batch.put_file(path, "/" + name)
    _require_remote_dataset(name)
    remote = "/" + name
    command = [
        "-m", "chemlab_rl.vla_bc", "--dataset", "/datasets" + remote,
        "--output", "/runs/" + run_id, "--steps", str(steps),
        "--batch-size", str(batch_size), "--max-seconds", str(max_minutes * 60 - 60),
        "--heldout-powders", "5", "--bf16",
    ]
    if allow_legacy_diagnostic:
        command.append("--allow-legacy-diagnostic")
    # One fixed deadline is passed by the caller and survives remote rescheduling.
    result = execute.remote(run_id, command, time.time() + max_minutes * 60)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def evaluate(
    checkpoint_run: str,
    checkpoint_step: int,
    run_id: str = "smolvla-eval-001",
    split: str = "seen",
    powder: int = 0,
    max_minutes: int = 120,
    max_steps: int = 5000,
    target: float = 0.5,
    seed: int = 10000,
) -> None:
    verify_workspace()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", checkpoint_run):
        raise ValueError("Invalid checkpoint run")
    if split not in {"seen", "heldout"} or not 5 <= max_minutes <= 120:
        raise ValueError("Select seen/heldout and a 5–120 minute evaluation cap")
    if not 0 <= powder <= 5 or not 1 <= max_steps <= 5000 or checkpoint_step < 0:
        raise ValueError("Invalid powder, step count or checkpoint")
    if not math.isfinite(target) or target <= 0 or not 0 <= seed < 2**31:
        raise ValueError("Evaluation requires a finite positive target and a nonnegative 31-bit seed")
    checkpoint = f"/runs/{checkpoint_run}/checkpoint_{checkpoint_step:06d}"
    command = ["-m", "scripts.evaluate_vla", "--checkpoint", checkpoint,
               "--output", f"/runs/{run_id}", "--split", split, "--powders", str(powder),
               "--targets", str(target), "--seed", str(seed),
               "--episodes-per-powder", "1",
               "--max-steps", str(max_steps), "--max-seconds", str(max_minutes * 60 - 120), "--video"]
    print(json.dumps(execute_evaluation.remote(run_id, command, time.time() + max_minutes * 60), indent=2))


@app.local_entrypoint()
def verify(checkpoint_run: str, checkpoint_step: int, dataset: str = "scrape_vial_bc.hdf5") -> None:
    verify_workspace()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", checkpoint_run):
        raise ValueError("Invalid checkpoint run")
    if Path(dataset).name != dataset or checkpoint_step < 0:
        raise ValueError("Invalid dataset/checkpoint")
    checkpoint = f"/runs/{checkpoint_run}/checkpoint_{checkpoint_step:06d}"
    command = ["-m", "chemlab_rl.vla_bc", "--dataset", f"/datasets/{dataset}",
               "--verify-checkpoint", checkpoint, "--device", "cuda"]
    print(json.dumps(execute.remote(f"{checkpoint_run}-verify", command, time.time() + 900), indent=2))


@app.local_entrypoint()
def ogpo_gradient_smoke(
    checkpoint_run: str = "smolvla-api-smoke-003",
    checkpoint_step: int = 100,
    dataset: str = "legacy_192frame_smoke.hdf5",
    run_id: str = "ogpo-gradient-smoke-001",
    max_minutes: int = 20,
    flow_steps: int = 2,
) -> None:
    """Verify real pretrained path-density gradients without online training."""
    verify_workspace()
    for value in (checkpoint_run, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
            raise ValueError("Invalid checkpoint/run identifier")
    if Path(dataset).name != dataset or checkpoint_step < 0:
        raise ValueError("Invalid dataset/checkpoint")
    if not 2 <= max_minutes <= 20 or not 1 <= flow_steps <= 10:
        raise ValueError("Gradient checks require a 2--20 minute cap and 1--10 flow steps")
    checkpoint = f"/runs/{checkpoint_run}/checkpoint_{checkpoint_step:06d}"
    command = ["-m", "scripts.train_ogpo_vadar", "--checkpoint", checkpoint,
               "--dataset", f"/datasets/{dataset}", "--output", f"/runs/{run_id}",
               "--gradient-smoke", "--flow-steps", str(flow_steps), "--device", "cuda"]
    deadline = time.time() + max_minutes * 60 - 30
    print(json.dumps(execute_ogpo_gradient.remote(run_id, command, deadline), indent=2))
