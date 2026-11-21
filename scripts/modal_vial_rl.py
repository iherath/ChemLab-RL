"""Bounded native-vial baselines, optimizer smoke, and three-method comparison.

No automatic retries or implicit comparison launch. A comparison requires a
successful real-optimizer smoke artifact for the identical BC/data/code inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

import modal

from scripts.modal_train import (ROOT, REMOTE_ROOT, DATA_VOLUME, RUN_VOLUME, image,
                                data_volume, run_volume, cache_volume, verify_workspace,
                                _dataset_basename, _require_remote_dataset)

app = modal.App("chemlab-vial-rl-comparison")
LIMITS = {
    "smoke": dict(total_seconds=900, runner_seconds=600, batch_size=2, group_size=2,
                  warmup=1, updates=3, controls=4, eval_controls=4, checkpoint_every=1),
    "baseline_ode": dict(total_seconds=1200),
    "baseline_stochastic": dict(total_seconds=1200),
    "comparison": dict(total_seconds=4500, runner_seconds=3600, batch_size=4, group_size=4,
                       warmup=100, updates=900, controls=1600, eval_controls=1600, checkpoint_every=25),
}
ALGORITHM_FILES = ("chemlab_rl/vial_ogpo_training.py", "chemlab_rl/ogpo_training.py", "chemlab_rl/smolvla_ogpo.py",
                   "chemlab_rl/vial_vla_data.py", "scripts/train_vial_ogpo.py")


def _identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Invalid run identifier")
    return value


def algorithm_identity(root=ROOT):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ALGORITHM_FILES}


def validate_request(request):
    _identifier(request["run_id"])
    _identifier(request["checkpoint_run"])
    _dataset_basename(request["dataset"])
    checkpoint_limit = 500000 if request["mode"] == "baseline_ode" else 10000
    if type(request["checkpoint_step"]) is not int or not 1 <= request["checkpoint_step"] <= checkpoint_limit:
        raise ValueError(f"Choose a saved trained BC checkpoint step1..{checkpoint_limit}")
    if request["mode"] not in LIMITS or request["method"] not in {"bon", "ogpo", "vadar"}:
        raise ValueError("Unknown bounded vial mode/method")
    if request["mode"].startswith("baseline_") and request["method"] != "ogpo":
        raise ValueError("BC baselines use scalarOGPO plumbing without BoN/VaDar")
    if type(request["seed"]) is not int or not 0 <= request["seed"] < 2**31 - 2000:
        raise ValueError("Invalid shared random seed")


def build_command(request):
    validate_request(request)
    mode = request["mode"]
    limits = LIMITS[mode]
    checkpoint = f"/runs/{request['checkpoint_run']}/checkpoint_{request['checkpoint_step']:06d}"
    output = "/runs/" + request["run_id"]
    if mode == "baseline_ode":
        return ["-m", "scripts.evaluate_vial_vla", "--checkpoint", checkpoint, "--output", output,
                "--max-steps", "1600", "--max-seconds", "1080", "--seed", str(request["seed"] + 1000)]
    command = ["-m", "scripts.train_vial_ogpo", "--checkpoint", checkpoint,
               "--dataset", "/datasets/" + request["dataset"], "--output", output,
               "--method", request["method"], "--seed", str(request["seed"]),
               "--flow-steps", "10", "--diffusion-scale", ".05", "--best-of-n", "8",
               "--actor-microbatch-size", "1", "--actor-lr", "1e-6", "--critic-lr", "3e-4",
               "--critic-members", "2", "--critic-hidden", "256", "--max-episode-steps", "1600",
               "--eval-episodes", "1", "--replay-capacity", "30000"]
    if mode == "baseline_stochastic":
        return command + ["--baseline-only", "--eval-max-steps", "1600"]
    for flag, field in (("--batch-size", "batch_size"), ("--group-size", "group_size"),
                        ("--critic-warmup-updates", "warmup"), ("--max-updates", "updates"),
                        ("--max-environment-steps", "controls"), ("--eval-max-steps", "eval_controls"),
                        ("--max-seconds", "runner_seconds"), ("--checkpoint-every", "checkpoint_every")):
        command += [flag, str(limits[field])]
    if mode == "smoke":
        command.append("--no-bf16")
    return command


def _read_json(path):
    payload = bytearray()
    for chunk in run_volume.read_file(path):
        payload.extend(chunk)
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("Bounded metadata read exceeds2MiB")
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def preflight(request):
    """Small volume metadata and local source only; no GPU or model download."""
    from modal.types import FileEntryType
    from chemlab_rl.vial_vla_data import manifest_state_contract
    from chemlab_rl.vial_vla_env import source_identity
    validate_request(request)
    directory = f"/{request['checkpoint_run']}/checkpoint_{request['checkpoint_step']:06d}"
    entries = {entry.path.lstrip("/"): entry for entry in run_volume.listdir(directory) if entry.type == FileEntryType.FILE}
    required = ("config.json", "model.safetensors", "data_manifest.json", "policy_preprocessor.json",
                "policy_postprocessor.json", "run_config.json", "heldout_evaluation.json")
    for name in required:
        key = directory.lstrip("/") + "/" + name
        if key not in entries or entries[key].size <= 0:
            raise FileNotFoundError(f"Incomplete BC checkpoint: {name}")
    manifest, digest = _read_json(directory + "/data_manifest.json")
    manifest_state_contract(manifest)
    metric, _ = _read_json(directory + "/heldout_evaluation.json")
    if metric.get("completed") is not True:
        raise ValueError("Shared BC checkpoint must have a completed held-out metric")
    _require_remote_dataset(request["dataset"])
    if any(entry.path.strip("/") == request["run_id"] for entry in run_volume.listdir("/")):
        raise FileExistsError("This comparison never overwrites an existing run")
    return dict(checkpoint_path="/runs" + directory, manifest_sha256=digest,
                dataset_path="/datasets/" + request["dataset"], dataset_sha256=manifest["dataset_sha256"],
                simulation_source=source_identity(), algorithm_sources=algorithm_identity(), task_id=manifest["task_id"])


def validate_smoke(result, expected):
    actual = result.get("preflight", {})
    training = result.get("training") or {}
    if (result.get("mode") != "smoke" or result.get("exit_code") != 0 or result.get("timed_out") is not False
            or training.get("actor_updates", 0) < 1
            or training.get("critic_updates", 0) < 1
            or training.get("density_check", {}).get("passed") is not True
            or any(actual.get(key) != expected.get(key) for key in
                   ("checkpoint_path", "manifest_sha256", "dataset_sha256", "simulation_source", "algorithm_sources"))):
        raise ValueError("Comparison requires a passed real-optimizer smoke for identical BC/data/algorithm/simulation")


def evaluation_completed(evaluation):
    if not evaluation:
        return False
    if "completed_evaluation" in evaluation:
        return evaluation["completed_evaluation"] is True
    episodes = evaluation.get("episodes", [])
    return bool(episodes) and all(row.get("terminated") or row.get("controls", 0) >= 1600 for row in episodes)


def _execute(request, deadline, expected_mode):
    import subprocess
    import sys
    import threading
    import torch
    from chemlab_rl.vial_vla_data import file_sha256
    from chemlab_rl.vial_vla_env import source_identity
    if request["mode"] not in expected_mode:
        raise ValueError("Bounded function/profile mismatch")
    command = build_command(request)
    deadline = min(deadline, time.time() + LIMITS[request["mode"]]["total_seconds"])
    if deadline - time.time() < 90:
        raise TimeoutError("Absolute job budget expired before startup")
    proof = request["preflight"]
    if (source_identity() != proof["simulation_source"] or algorithm_identity(Path(REMOTE_ROOT)) != proof["algorithm_sources"]
            or file_sha256(Path(proof["checkpoint_path"]) / "data_manifest.json") != proof["manifest_sha256"]
            or file_sha256(proof["dataset_path"]) != proof["dataset_sha256"]):
        raise ValueError("Remote checkpoint/data/source changed after preflight")
    out = Path("/runs") / request["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = dict(**request, command=command, started_epoch=started, hard_deadline_epoch=deadline,
                    limits=LIMITS[request["mode"]], retries=0, gpu_count=1, gpu=torch.cuda.get_device_name(0))
    (out / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_volume.commit()
    proc = subprocess.Popen([sys.executable, *command], cwd=REMOTE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env={**os.environ, "CHEMLAB_DEADLINE_UTC": str(deadline - 15)})
    timed_out = threading.Event()

    def terminate():
        timed_out.set()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    timer = threading.Timer(max(1, deadline - time.time() - 10), terminate)
    timer.daemon = True
    timer.start()
    try:
        with (out / "console.log").open("a", buffering=1) as log:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            terminate()
        cache_volume.commit()
        run_volume.commit()
    read = lambda name: json.loads((out / name).read_text()) if (out / name).exists() else None
    evaluation = read("evaluation.json")
    result = dict(**metadata, exit_code=code, timed_out=timed_out.is_set(), elapsed_seconds=time.time() - started,
                  training=read("run_result.json"), evaluation=evaluation,
                  status_scope="process_execution_only", full_episode_evaluated=evaluation_completed(evaluation),
                  status="completed" if code == 0 and not timed_out.is_set() else "incomplete")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    run_volume.commit()
    return result


_volumes = {"/datasets": data_volume.with_mount_options(read_only=True), "/runs": run_volume, "/model-cache": cache_volume}


@app.function(image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768), timeout=900,
              startup_timeout=120, max_containers=1, retries=0, scaledown_window=2, volumes=_volumes)
def execute_smoke(request, deadline):
    return _execute(request, deadline, {"smoke"})


@app.function(image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768), timeout=1200,
              startup_timeout=120, max_containers=2, retries=0, scaledown_window=2, volumes=_volumes)
def execute_baseline(request, deadline):
    return _execute(request, deadline, {"baseline_ode", "baseline_stochastic"})


@app.function(image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768), timeout=4500,
              startup_timeout=120, max_containers=3, retries=0, scaledown_window=2, volumes=_volumes)
def execute_comparison(request, deadline):
    return _execute(request, deadline, {"comparison"})


def _request(run_id, checkpoint_run, checkpoint_step, dataset, seed, mode, method="ogpo"):
    request = dict(run_id=run_id, checkpoint_run=checkpoint_run, checkpoint_step=checkpoint_step,
                   dataset=dataset, seed=seed, mode=mode, method=method)
    request["preflight"] = preflight(request)
    return request


def _single(request, function):
    out = ROOT / "docs/pilot" / request["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + LIMITS[request["mode"]]["total_seconds"]
    (out / "wrapper_launch.json").write_text(json.dumps(dict(**request, deadline_epoch=deadline), indent=2) + "\n")
    try:
        result = function.remote(request, deadline)
        (out / "wrapper_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result.get("exit_code") != 0 or result.get("timed_out"):
            raise RuntimeError("Vial process failed or exceeded its cap; saved wrapper_result preserves evidence")
        return result
    except Exception as error:
        (out / "wrapper_failure.json").write_text(json.dumps(dict(error_type=type(error).__name__, retries=0), indent=2) + "\n")
        raise


@app.local_entrypoint()
def smoke(checkpoint_run: str, checkpoint_step: int, run_id: str = "smolvla-vial-optimizer-smoke-001",
          dataset: str = "weigh_vial_grok_v2.hdf5", seed: int = 0, method: str = "ogpo"):
    verify_workspace()
    return _single(_request(run_id, checkpoint_run, checkpoint_step, dataset, seed, "smoke", method), execute_smoke)


@app.local_entrypoint()
def baseline(checkpoint_run: str, checkpoint_step: int, run_id: str, sampler: str = "ode",
             dataset: str = "weigh_vial_grok_v2.hdf5", seed: int = 0):
    if sampler not in {"ode", "stochastic"}:
        raise ValueError("Baseline sampler must be ode or stochastic")
    verify_workspace()
    return _single(_request(run_id, checkpoint_run, checkpoint_step, dataset, seed, "baseline_" + sampler), execute_baseline)


@app.local_entrypoint()
def compare(checkpoint_run: str, checkpoint_step: int, batch_id: str, smoke_result: str,
            dataset: str = "weigh_vial_grok_v2.hdf5", seed: int = 0):
    """Explicitly start the three fixed methods together after a passed smoke."""
    verify_workspace()
    _identifier(batch_id)
    proof = json.loads(Path(smoke_result).read_text())
    requests = [_request(batch_id + "-" + method, checkpoint_run, checkpoint_step, dataset, seed, "comparison", method)
                for method in ("bon", "ogpo", "vadar")]
    for request in requests:
        validate_smoke(proof, request["preflight"])
    out = ROOT / "docs/pilot" / batch_id
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + LIMITS["comparison"]["total_seconds"]
    manifest = dict(requests=requests, deadline_epoch=deadline, max_parallel_gpus=3, retries=0, jobs=[])

    def save():
        (out / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    calls = []
    try:
        for request in requests:
            call = execute_comparison.spawn(request, deadline)
            calls.append(call)
            manifest["jobs"].append(dict(run_id=request["run_id"], method=request["method"], call_id=call.object_id, status="submitted"))
            save()
    except Exception as error:
        manifest["submission_failure"] = type(error).__name__
        save()
    pending = set(range(len(calls)))
    while pending and time.time() < deadline + 30:
        for i in list(pending):
            try:
                result = calls[i].get(timeout=0)
                manifest["jobs"][i].update(status=result["status"], result=result)
                pending.remove(i)
                save()
            except modal.exception.TimeoutError:
                pass
            except Exception as error:
                manifest["jobs"][i].update(status="failed", error_type=type(error).__name__)
                pending.remove(i)
                save()
        if pending:
            time.sleep(2)
    for i in pending:
        calls[i].cancel()
        manifest["jobs"][i]["status"] = "global_deadline"
    manifest["completed_methods"] = [j["method"] for j in manifest["jobs"] if j["status"] == "completed"]
    save()
    print(json.dumps(manifest), flush=True)
    if len(manifest["completed_methods"]) != 3:
        raise RuntimeError("One or more comparison methods did not complete; preserved all independent outcomes")
