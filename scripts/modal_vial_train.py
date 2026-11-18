"""One bounded A100 vial BC job; separate from all powder launchers and gates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

import modal

from scripts.modal_train import (ROOT, REMOTE_ROOT, DATA_VOLUME, RUN_VOLUME, image,
                                data_volume, run_volume, cache_volume,
                                verify_workspace, _dataset_basename, _require_remote_dataset)

app = modal.App("chemlab-vial-vla-pilot")


def validate_request(request):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", request["run_id"]):
        raise ValueError("Invalid vial run identifier")
    _dataset_basename(request["dataset"])
    if not re.fullmatch(r"[a-f0-9]{64}", request["dataset_sha256"]):
        raise ValueError("Expected a full dataset SHA256")
    if not 5 <= request["max_minutes"] <= 60 or not 1 <= request["steps"] <= 10000 or not 1 <= request["batch_size"] <= 32:
        raise ValueError("Vial BC cap:5–60minutes,1–10000steps,1–32batch")
    if not 1 <= request["save_every"] <= request["steps"]:
        raise ValueError("Checkpoint interval must be within the requested update budget")
    if request.get("split") is not None:
        from chemlab_rl.vial_vla_data import validate_split
        split = request["split"]
        if set(split) != {"train_episodes", "heldout_episodes", "excluded_episodes"}:
            raise ValueError("Explicit split requires train/heldout/excluded episode lists")
        validate_split(split["train_episodes"], split["heldout_episodes"], split["excluded_episodes"])


def build_command(request):
    validate_request(request)
    command = ["-m", "scripts.train_vial_vla_bc", "--dataset", "/datasets/" + request["dataset"],
            "--output", "/runs/" + request["run_id"], "--steps", str(request["steps"]),
            "--batch-size", str(request["batch_size"]), "--max-seconds", str(request["max_minutes"] * 60 - 180),
            "--save-every", str(request["save_every"]), "--seed", str(request["seed"]),
            "--chunk-size", "50", "--n-action-steps", "2", "--heldout-samples", "128", "--bf16"]
    if request.get("split") is not None:
        command += ["--split-json", "/runs/" + request["run_id"] + "/split.json"]
    return command


@app.function(image=image, gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768),
              timeout=3600, startup_timeout=120, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/datasets": data_volume.with_mount_options(read_only=True),
                       "/runs": run_volume, "/model-cache": cache_volume})
def execute(request: dict, deadline: float) -> dict:
    """Absolute caller deadline includes startup; no retry or overwrite."""
    import subprocess
    import sys
    import threading
    import torch
    from chemlab_rl.vial_vla_data import file_sha256

    command = build_command(request)
    deadline = min(deadline, time.time() + request["max_minutes"] * 60)
    if deadline - time.time() < 180:
        raise TimeoutError("Vial BC deadline expired before startup")
    dataset = Path("/datasets") / request["dataset"]
    if not dataset.is_file() or file_sha256(dataset) != request["dataset_sha256"]:
        raise ValueError("Remote vial dataset does not match verified local bytes")
    out = Path("/runs") / request["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    if request.get("split") is not None:
        (out / "split.json").write_text(json.dumps(request["split"], indent=2) + "\n")
    started = time.time()
    metadata = dict(**request, command=command, started_epoch=started, hard_deadline_epoch=deadline,
                    gpu=torch.cuda.get_device_name(0), gpu_count=1, retries=0,
                    source_files_sha256={name: hashlib.sha256((Path(REMOTE_ROOT) / name).read_bytes()).hexdigest()
                                         for name in ("chemlab_rl/vial_vla_data.py", "chemlab_rl/vla_bc.py",
                                                      "scripts/train_vial_vla_bc.py", "scripts/modal_vial_train.py")})
    (out / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_volume.commit()
    proc = subprocess.Popen([sys.executable, *command], cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "CHEMLAB_DEADLINE_UTC": str(deadline - 15)})
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
                try:
                    if json.loads(line).get("event") == "checkpoint":
                        run_volume.commit()
                except (ValueError, AttributeError):
                    pass
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            terminate()
        cache_volume.commit()
        run_volume.commit()
    training_file = out / "training_result.json"
    training = json.loads(training_file.read_text()) if training_file.exists() else None
    result = dict(**metadata, exit_code=code, timed_out=timed_out.is_set(), elapsed_seconds=time.time() - started,
                  training=training, status="completed" if code == 0 and training and training["steps"] == request["steps"] else "incomplete")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    run_volume.commit()
    if code:
        raise RuntimeError(f"Vial BC exited{code}; evidence at{RUN_VOLUME}/{request['run_id']}")
    return result


@app.local_entrypoint()
def main(dataset: str = "data/teleop/weigh_vial_grok_v2.hdf5", run_id: str = "smolvla-vial-bc7-001",
         steps: int = 2000, batch_size: int = 8, max_minutes: int = 30, save_every: int = 500,
         seed: int = 0, split_json: str = ""):
    """Data must already exist in the verified ish70 volume; uploads are separate."""
    verify_workspace()
    path = Path(dataset).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    from chemlab_rl.vial_vla_data import file_sha256, prepare_manifest
    request = dict(dataset=path.name, dataset_sha256=file_sha256(path), run_id=run_id, steps=steps,
                   batch_size=batch_size, max_minutes=max_minutes, save_every=save_every, seed=seed)
    if split_json:
        request["split"] = json.loads(Path(split_json).read_text())
    validate_request(request)
    _require_remote_dataset(path.name)
    local = ROOT / "docs/pilot" / run_id
    local.mkdir(parents=True, exist_ok=False)
    # Full schema/image/quality audit precedes the GPU call.
    manifest = prepare_manifest(path, local, **(request.get("split") or {}))
    if manifest["dataset_sha256"] != request["dataset_sha256"]:
        raise ValueError("Dataset changed during local preflight")
    started = time.time()
    deadline = started + max_minutes * 60
    (local / "wrapper_launch.json").write_text(json.dumps(dict(**request, deadline_epoch=deadline), indent=2) + "\n")
    print(json.dumps(dict(event="vial_bc_preflight", **request, deadline_epoch=deadline)), flush=True)
    try:
        result = execute.remote(request, deadline)
        (local / "wrapper_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except Exception as error:
        (local / "wrapper_failure.json").write_text(json.dumps(dict(error_type=type(error).__name__, retries=0,
            elapsed_seconds=time.time() - started, remote_artifacts=f"{RUN_VOLUME}/{run_id}"), indent=2) + "\n")
        raise
