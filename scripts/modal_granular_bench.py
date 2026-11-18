"""Single-container, 15-minute-maximum GPU physics benchmark on ish70."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/chemlab"
app = modal.App("chemlab-granular-benchmark")
runs = modal.Volume.from_name("chemlab-rl-runs", create_if_missing=True, version=2)


def exclude(path: Path) -> bool:
    return any(p in {".git", ".venv", "__pycache__", ".pytest_cache", "work", "recordings"}
               for p in path.parts) or path.suffix in {".pyc", ".log"}


image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libegl1")
         .pip_install("mujoco==3.16.0", "mujoco-warp==3.16.0", "warp-lang==1.15.0", "numpy>=2,<2.3", "trimesh==5.1.1", "gymnasium==1.4.0")
         .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MUJOCO_GL": "egl",
               "PYTHONPATH": REMOTE_ROOT + ":" + REMOTE_ROOT + "/chem_rl_sim:" + REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim"})
         .add_local_file(ROOT / "scripts/benchmark_granular_warp.py", REMOTE_ROOT + "/scripts/benchmark_granular_warp.py")
         .add_local_dir(ROOT / "chem_rl_sim/chem_rl_sim", REMOTE_ROOT + "/chem_rl_sim/chem_rl_sim", ignore=exclude)
         .add_local_dir(ROOT / "chem_rl_sim/third_party/urgantry_sim", REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim", ignore=exclude))


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=900,
              startup_timeout=120, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/runs": runs})
def execute(run_id: str, deadline: float) -> dict:
    import subprocess
    import sys
    import threading
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("Invalid run identifier")
    remaining = min(880, int(deadline - time.time()) - 15)
    if remaining < 60:
        raise TimeoutError("Benchmark deadline expired before startup")
    out = Path("/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, REMOTE_ROOT + "/scripts/benchmark_granular_warp.py",
               "--output", str(out / "benchmark.json"), "--max-seconds", str(remaining)]
    metadata = {"run_id": run_id, "started_epoch": time.time(), "deadline_epoch": deadline,
                "command": command, "gpu": "A100-80GB", "max_containers": 1, "retries": 0}
    (out / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=15)
        metadata["hardware"] = hardware.stdout.strip()
    except Exception as exc:
        metadata["hardware_query_error"] = type(exc).__name__
    print(json.dumps({"event": "benchmark_launch", **metadata}), flush=True)
    runs.commit()
    proc = subprocess.Popen(command, cwd=REMOTE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    timed_out = threading.Event()
    def terminate():
        import signal
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    timer = threading.Timer(max(1, deadline - time.time() - 5), terminate)
    timer.daemon = True
    timer.start()
    try:
        with (out / "console.log").open("a", buffering=1) as stream:
            for line in proc.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                if '"event": "gpu_preparing"' in line or '"event": "gpu_stage"' in line:
                    runs.commit()
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            terminate()
        runs.commit()
    result = {**metadata, "exit_code": code, "timed_out": timed_out.is_set(),
              "elapsed_seconds": time.time() - metadata["started_epoch"]}
    if (out / "benchmark.json").is_file():
        result["benchmark"] = json.loads((out / "benchmark.json").read_text())
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    runs.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "granular-warp-benchmark-001", output: str = "/tmp/chemlab-granular-warp-001.json"):
    import subprocess
    import sys
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        raise RuntimeError("Use authenticated config profile without token environment overrides")
    identity = subprocess.run([sys.executable, "-m", "modal", "profile", "list", "--json"],
                              check=True, capture_output=True, text=True, timeout=30)
    selected = os.environ.get("MODAL_PROFILE")
    profiles = json.loads(identity.stdout)
    active = [p for p in profiles if p["name"] == selected] if selected else [p for p in profiles if p["active"]]
    if len(active) != 1 or active[0]["workspace"] != "ish70":
        raise RuntimeError("Refusing GPU compute outside authenticated ish70 workspace")
    print(json.dumps({"event": "workspace_verified", "workspace": "ish70", "max_seconds": 900}), flush=True)
    result = execute.remote(run_id, time.time() + 900)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "benchmark_saved", "local": str(path),
                      "volume": "chemlab-rl-runs/" + run_id,
                      "exit_code": result["exit_code"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Launch through modal run scripts/modal_granular_bench.py")
