"""Bounded CPU collection of physical-grain robot demonstrations on ish70."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/chemlab"
MAX_COLLECTION_CONTAINERS = 30
CPU_PER_CONTAINER = 2
MEMORY_MIB_PER_CONTAINER = 12288
MERGE_MAX_SECONDS = 3600
app = modal.App("chemlab-physical-demo-collection")
runs = modal.Volume.from_name("chemlab-rl-runs", create_if_missing=True, version=2)
datasets = modal.Volume.from_name("chemlab-rl-data", create_if_missing=True, version=2)


def exclude(path):
    return any(p in {".git", ".venv", "__pycache__", ".pytest_cache", "work", "recordings"}
               for p in path.parts) or path.suffix in {".pyc", ".log"}


image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libegl1", "libosmesa6")
         .pip_install("mujoco==3.16.0", "numpy>=2,<2.3", "trimesh==5.1.1", "gymnasium==1.4.0",
                      "h5py>=3.11", "pillow", "imageio", "pytest")
         .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MUJOCO_GL": "osmesa",
               "PYTHONPATH": REMOTE_ROOT + ":" + REMOTE_ROOT + "/chem_rl_sim:" + REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim"})
         .add_local_dir(ROOT / "chemlab_rl", REMOTE_ROOT + "/chemlab_rl", ignore=exclude)
         .add_local_dir(ROOT / "scripts", REMOTE_ROOT + "/scripts", ignore=exclude)
         .add_local_dir(ROOT / "chem_rl_sim/chem_rl_sim", REMOTE_ROOT + "/chem_rl_sim/chem_rl_sim", ignore=exclude)
         .add_local_dir(ROOT / "chem_rl_sim/third_party/urgantry_sim", REMOTE_ROOT + "/chem_rl_sim/third_party/urgantry_sim", ignore=exclude))


@app.function(image=image, cpu=(CPU_PER_CONTAINER, CPU_PER_CONTAINER),
              memory=(MEMORY_MIB_PER_CONTAINER, MEMORY_MIB_PER_CONTAINER),
              timeout=7200, startup_timeout=300,
              max_containers=MAX_COLLECTION_CONTAINERS, retries=0, scaledown_window=2,
              volumes={"/runs": runs, "/datasets": datasets})
def collect(run_id, powder, seed, target, grains, diagnostic, approach_only, deadline, max_job_seconds=7200,
            expected_source_version=None, merge_run_ids=None, merge_expected_episodes=None):
    import subprocess
    import sys
    import threading
    import signal
    # Merges share this function's bounded pool. A separate merge
    # function would have its own independent autoscaling limit.
    if merge_run_ids is not None:
        return merge_accepted(run_id, merge_run_ids, merge_expected_episodes, deadline)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("Invalid run id")
    if not 60 <= max_job_seconds <= 7200:
        raise ValueError("Each collection job is capped at 120 minutes")
    if deadline - time.time() < 60:
        raise TimeoutError("Collection deadline already expired")
    from scripts.collect_granular_demos import source_identity
    source_version, source_files = source_identity()
    if expected_source_version is not None and source_version != expected_source_version:
        raise ValueError("Remote collector source differs from the validated reference")
    directory = Path("/runs") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    output = Path("/datasets") / f"{run_id}.hdf5"
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable, "-m", "scripts.collect_granular_demos", "--out", str(output),
               "--powder", str(powder), "--seed", str(seed), "--target", str(target), "--grains", str(grains)]
    if diagnostic:
        command.append("--diagnostic")
    if approach_only:
        command.append("--approach-only")
    started = time.time()
    job_deadline = min(deadline, started + max_job_seconds)
    metadata = dict(run_id=run_id, command=command, started_epoch=started, deadline_epoch=deadline,
                    job_deadline_epoch=job_deadline, max_job_seconds=max_job_seconds,
                    cpu=CPU_PER_CONTAINER, memory_mib=MEMORY_MIB_PER_CONTAINER,
                    cpu_limit=CPU_PER_CONTAINER, memory_limit_mib=MEMORY_MIB_PER_CONTAINER,
                    retries=0, gpu=None, dataset_path=str(output),
                    powder_id=powder, seed=seed, target_g=target, grain_count=grains,
                    source_version=source_version, source_files_sha256=source_files,
                    diagnostic=bool(diagnostic), approach_only=bool(approach_only))
    (directory / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    runs.commit()
    proc = subprocess.Popen(command, cwd=REMOTE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    timed_out = threading.Event()
    def terminate():
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    timer = threading.Timer(max(1, job_deadline - time.time() - 15), terminate)
    timer.daemon = True
    timer.start()
    try:
        with (directory / "console.log").open("a", buffering=1) as log:
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
                if not line.startswith(" "):
                    runs.commit()
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            terminate()
        datasets.commit()
    result = {**metadata, "exit_code": code, "elapsed_seconds": time.time() - started,
              "timed_out": timed_out.is_set(), "dataset_written": output.exists()}
    outcome = output.with_suffix(".outcome.json")
    if outcome.exists():
        result["outcome"] = json.loads(outcome.read_text())
    result["accepted"] = False
    if code == 0 and not timed_out.is_set() and output.exists() and not diagnostic and not approach_only:
        try:
            import h5py
            from scripts.merge_granular_demos import _metadata, _validate_episode
            from chemlab_rl.vla_bc import inspect_dataset
            audit = inspect_dataset(output, (5,))
            if audit["num_episodes"] != 1:
                raise ValueError("A collection job must produce exactly one episode")
            with h5py.File(output, "r") as handle:
                identity, fingerprint = _validate_episode(handle["data"][audit["episodes"][0]["name"]], _metadata(handle))
            result.update(accepted=True, episode_identity=list(identity), trajectory_sha256=fingerprint,
                          validation="physical_grain_episode_audit_v1")
        except Exception as error:
            result["validation_error"] = f"{type(error).__name__}: {error}"
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    runs.commit()
    return result


@app.local_entrypoint()
def main(run_id: str, powder: int = 0, seed: int = 0, target: float = 0.28,
         grains: int = 400, diagnostic: bool = True, approach_only: bool = False,
         max_minutes: int = 120):
    from scripts.modal_train import verify_workspace
    verify_workspace()
    if not 5 <= max_minutes <= 120 or not 0 <= powder <= 5 or not 1 <= grains <= 400:
        raise ValueError("Collection parameters outside bounded range")
    result = collect.remote(run_id, powder, seed, target, grains, diagnostic, approach_only,
                            time.time() + 60 * max_minutes)
    local = ROOT / "docs/pilot/physics_validation" / f"{run_id}.remote.json"
    local.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def collection_plan(batch_id, count=5, seed_base=1001, grains=200):
    """Six per material for 30 demos; first five are the 0.5-g pilot."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}", batch_id):
        raise ValueError("Batch id must be a safe identifier of at most 60 characters")
    if (type(count) is not int or count not in (5, 30) or type(grains) is not int or grains != 200
            or type(seed_base) is not int or not 1 <= seed_base <= 2**31 - count):
        raise ValueError("Batch protocol requires count 5 or 30, 200 grains, and unique positive integer seeds")
    targets = [.5, .3, .5, .7, .3, .7]
    return [dict(index=index, run_id=f"{batch_id}-e{index:02d}-p{index % 5}-s{seed_base + index}",
                 powder=index % 5, seed=seed_base + index, target=targets[index // 5], grains=grains,
                 reset_variation="seeded_initial_grain_lattice_rotation", status="planned")
            for index in range(count)]


def _source_identity():
    """Mirror the collector's pure file inventory without simulator imports.

    The offline regression checks parity with source_identity in the collector.
    Keep both inventories in sync whenever a new physics dependency is added.
    """
    names = ["chemlab_rl/scrape_env.py", "chemlab_rl/weigh_scene.py",
             "chemlab_rl/granular_env.py", "chemlab_rl/granular_scene.py",
             "chemlab_rl/granular_material.py", "chemlab_rl/granular_geometry.py",
             "chemlab_rl/granular_accounting.py", "scripts/collect_leo_scrape.py",
             "scripts/collect_scrape_v2.py", "scripts/collect_granular_demos.py"]
    paths = [ROOT / name for name in names]
    for directory in (ROOT / "chem_rl_sim/chem_rl_sim", ROOT / "chem_rl_sim/third_party/urgantry_sim/urgantry_sim"):
        paths.extend(path for path in directory.rglob("*") if path.is_file()
                     and path.suffix.lower() in {".py", ".xml", ".obj", ".stl", ".png", ".jpg", ".json"}
                     and "__pycache__" not in path.parts)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(set(paths))}
    version = "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return version, hashes


def validate_reference_result(path, source_path=None):
    """Enforce the explicit first-valid-robot-expert gate before any spawn."""
    path = Path(path)
    payload = json.loads(path.read_text())
    outcome = payload.get("outcome", payload)
    if payload.get("exit_code", 0) != 0 or payload.get("timed_out", False):
        raise ValueError("Reference robot run failed or timed out")
    if (outcome.get("success") is not True or outcome.get("physical_valid") is not True
            or outcome.get("mass_force_agreement") is not True or outcome.get("failure")
            or outcome.get("grain_count") != 200
            or outcome.get("powder_dynamics") != "physical_grains_v1"
            or outcome.get("physical_validation_scope") != "rigid_bodies_and_grains"
            or outcome.get("physics_checks", 0) <= 0):
        raise ValueError("Batch collection requires a successful physically validated 200-grain robot expert")
    required = ["mass_g", "raw_net_mass_g", "conservation_error_g", "grain_radius_m",
                "max_grain_contact_penetration_m", "spilled_g", "target_g", "balance_g"]
    if any(not isinstance(outcome.get(key), (float, int)) or not math.isfinite(outcome[key]) for key in required):
        raise ValueError("Reference robot outcome lacks finite mass/contact evidence")
    if (abs(outcome["conservation_error_g"]) > 1e-8 or outcome["grain_radius_m"] <= 0
            or not 0 <= outcome["max_grain_contact_penetration_m"] <= .2 * outcome["grain_radius_m"] + 1e-12
            or abs(outcome["raw_net_mass_g"] - outcome["mass_g"]) > .010 + 1e-12
            or not 0 <= outcome["spilled_g"] <= .05 + 1e-9
            or abs(outcome["mass_g"] - outcome["target_g"]) > .05 + 1e-9
            or abs(outcome["balance_g"] - outcome["target_g"]) > .05 + 1e-9):
        raise ValueError("Reference robot outcome failed its numeric acceptance limits")
    provenance = outcome if "source_version" in outcome else payload
    evidence = None
    if source_path:
        source_path = Path(source_path)
        external = json.loads(source_path.read_text())
        if provenance.get("source_version") and provenance["source_version"] != external.get("source_version"):
            raise ValueError("External source evidence contradicts the reference result's source version")
        provenance = external
        if not payload.get("run_id") or provenance.get("run_id") != payload["run_id"]:
            raise ValueError("External source evidence must identify the exact reference run_id")
        evidence = {"path": str(source_path.resolve()),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "capture_method": provenance.get("capture_method", provenance.get("capture_note")),
                    "remote_source_verified": provenance.get("remote_source_verified", False),
                    "modal_app_id": provenance.get("modal_app_id")}
    version, hashes = provenance.get("source_version"), provenance.get("source_files_sha256")
    if isinstance(hashes, str):
        hashes = json.loads(hashes)
    if (not isinstance(hashes, dict) or not hashes or not isinstance(version, str)
            or version != "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()):
        raise ValueError("Reference lacks a consistent source version and file-hash manifest")
    current_version, current_hashes = _source_identity()
    if current_version != version or current_hashes != hashes:
        raise ValueError("Collector/physics source has changed since the validated reference; revalidate before collection")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reported_success": True, "grain_count": 200, "source_version": version,
            "source_files_sha256": hashes, "external_source_evidence": evidence}


def _checked_repo_file(value, digest, label):
    """Read immutable evidence inside this checkout, rejecting escaping links."""
    if not isinstance(value, str) or not value or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Missing declared path/SHA for {label}")
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"{label} must be a file inside the repository")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{label} SHA does not match its immutable file")
    return path, raw


def _validate_initial_state_provenance(payload):
    """Verify the actual regrasp suffix state separately from failure motivation."""
    import numpy as np
    evidence = payload.get("initial_state_provenance")
    if not isinstance(evidence, dict):
        raise ValueError("Continuation must declare its actual initial-state provenance")
    files = {}
    for key in ("snapshot", "outcome", "manifest", "ancestor_snapshot", "ancestor_result", "ancestor_source_manifest"):
        files[key] = _checked_repo_file(evidence.get(f"{key}_path"), evidence.get(f"{key}_sha256"), key)
    snapshot, _ = files["snapshot"]
    initial = json.loads(files["outcome"][1])
    manifest_path, manifest_raw = files["manifest"]
    inventory = json.loads(manifest_raw)
    if not isinstance(inventory, dict):
        raise ValueError("Invalid initial-artifact manifest")
    for key in ("snapshot", "outcome"):
        relative = str(files[key][0].relative_to(manifest_path.parent))
        if inventory.get(relative) != evidence[f"{key}_sha256"]:
            raise ValueError("Initial snapshot/outcome differs from its recorded artifact manifest")
    for name in ("fixture.mjb", "physical_probe.py"):
        _checked_repo_file(str(manifest_path.parent / name), inventory.get(name), f"initial artifact {name}")
    ancestor = json.loads(files["ancestor_result"][1])
    ancestor_sources = json.loads(files["ancestor_source_manifest"][1])
    ancestor_hashes = ancestor_sources.get("source_files_sha256")
    if (not isinstance(ancestor_hashes, dict) or not ancestor_hashes
            or ancestor_sources.get("source_version") != "sha256:" + hashlib.sha256(json.dumps(ancestor_hashes, sort_keys=True).encode()).hexdigest()
            or not ancestor.get("run_id") or ancestor_sources.get("run_id") != ancestor["run_id"]
            or ancestor.get("accepted") is not False or ancestor.get("timed_out") is not False
            or ancestor.get("exit_code") != 1
            or initial.get("source_snapshot_sha256") != evidence["ancestor_snapshot_sha256"]
            or initial.get("source_snapshot_unchanged") is not True):
        raise ValueError("Initial-state ancestor chain is inconsistent")
    ancestor_snapshot = files["ancestor_snapshot"][0]
    if ancestor_snapshot.name != ancestor["run_id"] + ".final.npz":
        raise ValueError("Ancestor snapshot does not identify its declared failed run")
    current_hashes = payload["source_files_sha256"]
    allowed_controller_changes = {"scripts/collect_granular_demos.py", "scripts/collect_scrape_v2.py"}
    changed_files = {name for name in ancestor_hashes if ancestor_hashes[name] != current_hashes.get(name)}
    if set(ancestor_hashes) != set(current_hashes) or not changed_files <= allowed_controller_changes:
        raise ValueError("Initial-state continuation must preserve native physics/assets and change only the declared controllers")
    checks = evidence.get("checks", {})
    if (not isinstance(checks, dict)
            or any(checks.get(key) is not True for key in ("source_held", "tool_on_bench", "finite_state", "physics_valid"))
            or checks.get("right_tool_held") is not False
            or type(checks.get("bench_support_gap_m")) not in (int, float)
            or not math.isfinite(checks["bench_support_gap_m"]) or abs(checks["bench_support_gap_m"]) > .001):
        raise ValueError("Initial state must have a physically held source and supported ungrasped tool")
    with np.load(snapshot, allow_pickle=False) as state, np.load(ancestor_snapshot, allow_pickle=False) as ancestor_state:
        for key in ("qpos", "qvel", "ctrl"):
            if (key not in state or key not in ancestor_state or state[key].ndim != 1 or not state[key].size
                    or state[key].shape != ancestor_state[key].shape
                    or not np.isfinite(state[key]).all() or not np.isfinite(ancestor_state[key]).all()):
                raise ValueError("Initial simulator snapshot must contain finite complete state arrays")
        if state["ctrl"].shape != (14,) or state["ctrl"][6] < 200 or state["ctrl"][13] > 50:
            raise ValueError("Initial snapshot must retain the left grasp and keep the right gripper open")
    numeric = ("grain_count", "grain_radius_m", "physics_checks", "max_grain_contact_penetration_m",
               "conservation_error_g", "spilled_g", "source_g", "tool_g", "mass_g", "inflight_g", "total_mass_g")
    if any(type(initial.get(key)) not in (int, float) or not math.isfinite(initial[key]) for key in numeric):
        raise ValueError("Initial-state outcome lacks finite physical inventory/contact evidence")
    if (initial.get("continuation_passed") is not True or initial.get("physical_valid") is not True
            or initial.get("success") is not False or initial.get("failure")
            or initial.get("physical_validation_scope") != "rigid_bodies_and_grains"
            or initial.get("powder_dynamics") != "physical_grains_v1"
            or initial.get("no_state_writes_after_restore") is not True
            or not isinstance(initial.get("scope"), str) or "not a from-reset" not in initial["scope"]
            or initial["grain_count"] != 200 or initial["physics_checks"] <= 0
            or initial["grain_radius_m"] <= 0
            or not 0 <= initial["max_grain_contact_penetration_m"] <= .2 * initial["grain_radius_m"] + 1e-12
            or abs(initial["conservation_error_g"]) > 1e-8
            or any(abs(initial[key]) > 1e-12 for key in ("spilled_g", "tool_g", "mass_g", "inflight_g"))
            or initial["source_g"] <= 0 or abs(initial["source_g"] - initial["total_mass_g"]) > 1e-8
            or any(payload.get(key) != initial.get(key) for key in ("powder_id", "seed", "target_g", "grain_count"))):
        raise ValueError("Declared initial state is not the physically valid source-held diagnostic suffix")
    stages = {row.get("stage"): row for row in initial.get("stages", []) if isinstance(row, dict)}
    try:
        held, lifted = stages["regrasped"], stages["lifted_regrasped_source"]
        source_delta = np.asarray(lifted["source"], float) - np.asarray(held["source"], float)
        tool_delta = np.asarray(lifted["left_tool"], float) - np.asarray(held["left_tool"], float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Initial source hold lacks the recorded physical lift evidence") from error
    if (source_delta.shape != (3,) or tool_delta.shape != (3,) or not np.isfinite(source_delta).all()
            or not np.isfinite(tool_delta).all() or source_delta[2] < .05
            or np.linalg.norm(source_delta - tool_delta) > .005 or held.get("failure") or lifted.get("failure")):
        raise ValueError("Recorded source did not follow the physical left-gripper lift")
    motivation = payload.get("motivation", {})
    motivation_path, motivation_raw = _checked_repo_file(motivation.get("result_path"), motivation.get("result_sha256"), "motivation")
    failed = json.loads(motivation_raw)
    if (failed.get("run_id") != motivation.get("run_id") or failed.get("accepted") is not False
            or failed.get("exit_code") != 1 or failed.get("timed_out") is not False
            or not str(failed.get("outcome", {}).get("failure", "")).startswith("loading depth miss:")):
        raise ValueError("Motivating failed scoop run must remain separate and accurately identified")
    return {**evidence, "initial_scope": initial["scope"], "initial_seed": initial.get("seed"),
            "ancestor_run_id": ancestor["run_id"], "ancestor_source_version": ancestor_sources["source_version"],
            "ancestor_remote_source_verified": ancestor_sources.get("remote_source_verified", False),
            "controller_files_changed_from_ancestor": sorted(changed_files),
            "ancestor_source_capture_note": ancestor_sources.get("capture_note"),
            "motivation": {**motivation, "result_path": str(motivation_path), "source_version": failed.get("source_version")}}


def _verify_prefix_controller_compatibility(old_source, new_source):
    """Prove this specific refactor leaves every successful old prefix unchanged.

    This is deliberately structural rather than a general equivalence claim.
    Unknown controller edits fail closed and require a newly executed prefix.
    """
    import ast
    import copy

    def dump(node):
        return ast.dump(node, include_attributes=False)

    old, new = ast.parse(old_source), ast.parse(new_source)
    allowed_functions = {"_move_cup", "_pour_complete_batch", "_tilt_complete_batch"}
    allowed_constants = {"EXPERT_CONTROLLER", "POUR_ANGLE_DEG", "POUR_PIVOT_HEIGHT_M"}

    def unchanged_nodes(tree):
        return [dump(node) for node in tree.body
                if not (isinstance(node, ast.FunctionDef) and node.name in allowed_functions)
                and not (isinstance(node, ast.Assign) and len(node.targets) == 1
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id in allowed_constants)]

    if unchanged_nodes(old) != unchanged_nodes(new):
        raise ValueError("Chained prefix requires all non-pour controller behavior to remain unchanged")
    functions = [{n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in (old, new)]
    if "_move_cup" not in functions[0] or "_move_cup" not in functions[1] or "_load_batch" not in functions[0]:
        raise ValueError("Missing production prefix controller functions")
    original, current = functions[0]["_move_cup"], functions[1]["_move_cup"]
    if dump(original) == dump(current):
        return "identical_prefix_controller"
    expected = copy.deepcopy(original)
    try:
        index = next(i for i, n in enumerate(expected.body) if isinstance(n, ast.For))
        loop = expected.body[index]
        split = next(i for i, n in enumerate(loop.body)
                     if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "rot")
        if dump(loop.body[split]) != dump(ast.parse("rot = mat.reshape(3, 3)").body[0]):
            raise ValueError("Unexpected prefix interpolation")
        if dump(loop.body[-1]) != dump(ast.parse("if env.scene.bad:\n    return False").body[0]):
            raise ValueError("Unexpected prefix failure guard")
        helper = ast.parse("def command_pose(tip, rot):\n    pass").body[0]
        helper.body = loop.body[split + 1:-1] + ast.parse("return not env.scene.bad").body
        loop.body = loop.body[:split] + ast.parse("if not command_pose(tip, mat.reshape(3, 3)):\n    return False").body
        tail = expected.body[index + 1:]
        guard = ast.parse('if error > 0.004:\n    env.scene.bad = f"scoop tracking error {error * 1000:.2f} mm"\n    return False\nreturn True').body
        if [dump(n) for n in tail[1:]] != [dump(n) for n in guard]:
            raise ValueError("Old successful prefix does not prove the unchanged 4 mm endpoint guard")
        retry = ast.parse("if error > 0.004:\n    for _ in range(5):\n        if not command_pose(target, rotation):\n            return False\n        error = float(np.linalg.norm(legacy._tip(env.data) - target))\n        if error <= 0.001:\n            break").body
        expected.body = expected.body[:index] + [helper, loop, tail[0]] + retry + tail[1:]
    except (StopIteration, IndexError) as error:
        raise ValueError("Unrecognized prefix servo structure") from error
    if dump(expected) != dump(current):
        raise ValueError("Shared servo change is not the verified normal-path refactor with inactive prefix correction")
    return "normal_path_ast_equal_old_success_implies_retry_inactive"


def _validate_effective_physics(raw, record, manifest, outcome):
    """Compare actual runtime MJB captures, not inferred historical settings.

    MuJoCo serializes the compiled model including its physics options. Only
    geom_rgba is normalized on a copy because scene.show changes display alpha.
    Initial/final captures detect persistent probe/helper overrides; they are
    bounded audit evidence, not a sandbox against arbitrary transient Python.
    """
    evidence = json.loads(raw)
    if (evidence.get("schema_version") != 1 or evidence.get("capture_scope") != "actual_segment_runtime"
            or evidence.get("serialization") != "mujoco_mjb_copy_zero_geom_rgba_v1"
            or evidence.get("excluded_model_fields") != ["geom_rgba"]
            or evidence.get("mujoco_version") != "3.16.0"
            or evidence.get("source_version") != manifest["source_version"]
            or evidence.get("probe_sha256") != record["probe_sha256"]
            or outcome.get("effective_physics_sha256") != record["effective_physics_sha256"]):
        raise ValueError("Segment needs actual source-bound runtime model captures with only visual-color normalization")
    phases = {"expected_production": "fresh_production_before_overrides",
              "effective_initial": "after_initialization_and_overrides_before_first_step",
              "effective_final": "after_last_step_before_cleanup"}
    captures = evidence.get("captures")
    if not isinstance(captures, dict) or set(captures) != set(phases):
        raise ValueError("Missing expected, initial or final effective physics capture")
    models = {}
    for name, phase in phases.items():
        capture = captures[name]
        if not isinstance(capture, dict) or capture.get("capture_phase") != phase:
            raise ValueError("Model must be captured at its actual declared execution phase")
        path, content = _checked_repo_file(capture.get("model_path"), capture.get("model_sha256"), name + " effective model")
        if not content or path.suffix != ".mjb":
            raise ValueError("Effective physics capture must preserve the serialized native model")
        models[name] = content
    if any(content != models["expected_production"] for content in models.values()):
        raise ValueError("Runtime physics/model parameters differ from the expected production model")
    return captures["expected_production"]["model_sha256"]


def _validate_continuation_chain(payload):
    """Validate two saved-state segments without promoting either to episode data."""
    import io
    import tarfile
    import numpy as np

    chain = payload.get("continuation_chain")
    if not isinstance(chain, dict) or chain.get("schema_version") != 1:
        raise ValueError("Missing versioned continuation chain")
    current_hashes = payload["source_files_sha256"]
    segments = {}
    for role in ("prefix", "suffix"):
        record = chain.get(role)
        if not isinstance(record, dict):
            raise ValueError("Continuation chain must contain prefix and suffix records")
        files = {key: _checked_repo_file(record.get(key + "_path"), record.get(key + "_sha256"), role + " " + key)
                 for key in ("source_manifest", "source_archive", "probe", "outcome", "input_snapshot", "output_snapshot", "effective_physics")}
        manifest = json.loads(files["source_manifest"][1])
        hashes = manifest.get("source_files_sha256")
        if (not isinstance(hashes, dict) or set(hashes) != set(current_hashes)
                or manifest.get("source_version") != "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                or manifest.get("probe_sha256") != record["probe_sha256"]):
            raise ValueError("Segment source/probe manifest is incomplete or inconsistent")
        changed = {name for name in hashes if hashes[name] != current_hashes[name]}
        if changed - {"scripts/collect_granular_demos.py"}:
            raise ValueError("Chained segments must share identical physics, assets, pickup and native dependencies")
        if role == "suffix" and hashes != current_hashes:
            raise ValueError("Final continuation must execute the current complete source manifest")
        archived = {}
        with tarfile.open(fileobj=io.BytesIO(files["source_archive"][1]), mode="r:gz") as archive:
            for member in archive.getmembers():
                if (not member.isfile() or member.name in archived or member.name not in hashes
                        or not member.name.endswith(".py") or member.size > 4 * 1024 * 1024):
                    raise ValueError("Source archive contains duplicate, unsupported or untracked entries")
                content = archive.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != hashes[member.name]:
                    raise ValueError("Archived source differs from the executed source manifest")
                archived[member.name] = content
        if set(archived) != {name for name in hashes if name.endswith(".py")}:
            raise ValueError("Source archive must preserve every covered Python file")
        outcome = json.loads(files["outcome"][1])
        if (outcome.get("source_version") != manifest["source_version"]
                or outcome.get("snapshot_sha256") != record["input_snapshot_sha256"]
                or outcome.get("snapshot_unchanged") is not True
                or outcome.get("no_state_writes_after_initialization") is not True
                or outcome.get("initialization_contact_model_override") is not None
                or any(outcome.get(key) != payload.get(key) for key in ("grain_count", "powder_id", "seed", "target_g"))
                or outcome.get("physical_validation_scope") != "rigid_bodies_and_grains"
                or outcome.get("powder_dynamics") != "physical_grains_v1"):
            raise ValueError("Segment outcome does not match its executed source, input state or physical scope")
        model_sha = _validate_effective_physics(files["effective_physics"][1], record, manifest, outcome)
        if outcome.get("checkpoint_sha256", {}).get(files["output_snapshot"][0].name) != record["output_snapshot_sha256"]:
            raise ValueError("Executed segment does not bind its selected output checkpoint")
        for key in ("physics_checks", "grain_radius_m", "grain_mass_g", "timestep_s", "max_contact_penetration_m",
                    "max_grain_contact_penetration_m", "conservation_error_g", "total_mass_g"):
            if type(outcome.get(key)) not in (int, float) or not math.isfinite(outcome[key]):
                raise ValueError("Segment lacks finite cumulative physical evidence")
        if (type(outcome["physics_checks"]) is not int or outcome["physics_checks"] <= 0
                or outcome["grain_radius_m"] <= 0 or outcome["grain_mass_g"] <= 0 or outcome["timestep_s"] != .00025
                or outcome.get("physics_validation") != "rigid_contact_v1"
                or not 0 <= outcome["max_contact_penetration_m"] <= .001 + 1e-12
                or not 0 <= outcome["max_grain_contact_penetration_m"] <= .2 * outcome["grain_radius_m"] + 1e-12
                or abs(outcome["conservation_error_g"]) > 1e-8):
            raise ValueError("Segment cumulative contact or conservation audit failed")
        segments[role] = dict(record=record, files=files, manifest=manifest, outcome=outcome, archived=archived, model_sha=model_sha)
    prefix, suffix = segments["prefix"], segments["suffix"]
    if prefix["model_sha"] != suffix["model_sha"]:
        raise ValueError("Chained segments used different effective native physics models")
    p, s = prefix["outcome"], suffix["outcome"]
    initial_sha = payload["initial_state_provenance"]["snapshot_sha256"]
    boundary_sha = prefix["record"]["output_snapshot_sha256"]
    if (prefix["record"]["input_snapshot_sha256"] != initial_sha
            or boundary_sha != suffix["record"]["input_snapshot_sha256"]
            or prefix["files"]["output_snapshot"][1] != suffix["files"]["input_snapshot"][1]):
        raise ValueError("Continuation segments are not connected by the exact initial and boundary state bytes")
    if (p.get("variant") != "pickup_suffix" or p.get("correction_passed") is not True
            or p.get("lift_passed") is not True or p.get("pour_passed") is not False
            or p.get("continuation_passed") is not False or p.get("physical_valid") is not False
            or not str(p.get("failure", "")).startswith("scoop IK miss ")):
        raise ValueError("Prefix must retain its original later IK failure and verified loading/lift scope")
    stages = p.get("stages", [])
    indices = [i for i, row in enumerate(stages) if row.get("stage") == "loaded_and_lifted"]
    if len(indices) != 1:
        raise ValueError("Prefix needs one unambiguous loaded-and-lifted cut")
    index = indices[0]
    cut = stages[index]
    if (any(type(cut.get(key)) not in (int, float) or not math.isfinite(cut[key])
            for key in ("source_g", "tool_g", "receiver_g", "inflight_g", "spill_g", "sim_seconds"))
            or any(row.get("failure") for row in stages[:index + 1])
            or not any(row.get("failure") == p["failure"] for row in stages[index + 1:])
            or cut.get("failure") or cut.get("receiver_g") != 0 or cut.get("inflight_g") != 0
            or cut.get("spill_g") != 0 or not isinstance(cut.get("tool_g"), (int, float)) or cut["tool_g"] <= .05):
        raise ValueError("Prefix failure precedes its cut or its retained batch is not physically valid")
    trace = p.get("expert_trace", [])
    retained = [row for row in trace if row.get("phase") == "retained_batch" and row.get("batch_fits") is True]
    with np.load(prefix["files"]["output_snapshot"][0], allow_pickle=False) as state:
        required = {"qpos", "qvel", "ctrl", "qacc_warmstart", "integration_state", "integration_state_spec",
                    "time", "q_cmd_left", "q_cmd_right", "grip_cmd_left", "grip_cmd_right"}
        if (not required <= set(state.files) or any(not np.isfinite(state[key]).all() for key in required)
                or state["ctrl"].shape != (14,) or state["q_cmd_left"].shape != (6,) or state["q_cmd_right"].shape != (6,)
                or int(state["integration_state_spec"]) != 16383):
            raise ValueError("Boundary must contain finite complete integration and controller state")
        boundary_time = float(state["time"])
        boundary_shapes = {key: state[key].shape for key in required}
    with np.load(prefix["files"]["input_snapshot"][0], allow_pickle=False) as state:
        if "time" not in state or not math.isfinite(float(state["time"])) or abs(boundary_time - float(state["time"]) - cut["sim_seconds"]) > 1e-8:
            raise ValueError("Prefix cut time differs from its recorded integration-state interval")
        if any(state[key].shape != boundary_shapes[key] for key in ("qpos", "qvel", "ctrl")):
            raise ValueError("Boundary state dimensions differ from the preserved native model")
    matching = [row for row in retained if abs(row.get("sim_seconds", -1) - boundary_time) <= 1e-9]
    if len(matching) != 1 or abs(matching[0]["tool_g"] - cut["tool_g"]) > 1e-9:
        raise ValueError("Boundary snapshot is not the physically accepted retained-batch endpoint")
    depths = [row for row in trace if row.get("phase") == "depth_check"
              and row.get("sim_seconds", math.inf) < boundary_time
              and abs(row.get("z_error_m", math.inf)) <= .00025
              and 0 <= row.get("lateral_error_m", math.inf) <= .001]
    if not depths:
        raise ValueError("Prefix has no executed depth check within unchanged production tolerances")
    depth = depths[-1]
    compatibility = _verify_prefix_controller_compatibility(
        prefix["archived"]["scripts/collect_granular_demos.py"], suffix["archived"]["scripts/collect_granular_demos.py"])
    audit_path, audit_raw = _checked_repo_file(chain.get("boundary_audit_path"), chain.get("boundary_audit_sha256"), "boundary audit")
    audit = json.loads(audit_raw)
    booleans = ("integration_state_restored_exactly", "controller_state_restored_exactly", "finite_state", "source_held",
                "right_tool_held", "receiver_empty_before_tare", "integration_state_size_valid",
                "snapshot_redundant_fields_match_integration_state")
    if (audit.get("snapshot_sha256") != boundary_sha or audit.get("source_version") != payload["source_version"]
            or audit.get("normalized_model_sha256") != suffix["model_sha"]
            or any(audit.get(key) is not True for key in booleans)):
        raise ValueError("Boundary restore, grasp or empty-receiver tare audit failed")
    for key, expected in (("source_g", cut["source_g"]), ("tool_g", cut["tool_g"]), ("receiver_g", 0.),
                          ("inflight_g", 0.), ("spilled_g", 0.), ("total_mass_g", p["total_mass_g"]), ("conservation_error_g", 0.)):
        if type(audit.get(key)) not in (int, float) or not math.isfinite(audit[key]) or abs(audit[key] - expected) > 1e-8:
            raise ValueError("Boundary audit does not preserve the physical retained inventory")
    if (any(s[key] != p[key] for key in ("grain_radius_m", "grain_mass_g", "timestep_s"))
            or s.get("variant") != "receiver_transit_suffix" or s.get("continuation_passed") is not True
            or s.get("physical_valid") is not True or s.get("pour_passed") is not True or s.get("failure")
            or s.get("settled") is not True or s.get("mass_force_agreement") is not True
            or any(type(s.get(key)) not in (int, float) or not math.isfinite(s[key])
                   for key in ("tool_g", "inflight_g", "spilled_g", "mass_g", "raw_net_mass_g", "total_mass_g"))
            or any(abs(s[key]) > 1e-12 for key in ("tool_g", "inflight_g", "spilled_g"))
            or abs(s["mass_g"] - cut["tool_g"]) > 1e-8
            or abs(s["total_mass_g"] - p["total_mass_g"]) > 1e-8
            or abs(s["raw_net_mass_g"] - s["mass_g"]) > .01 + 1e-12):
        raise ValueError("Current suffix did not completely drain the retained batch with settled independent force agreement")
    settled_stages = [row for row in s.get("stages", []) if row.get("stage") == "post_pour_settled" and not row.get("failure")]
    if (len(settled_stages) != 1 or type(settled_stages[0].get("sim_seconds")) not in (int, float)
            or not math.isfinite(settled_stages[0]["sim_seconds"]) or settled_stages[0]["sim_seconds"] <= 0):
        raise ValueError("Missing successful post-pour settled endpoint")
    if suffix["files"]["output_snapshot"][0].name != "post_pour_settled.npz":
        raise ValueError("Suffix output must be the complete post-pour integration/controller snapshot")
    with np.load(suffix["files"]["output_snapshot"][0], allow_pickle=False) as state:
        if (not required <= set(state.files) or any(not np.isfinite(state[key]).all() for key in required)
                or any(state[key].shape != boundary_shapes[key] for key in required)
                or int(state["integration_state_spec"]) != 16383
                or abs(float(state["time"]) - boundary_time - settled_stages[0]["sim_seconds"]) > 1e-8):
            raise ValueError("Final settled snapshot is incomplete, nonfinite or precedes the chain boundary")
    summary = payload["stage_summary"]
    derived = dict(physics_checks=p["physics_checks"] + s["physics_checks"],
                   grain_radius_m=s["grain_radius_m"], max_grain_contact_penetration_m=max(p["max_grain_contact_penetration_m"], s["max_grain_contact_penetration_m"]),
                   conservation_error_g=max(abs(p["conservation_error_g"]), abs(s["conservation_error_g"])),
                   spilled_g=s["spilled_g"], transferred_g=s["mass_g"], mass_g=s["mass_g"], raw_net_mass_g=s["raw_net_mass_g"],
                   depth_error_m=depth["z_error_m"], lateral_error_m=depth["lateral_error_m"])
    if (summary.get("physics_checks_scope") != "prefix_run_including_excluded_tail_plus_suffix_upper_bound"
            or any(type(summary.get(key)) not in (int, float) or not math.isfinite(summary[key])
                   or abs(summary[key] - value) > 1e-12 for key, value in derived.items())):
        raise ValueError("Chain summary must derive from immutable evidence and explicitly label cumulative upper bounds")
    return dict(schema_version=1, prefix=prefix["record"], suffix=suffix["record"],
                boundary_audit_path=str(audit_path), boundary_audit_sha256=chain["boundary_audit_sha256"],
                prefix_cut="loaded_and_lifted", prefix_later_failure_preserved=p["failure"],
                source_compatibility=compatibility, physics_checks_scope=summary["physics_checks_scope"])


def _validate_loaded_transfer_feasibility(payload, path, raw):
    """Schedule fresh candidates from forward-only evidence; validate no prefix."""
    import io
    import tarfile
    import numpy as np
    if (payload.get("historical_prefix_validated") is not False
            or type(payload.get("accepted_demo_count")) is not int or payload["accepted_demo_count"] != 0
            or payload.get("full_episode_accepted") is not False or payload.get("candidate_data_eligible") is not False
            or payload.get("stage_physical_valid") is not True or payload.get("grain_count") != 200
            or payload.get("correction_passed") is True or payload.get("lift_passed") is True
            or payload.get("failure")):
        raise ValueError("Loaded transfer feasibility must claim zero accepted episodes and no validated historical prefix")
    version, hashes = _source_identity()
    if payload.get("source_version") != version or payload.get("source_files_sha256") != hashes:
        raise ValueError("Loaded transfer evidence must match the complete current production source")
    record = payload.get("transfer_segment")
    if not isinstance(record, dict):
        raise ValueError("Missing forward transfer segment")
    files = {key: _checked_repo_file(record.get(key + "_path"), record.get(key + "_sha256"), "transfer " + key)
             for key in ("source_manifest", "source_archive", "probe", "outcome", "input_snapshot", "output_snapshot", "effective_physics", "boundary_audit")}
    manifest = json.loads(files["source_manifest"][1])
    if (manifest.get("source_version") != version or manifest.get("source_files_sha256") != hashes
            or version != "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            or manifest.get("probe_sha256") != record["probe_sha256"]):
        raise ValueError("Transfer source/probe manifest does not match current production")
    archived = set()
    with tarfile.open(fileobj=io.BytesIO(files["source_archive"][1]), mode="r:gz") as archive:
        for member in archive.getmembers():
            if (not member.isfile() or member.name in archived or member.name not in hashes
                    or not member.name.endswith(".py") or member.size > 4 * 1024 * 1024
                    or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != hashes[member.name]):
                raise ValueError("Transfer archive differs from the complete executed Python source")
            archived.add(member.name)
    if archived != {name for name in hashes if name.endswith(".py")}:
        raise ValueError("Transfer source archive is incomplete")
    outcome = json.loads(files["outcome"][1])
    if (outcome.get("source_version") != version or outcome.get("snapshot_sha256") != record["input_snapshot_sha256"]
            or outcome.get("snapshot_unchanged") is not True or outcome.get("no_state_writes_after_initialization") is not True
            or outcome.get("initialization_contact_model_override") is not None
            or outcome.get("variant") != "receiver_transit_suffix"
            or outcome.get("physical_validation_scope") != "rigid_bodies_and_grains"
            or outcome.get("powder_dynamics") != "physical_grains_v1" or outcome.get("physics_validation") != "rigid_contact_v1"
            or any(outcome.get(key) != payload.get(key) for key in ("grain_count", "powder_id", "seed", "target_g"))
            or any(outcome.get(key) is not True for key in ("physical_valid", "continuation_passed", "pour_passed", "settled", "mass_force_agreement"))
            or outcome.get("failure")):
        raise ValueError("Current ordinary-control transfer has not passed its complete physical and settled-pour audit")
    model_sha = _validate_effective_physics(files["effective_physics"][1], record, manifest, outcome)
    audit = json.loads(files["boundary_audit"][1])
    if (audit.get("source_version") != version or audit.get("snapshot_sha256") != record["input_snapshot_sha256"]
            or audit.get("normalized_model_sha256") != model_sha
            or any(audit.get(key) is not True for key in ("integration_state_restored_exactly", "controller_state_restored_exactly",
                    "integration_state_size_valid", "snapshot_redundant_fields_match_integration_state", "finite_state", "source_held",
                    "right_tool_held", "receiver_empty_before_tare"))):
        raise ValueError("Loaded initial state lacks a current native-model restore, grasp and empty-tare audit")
    initial_keys = ("source_g", "tool_g", "receiver_g", "inflight_g", "spilled_g", "total_mass_g", "conservation_error_g")
    final_keys = ("source_g", "tool_g", "mass_g", "inflight_g", "spilled_g", "total_mass_g", "conservation_error_g",
                  "raw_net_mass_g", "physics_checks", "timestep_s", "grain_radius_m", "grain_mass_g", "max_contact_penetration_m", "max_grain_contact_penetration_m")
    if (any(type(audit.get(key)) not in (int, float) or not math.isfinite(audit[key]) for key in initial_keys)
            or any(type(outcome.get(key)) not in (int, float) or not math.isfinite(outcome[key]) for key in final_keys)):
        raise ValueError("Transfer requires finite measured inventory, contact and force evidence")
    if (audit["source_g"] < 0 or audit["tool_g"] <= .050
            or any(abs(audit[key]) > 1e-12 for key in ("receiver_g", "inflight_g", "spilled_g"))
            or abs(audit["source_g"] + audit["tool_g"] - audit["total_mass_g"]) > 1e-8
            or abs(audit["conservation_error_g"]) > 1e-8
            or any(abs(outcome[key]) > 1e-12 for key in ("tool_g", "inflight_g", "spilled_g"))
            or abs(outcome["mass_g"] - audit["tool_g"]) > 1e-8 or abs(outcome["source_g"] - audit["source_g"]) > 1e-8
            or abs(outcome["total_mass_g"] - audit["total_mass_g"]) > 1e-8
            or abs(outcome["source_g"] + outcome["mass_g"] - outcome["total_mass_g"]) > 1e-8
            or abs(outcome["conservation_error_g"]) > 1e-8
            or abs(outcome["raw_net_mass_g"] - outcome["mass_g"]) > .010 + 1e-12
            or type(outcome["physics_checks"]) is not int or outcome["physics_checks"] <= 0
            or outcome["timestep_s"] != .00025 or outcome["grain_radius_m"] <= 0 or outcome["grain_mass_g"] <= 0
            or abs(outcome["grain_mass_g"] * 200 - outcome["total_mass_g"]) > 1e-8
            or not 0 <= outcome["max_contact_penetration_m"] <= .001 + 1e-12
            or not 0 <= outcome["max_grain_contact_penetration_m"] <= .2 * outcome["grain_radius_m"] + 1e-12):
        raise ValueError("Loaded transfer failed its retained-load, complete drain, conservation, contact or force limits")
    required = {"qpos", "qvel", "ctrl", "qacc_warmstart", "integration_state", "integration_state_spec",
                "time", "q_cmd_left", "q_cmd_right", "grip_cmd_left", "grip_cmd_right"}
    with np.load(files["input_snapshot"][0], allow_pickle=False) as start, np.load(files["output_snapshot"][0], allow_pickle=False) as end:
        if (any(not required <= set(state.files) for state in (start, end))
                or any(not np.isfinite(state[key]).all() for state in (start, end) for key in required)
                or any(start[key].shape != end[key].shape for key in required)
                or any(int(state["integration_state_spec"]) != 16383 for state in (start, end))
                or start["ctrl"].shape != (14,) or start["q_cmd_left"].shape != (6,) or start["q_cmd_right"].shape != (6,)):
            raise ValueError("Transfer endpoint snapshots must preserve complete finite native integration/controller state")
        start_time, end_time = float(start["time"]), float(end["time"])
    elapsed = end_time - start_time
    stages = outcome.get("stages", [])
    starts = [row for row in stages if row.get("stage") == "initial"]
    ends = [row for row in stages if row.get("stage") == "post_pour_settled"]
    if (len(starts) != 1 or len(ends) != 1 or starts[0].get("sim_seconds") != 0
            or any(row.get("failure") for row in stages)
            or type(ends[0].get("sim_seconds")) not in (int, float) or not math.isfinite(ends[0]["sim_seconds"])
            or elapsed <= 0 or abs(elapsed - ends[0]["sim_seconds"]) > 1e-8
            or outcome["physics_checks"] < math.ceil(elapsed / outcome["timestep_s"] - 1e-5)
            or files["output_snapshot"][0].name != "post_pour_settled.npz"
            or outcome.get("checkpoint_sha256", {}).get(files["output_snapshot"][0].name) != record["output_snapshot_sha256"]):
        raise ValueError("Transfer needs every forward substep checked and a hash/time-bound settled endpoint")
    phases = [row.get("phase") for row in outcome.get("expert_trace", [])]
    required_phases = ["receiver_transit", "pre_pour_guard", "complete_pour", "drain_hold", "return_upright", "receiver_retract"]
    cursor = 0
    for phase in phases:
        if cursor < len(required_phases) and phase == required_phases[cursor]:
            cursor += 1
    if cursor != len(required_phases):
        raise ValueError("Forward evidence must execute ordinary receiver transit, full pour, upright return and retraction")
    for row, expected, mapping in (
            (starts[0], audit, (("source_g", "source_g"), ("tool_g", "tool_g"), ("receiver_g", "receiver_g"), ("inflight_g", "inflight_g"), ("spill_g", "spilled_g"))),
            (ends[0], outcome, (("source_g", "source_g"), ("tool_g", "tool_g"), ("receiver_g", "mass_g"), ("inflight_g", "inflight_g"), ("spill_g", "spilled_g")))):
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
               or abs(row[key] - expected[expected_key]) > 1e-8 for key, expected_key in mapping):
            raise ValueError("Executed endpoint inventory differs from the loaded audit or final physical result")
    summary = {key: outcome[key] for key in final_keys}
    summary.update(transferred_g=outcome["mass_g"], settled=True, elapsed_sim_seconds=elapsed,
                   physics_checks_scope="forward_segment_only", normalized_model_sha256=model_sha)
    return dict(kind=payload["scope"], scope=payload["scope"], path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                source_version=version, source_files_sha256=hashes, transfer_segment=record, stage_summary=summary,
                historical_prefix_validated=False, accepted_demo_count=0, accepted_demonstrations_from_gate=0,
                full_episode_accepted=False, candidate_data_eligible=False)


def validate_stage_validation(path):
    """Validate a corrective suffix as a launch gate, never as accepted data.

    Candidate episodes must still start from reset and pass the full remote
    episode audit. This artifact only justifies trying those independent jobs.
    """
    import ast
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("scope") == "saved_state_loaded_transfer_feasibility_v1":
        return _validate_loaded_transfer_feasibility(payload, path, raw)
    if (payload.get("scope") not in {"saved_state_corrective_continuation", "saved_state_chained_corrective_continuation_v1"}
            or payload.get("full_episode_accepted") is not False
            or payload.get("candidate_data_eligible") is not False
            or payload.get("stage_physical_valid") is not True
            or payload.get("no_state_writes_after_initialization") is not True
            or payload.get("initialization_contact_model_override") is not None
            or any(payload.get(key) is not True for key in ("correction_passed", "lift_passed", "pour_passed"))
            or payload.get("failure") or payload.get("grain_count") != 200):
        raise ValueError("Candidate launch requires a complete valid corrective continuation, explicitly ineligible as episode data")
    summary = payload.get("stage_summary", {})
    if not isinstance(summary, dict):
        raise ValueError("Missing continuation physical evidence")
    numeric = ("physics_checks", "grain_radius_m", "max_grain_contact_penetration_m",
               "conservation_error_g", "spilled_g", "transferred_g", "mass_g", "raw_net_mass_g",
               "depth_error_m", "lateral_error_m", "depth_tolerance_m", "lateral_tolerance_m")
    if any(type(summary.get(key)) not in (int, float) or not math.isfinite(summary[key]) for key in numeric):
        raise ValueError("Continuation evidence must contain finite numeric contact, mass and pose measurements")
    if (summary.get("settled") is not True or summary.get("finite_state") is not True
            or summary.get("forbidden_contact_failure") is not False or summary.get("rigid_contact_valid") is not True
            or type(summary["physics_checks"]) is not int or summary["physics_checks"] <= 0
            or summary["grain_radius_m"] <= 0
            or not 0 <= summary["max_grain_contact_penetration_m"] <= .2 * summary["grain_radius_m"] + 1e-12
            or abs(summary["conservation_error_g"]) > 1e-8 or abs(summary["spilled_g"]) > 1e-12
            or summary["transferred_g"] <= .050 or summary["mass_g"] < summary["transferred_g"] - 1e-9
            or abs(summary["raw_net_mass_g"] - summary["mass_g"]) > .010 + 1e-12
            or summary["depth_tolerance_m"] != .00025 or summary["lateral_tolerance_m"] != .001
            or abs(summary["depth_error_m"]) > .00025 + 1e-12
            or not 0 <= summary["lateral_error_m"] <= .001 + 1e-12):
        raise ValueError("Continuation contact, inventory, pose, settled transfer or independent force check failed")
    # Do not import the simulator in the local Modal client merely to read two constants.
    tree = ast.parse((ROOT / "scripts/collect_granular_demos.py").read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"DEPTH_TOLERANCE_M", "LATERAL_TOLERANCE_M"}}
    if constants != {"DEPTH_TOLERANCE_M": .00025, "LATERAL_TOLERANCE_M": .001}:
        raise ValueError("Continuation gate requires unchanged production depth and lateral tolerances")
    version, hashes = payload.get("source_version"), payload.get("source_files_sha256")
    current_version, current_hashes = _source_identity()
    if (not isinstance(hashes, dict) or not hashes
            or version != "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            or version != current_version or hashes != current_hashes):
        raise ValueError("Continuation source manifest is incomplete, inconsistent or differs from current production source")
    initial = _validate_initial_state_provenance(payload)
    chain = _validate_continuation_chain(payload) if payload["scope"] == "saved_state_chained_corrective_continuation_v1" else None
    return {"kind": payload["scope"], "scope": payload["scope"],
            "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "source_version": version, "source_files_sha256": hashes,
            "initial_state_provenance": initial, "motivation": initial["motivation"],
            "stage_summary": summary, "full_episode_accepted": False, "candidate_data_eligible": False,
            "accepted_demonstrations_from_gate": 0, "continuation_chain": chain}


def _reuse_reference(job, reference):
    """Adopt only the exact recorded first planned episode, after its gate."""
    raw = Path(reference["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Reference result changed after its validation gate")
    result = json.loads(raw)
    if job.get("index") != 0 or job.get("status") != "planned":
        raise ValueError("Reference reuse is restricted to the first planned episode")
    fingerprint = result.get("trajectory_sha256")
    if (result.get("accepted") is not True or result.get("diagnostic") is not False
            or result.get("approach_only") is not False or result.get("dataset_written") is not True
            or result.get("exit_code") != 0 or result.get("timed_out") is not False
            or result.get("validation") != "physical_grain_episode_audit_v1"
            or not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
        raise ValueError("Reference reuse requires a recorded accepted full-task episode with complete dataset metadata")
    expected = {"run_id": job["run_id"], "powder_id": job["powder"], "seed": job["seed"],
                "target_g": job["target"], "grain_count": job["grains"],
                "source_version": reference["source_version"],
                "dataset_path": f"/datasets/{job['run_id']}.hdf5"}
    if (any(result.get(key) != value for key, value in expected.items())
            or any(type(result.get(key)) is not int for key in ("powder_id", "seed", "grain_count"))
            or result.get("source_files_sha256") != reference["source_files_sha256"]
            or result.get("episode_identity") != [job["powder"], job["seed"], job["target"]]):
        raise ValueError("Reference identity/source/dataset path does not exactly match planned episode zero")
    outcome = result.get("outcome", {})
    if any(outcome.get(key) != expected[key] for key in ("powder_id", "seed", "target_g", "grain_count")):
        raise ValueError("Reference outcome identity disagrees with its planned episode")
    job.update(status="accepted", result=result, reused_reference=True,
               reference_result_path=reference["path"], reference_result_sha256=reference["sha256"],
               reference_source_version=reference["source_version"])
    return result


def _merge_timeout(deadline):
    remaining = deadline - time.time() - 15
    if remaining <= 0:
        raise TimeoutError("Absolute batch budget expired before merge subprocess")
    return min(MERGE_MAX_SECONDS, remaining)


def merge_accepted(batch_id, run_ids, expected_episodes, deadline):
    """CPU merge of the exact accepted files, after collection containers finish."""
    import subprocess
    import sys
    if expected_episodes not in (5, 30) or len(run_ids) != expected_episodes or len(set(run_ids)) != len(run_ids):
        raise ValueError("Merge requires exactly five or thirty distinct accepted run ids")
    for value in (batch_id, *run_ids):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
            raise ValueError("Invalid collection/merge identifier")
    if deadline - time.time() < 60:
        raise TimeoutError("Insufficient overall budget to start the merge")
    runs.reload()
    datasets.reload()
    inputs = []
    for run_id in run_ids:
        result = json.loads((Path("/runs") / run_id / "result.json").read_text())
        path = Path("/datasets") / f"{run_id}.hdf5"
        if (result.get("accepted") is not True or result.get("exit_code") != 0 or result.get("timed_out")
                or result.get("dataset_path") != str(path) or not path.is_file()):
            raise ValueError(f"Refusing unaccepted collection result {run_id}")
        inputs.append(str(path))
    label = "pilot5" if expected_episodes == 5 else "full30"
    output = Path("/datasets") / f"{batch_id}-{label}.hdf5"
    directory = Path("/runs") / batch_id / f"merge_{expected_episodes}"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.merge_granular_demos", *inputs, "--out", str(output),
               "--expected-episodes", str(expected_episodes)]
    started = time.time()
    result = {"batch_id": batch_id, "expected_episodes": expected_episodes, "dataset_path": str(output),
              "run_ids": run_ids, "deadline_epoch": deadline, "accepted": False,
              "merge_subprocess_cap_seconds": MERGE_MAX_SECONDS,
              "cpu": CPU_PER_CONTAINER, "memory_mib": MEMORY_MIB_PER_CONTAINER, "gpu": None}
    try:
        completed = subprocess.run(command, cwd=REMOTE_ROOT, capture_output=True, text=True,
                                   timeout=_merge_timeout(deadline))
        (directory / "console.log").write_text(completed.stdout + completed.stderr)
        result["exit_code"] = completed.returncode
        if completed.returncode == 0:
            audit = json.loads(completed.stdout)
            (directory / "dataset_inspection.json").write_text(json.dumps(audit, indent=2) + "\n")
            result.update(accepted=True, dataset_sha256=audit["dataset_sha256"],
                          num_episodes=audit["num_episodes"], use=audit["use"])
        else:
            result["error"] = completed.stderr[-4000:]
    except subprocess.TimeoutExpired:
        result.update(timed_out=True, error="Merge exceeded its absolute budget")
    finally:
        result["elapsed_seconds"] = time.time() - started
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        datasets.commit()
        runs.commit()
    return result


def _persist_batch(local, manifest):
    temporary = local / "batch_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(local / "batch_manifest.json")


def _get_ready(call):
    """Poll once; a terminal function timeout must not look like a pending call."""
    try:
        return True, call.get(timeout=0)
    except modal.exception.TimeoutError as error:
        if type(error) is modal.exception.TimeoutError:
            return False, None
        raise


def _collect_wave(jobs, manifest, local, parallelism, max_job_seconds):
    """Stop new submissions after failure, then preserve all running outcomes."""
    if any(job.get("status") not in {"planned", "accepted"} for job in jobs):
        raise ValueError("A wave may contain only new planned jobs or already accepted references; failed jobs cannot resume")
    active = {}
    pending = iter(job for job in jobs if job["status"] == "planned")
    exhausted = False
    failures = []
    last_progress = 0.0
    try:
        while active or not exhausted:
            if time.time() >= manifest["deadline_epoch"]:
                raise TimeoutError("The absolute six-hour collection budget expired")
            while not exhausted and not failures and len(active) < parallelism:
                try:
                    job = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    call = collect.spawn(job["run_id"], job["powder"], job["seed"], job["target"], job["grains"],
                                         False, False, manifest["deadline_epoch"], max_job_seconds,
                                         expected_source_version=manifest["expected_source_version"])
                except Exception as error:
                    job.update(status="submission_unknown", error=f"{type(error).__name__}: {error}")
                    _persist_batch(local, manifest)
                    failures.append(f"Submission status unknown for {job['run_id']}: {error}")
                    break
                job.update(status="running", call_id=call.object_id, submitted_epoch=time.time())
                active[call.object_id] = (call, job)
                _persist_batch(local, manifest)
                print(json.dumps({"event": "collection_submitted", **job}), flush=True)
            for call_id, (call, job) in list(active.items()):
                try:
                    ready, result = _get_ready(call)
                except Exception as error:
                    active.pop(call_id)
                    job.update(status="failed", error=f"{type(error).__name__}: {error}")
                    (local / f"{job['run_id']}.json").write_text(json.dumps(job, indent=2) + "\n")
                    _persist_batch(local, manifest)
                    failures.append(f"Collection failed for {job['run_id']}: {error}")
                    continue
                if not ready:
                    continue
                active.pop(call_id)
                job.update(status="accepted" if result.get("accepted") is True else "failed", result=result)
                (local / f"{job['run_id']}.json").write_text(json.dumps(result, indent=2) + "\n")
                _persist_batch(local, manifest)
                print(json.dumps({"event": "collection_finished", "run_id": job["run_id"],
                                  "accepted": result.get("accepted"), "exit_code": result.get("exit_code")}), flush=True)
                if job["status"] != "accepted":
                    failures.append(f"Collection rejected for {job['run_id']}")
            if failures:
                exhausted = True
                for job in jobs:
                    if job["status"] == "planned":
                        job.update(status="not_started", reason="earlier collection failed; no automatic retry")
                manifest["collection_failures"] = failures.copy()
                _persist_batch(local, manifest)
            if active:
                if time.time() - last_progress >= 30:
                    print(json.dumps({"event": "collection_progress", "running": len(active),
                                      "accepted": sum(j["status"] == "accepted" for j in manifest["jobs"]),
                                      "remaining_seconds": max(0, manifest["deadline_epoch"] - time.time())}), flush=True)
                    last_progress = time.time()
                time.sleep(min(5, max(0, manifest["deadline_epoch"] - time.time())))
        if failures:
            raise RuntimeError("Collection stopped after failure; running outcomes preserved: " + "; ".join(failures))
    finally:
        # Only this batch's call IDs are canceled. Existing diagnostic apps are
        # unrelated and are never stopped by this orchestrator.
        for call, job in active.values():
            try:
                call.cancel(terminate_containers=True)
                if job["status"] == "running":
                    job.update(status="canceled", reason="global deadline or orchestration interruption; no automatic retry")
                job["cancellation_requested"] = True
            except Exception as error:
                job.update(status="cancel_failed", error=f"{type(error).__name__}: {error}")
            (local / f"{job['run_id']}.json").write_text(json.dumps(job, indent=2) + "\n")
        _persist_batch(local, manifest)


def _merge_batch(manifest, local, jobs, expected_episodes):
    if any(job["status"] != "accepted" for job in jobs):
        raise ValueError("Cannot merge incomplete or failed collection jobs")
    call = collect.spawn(manifest["batch_id"], 0, 0, .5, 200, False, False, manifest["deadline_epoch"],
                         merge_run_ids=[job["run_id"] for job in jobs],
                         merge_expected_episodes=expected_episodes)
    record = {"expected_episodes": expected_episodes, "call_id": call.object_id, "status": "running"}
    manifest["merges"].append(record)
    _persist_batch(local, manifest)
    completed = False
    try:
        while time.time() < manifest["deadline_epoch"]:
            ready, result = _get_ready(call)
            if ready:
                completed = True
                record.update(status="accepted" if result.get("accepted") is True else "failed", result=result)
                (local / f"merge_{expected_episodes}.json").write_text(json.dumps(result, indent=2) + "\n")
                _persist_batch(local, manifest)
                if result.get("accepted") is not True:
                    raise RuntimeError("Strict merge rejected the collected episodes")
                print(json.dumps({"event": "dataset_ready", **result}), flush=True)
                return result
            time.sleep(min(5, max(0, manifest["deadline_epoch"] - time.time())))
        raise TimeoutError("Absolute batch budget expired during merge")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if not completed:
            call.cancel(terminate_containers=True)
            record["cancellation_requested"] = True
        _persist_batch(local, manifest)


def _collect_overlap(manifest, local, parallelism, max_job_seconds):
    """Collect all remaining episodes while auditing the first five promptly.

    Collection and merge calls share one capped function pool. A failure outside
    the pilot prevents a full merge, but does not invalidate an accepted pilot.
    Already running independent calls finish; only the absolute deadline or an
    orchestration interruption cancels them. No call is retried.
    """
    jobs = manifest["jobs"]
    if len(jobs) != 30 or any(job.get("status") not in {"planned", "accepted"} for job in jobs):
        raise ValueError("Overlapping collection requires thirty new jobs or accepted references")
    active = {}
    failures = []
    last_progress = 0.0

    def merge_record(count):
        return next((row for row in manifest["merges"] if row["expected_episodes"] == count), None)

    def submit_merge(count):
        record = dict(expected_episodes=count, status="submitting")
        manifest["merges"].append(record)
        _persist_batch(local, manifest)
        try:
            call = collect.spawn(manifest["batch_id"], 0, 0, .5, 200, False, False,
                                 manifest["deadline_epoch"], merge_run_ids=[j["run_id"] for j in jobs[:count]],
                                 merge_expected_episodes=count)
        except Exception as error:
            record.update(status="submission_unknown", error=f"{type(error).__name__}: {error}")
            failures.append(f"Merge {count} submission status unknown; no retry: {error}")
        else:
            record.update(status="running", call_id=call.object_id)
            active[call.object_id] = (call, "merge", record)
        _persist_batch(local, manifest)

    try:
        while True:
            if time.time() >= manifest["deadline_epoch"]:
                raise TimeoutError("The absolute six-hour collection budget expired")
            for call_id, (call, kind, record) in list(active.items()):
                try:
                    ready, result = _get_ready(call)
                except Exception as error:
                    active.pop(call_id)
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    label = record.get("run_id", f"merge_{record.get('expected_episodes')}")
                    failures.append(f"{label} failed: {error}")
                    if kind == "collection":
                        manifest.setdefault("collection_failures", []).append(failures[-1])
                        (local / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
                    _persist_batch(local, manifest)
                    continue
                if not ready:
                    continue
                active.pop(call_id)
                record.update(status="accepted" if result.get("accepted") is True else "failed", result=result)
                if kind == "collection":
                    label = record["run_id"]
                    (local / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps({"event": "collection_finished", "run_id": label,
                                      "accepted": result.get("accepted"), "exit_code": result.get("exit_code")}), flush=True)
                else:
                    label = f"merge_{record['expected_episodes']}"
                    (local / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
                    if result.get("accepted") is True:
                        print(json.dumps({"event": "dataset_ready", **result}), flush=True)
                if record["status"] != "accepted":
                    failures.append(f"{label} rejected")
                    if kind == "collection":
                        manifest.setdefault("collection_failures", []).append(failures[-1])
                _persist_batch(local, manifest)

            # An independent non-pilot failure does not invalidate these five.
            if (merge_record(5) is None and all(j["status"] == "accepted" for j in jobs[:5])
                    and len(active) < parallelism):
                submit_merge(5)
            pilot = merge_record(5)
            if (merge_record(30) is None and pilot is not None and pilot["status"] == "accepted"
                    and all(j["status"] == "accepted" for j in jobs) and len(active) < parallelism):
                submit_merge(30)

            while not failures and len(active) < parallelism:
                job = next((j for j in jobs if j["status"] == "planned"), None)
                if job is None:
                    break
                try:
                    call = collect.spawn(job["run_id"], job["powder"], job["seed"], job["target"], job["grains"],
                                         False, False, manifest["deadline_epoch"], max_job_seconds,
                                         expected_source_version=manifest["expected_source_version"])
                except Exception as error:
                    job.update(status="submission_unknown", error=f"{type(error).__name__}: {error}")
                    failures.append(f"Submission status unknown for {job['run_id']}: {error}")
                    manifest.setdefault("collection_failures", []).append(failures[-1])
                    break
                job.update(status="running", call_id=call.object_id, submitted_epoch=time.time())
                active[call.object_id] = (call, "collection", job)
                print(json.dumps({"event": "collection_submitted", **job}), flush=True)
                _persist_batch(local, manifest)
            if failures:
                manifest["overlap_failures"] = failures.copy()
                for job in jobs:
                    if job["status"] == "planned":
                        job.update(status="not_started", reason="earlier failure; no automatic retry")
                _persist_batch(local, manifest)
            if not active:
                break
            if time.time() - last_progress >= 30:
                print(json.dumps({"event": "collection_progress", "running_calls_including_merges": len(active),
                                  "accepted": sum(j["status"] == "accepted" for j in jobs),
                                  "remaining_seconds": max(0, manifest["deadline_epoch"] - time.time())}), flush=True)
                last_progress = time.time()
            time.sleep(min(5, max(0, manifest["deadline_epoch"] - time.time())))
        full = merge_record(30)
        if failures or full is None or full["status"] != "accepted":
            raise RuntimeError("Full collection incomplete; independent accepted pilot and outcomes preserved: " + "; ".join(failures))
    finally:
        for call, kind, record in active.values():
            try:
                call.cancel(terminate_containers=True)
                record.update(status="canceled", cancellation_requested=True,
                              reason="global deadline or orchestration interruption; no automatic retry")
            except Exception as error:
                record.update(status="cancel_failed", error=f"{type(error).__name__}: {error}")
            if kind == "collection":
                (local / f"{record['run_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
        _persist_batch(local, manifest)


@app.local_entrypoint()
def batch(batch_id: str, reference_result: str, count: int = 5, seed_base: int = 1001,
          grains: int = 200, parallelism: int = 8, max_minutes: int = 360, job_minutes: int = 120,
          reference_source: str = "", reuse_reference: bool = False, overlap_waves: bool = False):
    """Collect a five-demo pilot, optionally extend to thirty, with no retries.

    --reuse-reference adopts an accepted recorded reference only when it is the
    exact planned episode zero. The strict five-episode merge audits it again.
    --overlap-waves starts remaining episodes before the first-five merge; it
    requires count30 and preserves an accepted pilot if another episode fails.
    """
    reference = validate_reference_result(reference_result, reference_source or None)
    jobs = collection_plan(batch_id, count, seed_base, grains)
    if (type(parallelism) is not int or not 1 <= parallelism <= MAX_COLLECTION_CONTAINERS
            or not 5 <= max_minutes <= 360 or not 1 <= job_minutes <= 120):
        raise ValueError("Limits: at most 30 shared CPU containers, 6 hours overall, and 120 minutes per job")
    if overlap_waves and count != 30:
        raise ValueError("--overlap-waves requires --count 30")
    reused = _reuse_reference(jobs[0], reference) if reuse_reference else None
    from scripts.modal_train import verify_workspace, _require_remote_dataset
    verify_workspace()
    if reused is not None:
        _require_remote_dataset(Path(reused["dataset_path"]).name)
    _run_batch(batch_id, jobs, {"kind": "full_robot_reference", **reference}, parallelism, max_minutes, job_minutes,
               overlap_waves=overlap_waves, reused=reused, reference=reference)


def _run_batch(batch_id, jobs, gate, parallelism, max_minutes, job_minutes, *,
               overlap_waves, reused=None, reference=None, candidate_mode=False):
    """Common execution with a source gate whose scope stays explicit."""
    if candidate_mode and (reused is not None or reference is not None
                           or any(job["status"] != "planned" for job in jobs)):
        raise ValueError("Candidate collection never reuses a reference or stage trajectory")
    local = ROOT / "docs/pilot/granular_batches" / batch_id
    local.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(batch_id=batch_id, requested_episodes=len(jobs), started_epoch=started,
                    deadline_epoch=started + max_minutes * 60, validation_gate=gate,
                    expected_source_version=gate["source_version"],
                    max_parallel_collectors=parallelism, per_job_seconds=job_minutes * 60,
                    cpu_per_container=CPU_PER_CONTAINER, memory_mib_per_container=MEMORY_MIB_PER_CONTAINER,
                    max_cpu_cores=parallelism * CPU_PER_CONTAINER,
                    max_memory_gib=parallelism * MEMORY_MIB_PER_CONTAINER / 1024,
                    resource_limits_enforced=True, gpu=None, retries=0,
                    training_powder_ids=[0, 1, 2, 3, 4], held_out_powder_ids=[5],
                    first_pilot_episodes=5, reuse_reference=reused is not None,
                    overlap_waves=bool(overlap_waves),
                    candidate_mode=bool(candidate_mode),
                    schedule=("overlapping_candidates_after_continuation" if candidate_mode else
                              "overlapping_after_reference" if overlap_waves else "pilot_then_remaining"),
                    jobs=jobs, merges=[], status="running")
    if reference is not None:
        manifest["reference_robot_result"] = reference
    if reused is not None:
        (local / f"{jobs[0]['run_id']}.json").write_text(json.dumps(reused, indent=2) + "\n")
    _persist_batch(local, manifest)
    try:
        # The selected cap applies to collection and merging in the same pool.
        # Official SDK: Function.autoscaler.update; no second dynamic pool.
        collect.autoscaler.update(max_containers=parallelism, min_containers=0, buffer_containers=0)
        if overlap_waves:
            _collect_overlap(manifest, local, parallelism, job_minutes * 60)
        else:
            _collect_wave(jobs[:5], manifest, local, parallelism, job_minutes * 60)
            _merge_batch(manifest, local, jobs[:5], 5)
            if len(jobs) == 30:
                _collect_wave(jobs[5:], manifest, local, parallelism, job_minutes * 60)
                _merge_batch(manifest, local, jobs, 30)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        for job in jobs:
            if job["status"] == "planned":
                job.update(status="not_started", reason="batch stopped after failure or deadline")
        raise
    finally:
        manifest["elapsed_seconds"] = time.time() - started
        _persist_batch(local, manifest)


@app.local_entrypoint()
def candidate_batch(batch_id: str, stage_validation: str, seed_base: int = 1001,
                    grains: int = 200, parallelism: int = 30, max_minutes: int = 360,
                    job_minutes: int = 120):
    """Attempt thirty new full-reset episodes after an explicitly partial stage test.

    A corrective continuation is a launch gate only and contributes no data.
    Every collected episode still passes the existing full physical audit.
    """
    gate = validate_stage_validation(stage_validation)
    jobs = collection_plan(batch_id, count=30, seed_base=seed_base, grains=grains)
    if (type(parallelism) is not int or not 1 <= parallelism <= MAX_COLLECTION_CONTAINERS
            or not 5 <= max_minutes <= 360 or not 1 <= job_minutes <= 120):
        raise ValueError("Limits: at most 30 shared CPU containers, 6 hours overall, 120 minutes per job")
    from scripts.modal_train import verify_workspace
    verify_workspace()
    _run_batch(batch_id, jobs, gate, parallelism, max_minutes, job_minutes,
               overlap_waves=True, candidate_mode=True)
