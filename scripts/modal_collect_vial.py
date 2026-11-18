"""Bounded parallel collection of 90 additional native vial demonstrations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import modal

from scripts.modal_train import image, verify_workspace, ROOT

app = modal.App("chemlab-vial-data")
datasets = modal.Volume.from_name("chemlab-rl-data", create_if_missing=True, version=2)
runs = modal.Volume.from_name("chemlab-rl-runs", create_if_missing=True, version=2)


def validate_request(batch_id, count, parallelism, max_attempts, max_minutes):
    if not re.fullmatch(r"vial90-[a-zA-Z0-9_-]+", batch_id):
        raise ValueError("Expected unique vial90-prefixed batch id")
    if not 1 <= count <= 90 or not 1 <= parallelism <= 10:
        raise ValueError("Collection supports at most90 accepted/10 parallel workers")
    if not count <= max_attempts <= 150 or not 1 <= max_minutes <= 90:
        raise ValueError("Bounded candidates/time required")


@app.function(image=image, gpu=["L4", "A10G", "T4", "A100-80GB"], cpu=(2, 2), memory=(4096, 4096), timeout=600,
              startup_timeout=120, max_containers=10, retries=0, scaledown_window=2,
              volumes={"/datasets": datasets, "/runs": runs})
def candidate(batch_id: str, index: int, seed: int, deadline: float):
    from scripts.collect_vial_demos import collect_one
    if time.time() >= deadline - 45:
        return {"accepted": False, "index": index, "error": "batch_deadline"}
    path = Path(f"/datasets/{batch_id}/candidate_{index:03d}.hdf5")
    result = collect_one(path, seed, max_seconds=min(540., deadline - time.time() - 30))
    result.update(index=index, dataset_path=str(path))
    datasets.commit()
    return result


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/datasets": datasets, "/runs": runs})
def merge(batch_id: str, accepted: list[dict], expected: int):
    import h5py
    import numpy as np
    datasets.reload()
    if len(accepted) != expected or not all(r.get("accepted") for r in accepted):
        raise ValueError("Only the exact requested count of accepted candidates can be merged")
    if len({r["trajectory_sha256"] for r in accepted}) != expected:
        raise ValueError("Duplicate trajectories refused")
    if len({r["source_version"] for r in accepted}) != 1 or len({r["collector_sha256"] for r in accepted}) != 1:
        raise ValueError("Mixed source versions refused")
    target = Path(f"/datasets/{batch_id}/additional_{expected}.hdf5")
    if target.exists():
        raise FileExistsError(target)
    total = 0
    with h5py.File(target, "w") as out:
        data = out.create_group("data")
        for i, result in enumerate(accepted):
            with h5py.File(result["dataset_path"], "r") as source:
                demo = source["data"][result["episode"]]
                if not demo.attrs.get("physical_valid") or not demo.attrs.get("success"):
                    raise ValueError("Unvalidated source demo")
                done = demo["dones"][:]
                if not done[-1] or np.any(done[:-1]):
                    raise ValueError("Post-terminal samples or missing success")
                if i == 0:
                    for key, value in source["data"].attrs.items():
                        data.attrs[key] = value
                source.copy(demo, data, name=f"demo_{i}")
                total += len(done)
        data.attrs["total"] = total
        out.attrs["collection_manifest"] = json.dumps(accepted)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    result = dict(dataset=str(target), sha256=digest.hexdigest(), accepted=expected,
                  samples=total, bytes=target.stat().st_size, fixed_scene=True)
    report = Path(f"/runs/{batch_id}")
    report.mkdir(parents=True, exist_ok=True)
    (report / "collection_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    datasets.commit()
    runs.commit()
    return result


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/datasets": datasets, "/runs": runs})
def combine_existing(batch_id: str):
    """Preserve legacy/audited provenance separately in an explicit98-demo file."""
    import h5py
    from chemlab_rl.vla_bc import file_sha256
    datasets.reload()
    original = Path("/datasets/weigh_vial_grok_v2.hdf5")
    additional = Path(f"/datasets/{batch_id}/additional_90.hdf5")
    target = Path(f"/datasets/{batch_id}_combined98.hdf5")
    old_sha = file_sha256(original)
    if old_sha != "1d2fc0e3a809b1bb00ceb12b4fa99bc8d7c39175e9e39b5a1592d20d7c942950":
        raise ValueError("Original dataset differs from the audited file")
    new_sha = file_sha256(additional)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(original, "r") as old, h5py.File(additional, "r") as new, h5py.File(target, "w") as out:
        if len(new["data"]) != 90:
            raise ValueError("Exactly90 accepted additional demos required")
        audits = json.loads(new.attrs["collection_manifest"])
        if len(audits) != 90:
            raise ValueError("Missing per-episode substep audits")
        data = out.create_group("data")
        for key, value in old["data"].attrs.items():
            data.attrs[key] = value
        args = json.loads(data.attrs["env_args"])
        args["collector"] = "hybrid_vial_98_v1"
        data.attrs["env_args"] = json.dumps(args)
        total = 0
        for i in range(98):
            source = old if i < 8 else new
            source_index = i if i < 8 else i - 8
            name = f"demo_{source_index}"
            origin = source["data"][name]
            if i >= 8 and (not origin.attrs.get("physical_valid")
                           or origin.attrs.get("physics_validation") != "native_vial_substep_contacts_v1"):
                raise ValueError("Additional episode missing physical acceptance")
            source.copy(origin, data, name=f"demo_{i}")
            demo = data[f"demo_{i}"]
            demo.attrs["collector"] = "grok_session_v2" if i < 8 else "audited_continuous_vial_v1"
            demo.attrs["source_dataset_sha256"] = old_sha if i < 8 else new_sha
            demo.attrs["source_episode"] = name
            if i >= 8:
                audit = audits[source_index]
                trajectory_sha = hashlib.sha256(demo["actions"][:].tobytes() + demo["obs/joint_pos"][:].tobytes()).hexdigest()
                if (not audit.get("accepted") or audit["trajectory_sha256"] != trajectory_sha
                        or audit["samples"] != len(demo["actions"])):
                    raise ValueError("Recorded episode and physical audit disagree")
                demo.attrs["physics_audit"] = json.dumps(audit)
            total += int(demo.attrs["num_samples"])
        data.attrs["total"] = total
        out.attrs["excluded_original_episode"] = "demo_8"
        out.attrs["exclusion_reason"] = "22 recorded frames above100g; peak3603.84g during placement"
    split = dict(train_episodes=[f"demo_{i}" for i in [*range(7), *range(8, 98)]],
                 heldout_episodes=["demo_7"], excluded_episodes=[])
    target.with_suffix(".split.json").write_text(json.dumps(split, indent=2) + "\n")
    result = dict(dataset=target.name, sha256=file_sha256(target), split=split,
                  original_sha256=old_sha, additional_sha256=new_sha,
                  training_episodes=97, heldout_episodes=1, additional_episodes=90,
                  historical_contacts_certified=False, fixed_scene=True)
    folder = Path(f"/runs/{batch_id}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "combined_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    datasets.commit()
    runs.commit()
    return result


@app.local_entrypoint()
def combine(batch_id: str = "vial90-audited-001"):
    validate_request(batch_id, 90, 10, 120, 60)
    verify_workspace()
    result = combine_existing.remote(batch_id)
    folder = ROOT / "docs/pilot" / batch_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "combined_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


@app.local_entrypoint()
def main(batch_id: str = "vial90-audited-001", count: int = 90, parallelism: int = 10,
         max_attempts: int = 120, max_minutes: int = 60, seed: int = 3000):
    validate_request(batch_id, count, parallelism, max_attempts, max_minutes)
    verify_workspace()
    local = ROOT / "docs/pilot" / batch_id
    local.mkdir(parents=True, exist_ok=True)
    state_path = local / "collection_state.json"
    if state_path.exists():
        raise FileExistsError("Use a new batch id; no silent restart or duplicate billing")
    started = time.time()
    deadline = started + max_minutes * 60
    state = dict(batch_id=batch_id, requested=count, parallelism=parallelism,
                 max_attempts=max_attempts, deadline=deadline, candidates=[], accepted=[],
                 scope="additional_fixed_scene_vial_placement_images", retries=0)

    def save():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    index = 0
    while len(state["accepted"]) < count and index < max_attempts and time.time() < deadline - 60:
        n = min(parallelism, count - len(state["accepted"]), max_attempts - index)
        requests = [(batch_id, i, seed + i, deadline) for i in range(index, index + n)]
        index += n
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(candidate.remote, *args): args[1] for args in requests}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    # Avoid returning SDK details that may contain signed URLs.
                    result = dict(index=futures[future], accepted=False, error=type(error).__name__)
                state["candidates"].append(result)
                if result.get("accepted"):
                    state["accepted"].append(result)
                save()
                print(json.dumps({"event": "candidate_complete", "index": result["index"],
                                  "accepted": result.get("accepted", False),
                                  "total_accepted": len(state["accepted"]), "attempted": len(state["candidates"]),
                                  "error": result.get("error")}), flush=True)
        # Stop a systematically failing setup before spending all attempts.
        if len(state["candidates"]) >= parallelism and not state["accepted"]:
            state["stop_reason"] = "first_wave_all_failed"
            break
    if len(state["accepted"]) == count:
        state["merge"] = merge.remote(batch_id, state["accepted"], count)
    else:
        state.setdefault("stop_reason", "count_not_reached_within_budget")
    state["elapsed_seconds"] = time.time() - started
    save()
    print(json.dumps({key: value for key, value in state.items() if key not in ("candidates", "accepted")}), flush=True)
