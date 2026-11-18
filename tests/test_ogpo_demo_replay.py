"""Synthetic file-contract tests only; no claim these fixtures simulate physics."""
import json

import h5py
import numpy as np
import pytest

from chemlab_rl.ogpo_training import PersistentReplay, import_demonstrations
from chemlab_rl.vla_bc import CAMERAS, file_sha256, prepare_manifest
from scripts.merge_granular_demos import merge_demos
from test_vla_data import make_physical_episode


@pytest.fixture
def accepted(tmp_path):
    inputs = []
    for powder in range(5):
        path = tmp_path / f"source_{powder}.hdf5"
        make_physical_episode(path, powder)
        with h5py.File(path, "r+") as handle:
            # Extend the existing small contract fixture to an odd terminal
            # horizon, 200 grains, distinct time-coded images and bounded actions.
            demo = handle["data/demo_0"]
            names = []
            demo.visititems(lambda name, value: names.append(name) if isinstance(value, h5py.Dataset)
                            and value.ndim > 0 and value.shape[0] == 4 else None)
            for name in names:
                value = demo[name][:]
                del demo[name]
                demo.create_dataset(name, data=np.concatenate([value, value[-1:]], axis=0))
            handle.attrs["grain_count"] = 200
            demo.attrs.update(grain_count=200, num_samples=5, physics_checks=2001)
            demo["physics/source_g"][:] -= 200 * demo.attrs["grain_mass_g"]
            demo["physics/checks"][:] = np.arange(1, 6) * 400 + 1
            demo["physics/elapsed_s"][:] = np.arange(1, 6) * .1
            demo["dones"][:] = [False, False, False, False, True]
            demo["subgoal_dones"][:] = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
            values = np.arange(6 * 14, dtype=np.float32).reshape(6, 14) / 100 + powder / 100
            demo["actions"][:] = values[:-1] + .01
            balance = demo["physics/balance_g"][:]
            for prefix, offset, scale in (("obs", 0, np.r_[0, balance[:-1]]), ("next_obs", 1, balance)):
                demo[f"{prefix}/proprio"][:] = values[offset:offset+5]
                demo[f"{prefix}/scale_reading_g"][:] = scale.reshape(-1, 1)
                for camera in CAMERAS:
                    demo[f"{prefix}/{camera}"][:] = np.broadcast_to(
                        (np.arange(offset, offset+5) + powder * 10).astype(np.uint8)[:, None, None, None],
                        (5, 8, 8, 3))
                demo.create_dataset(f"{prefix}/hidden_mass", data=np.full(5, 999.))
        inputs.append(path)
    merged = tmp_path / "accepted.hdf5"
    merge_demos(inputs, merged, expected_episodes=5)
    manifest = prepare_manifest(merged, tmp_path / "bc")
    with h5py.File(merged) as handle:
        sources = json.loads(handle.attrs["source_files_sha256"])
    replay = PersistentReplay(tmp_path / "replay.sqlite", capacity=100, provenance="fixture-physical-bc")
    arguments = dict(bc_manifest=manifest, expected_source_version=manifest["source_version"],
                     expected_source_hashes=sources, grain_count=200, max_episode_steps=2600,
                     action_low=[-1.] * 14, action_high=[1.] * 14, mujoco_version="3.16.0")
    yield replay, merged, arguments
    replay.close()


def rows(replay):
    return [replay._decode(row[0]) for row in replay.db.execute("SELECT payload FROM transitions ORDER BY id")]


def refresh_digest(path, arguments):
    arguments["bc_manifest"]["dataset_sha256"] = file_sha256(path)


def assert_empty(replay):
    assert len(replay) == 0
    assert replay.checkpoint_state()["revision"] == 0
    assert replay.db.execute("SELECT value FROM metadata WHERE key='demonstration_import'").fetchone() is None


def test_import_exact_h2_alignment_odd_terminal_sensor_allowlist_and_provenance(accepted):
    replay, path, arguments = accepted
    report = import_demonstrations(replay, path, **arguments)
    assert report["mode"] == "demonstration_seeded_off_policy"
    assert report["transition_count"] == len(replay) == 15
    assert report["control_count"] == 25 and report["warmup_changed"] is False
    assert {row["powder_id"] for row in report["episodes"]} == set(range(5))
    assert report["physical_metadata"]["held_out_powder_ids"] == [5]
    assert report["request"]["source_files_sha256"] == arguments["expected_source_hashes"]
    assert report["episodes"][0]["merge_origin"]["path"].endswith("source_0.hdf5")
    assert report["episodes"][0]["replay_first_id"] == 1
    assert report["episodes"][0]["replay_last_id"] == 3
    first, middle, terminal = rows(replay)[:3]
    assert first.executed_steps == middle.executed_steps == 2
    assert not first.terminated and not first.truncated
    assert terminal.executed_steps == 1 and terminal.terminated and not terminal.truncated
    np.testing.assert_array_equal(first.dones, [0, 0, 0])
    np.testing.assert_array_equal(first.next_dones, middle.dones)
    np.testing.assert_array_equal(middle.dones, [1, 0, 0])
    np.testing.assert_array_equal(middle.next_dones, terminal.dones)
    np.testing.assert_array_equal(terminal.dones, [1, 1, 0])
    np.testing.assert_array_equal(terminal.next_dones, [1, 1, 1])
    assert first.observation["top1"][0, 0, 0] == 0
    assert first.next_observation["top1"][0, 0, 0] == 2
    assert terminal.observation["top1"][0, 0, 0] == 4
    assert terminal.next_observation["top1"][0, 0, 0] == 5
    assert set(first.observation) == {"task", "proprio", "scale_reading_g", *CAMERAS}
    with h5py.File(path) as handle:
        demo = handle["data/demo_0"]
        np.testing.assert_array_equal(first.actions, demo["actions"][:2])
        np.testing.assert_array_equal(terminal.actions[0], demo["actions"][4])
        np.testing.assert_array_equal(first.next_observation["scale_reading_g"], demo["next_obs/scale_reading_g"][1])
    np.testing.assert_array_equal(terminal.actions[1], np.zeros(14))


def test_import_is_durable_idempotent_and_rejects_changed_resume_provenance(accepted):
    replay, path, arguments = accepted
    import_demonstrations(replay, path, **arguments)
    before = replay.checkpoint_state()
    second = PersistentReplay(replay.path, capacity=100, provenance="fixture-physical-bc")
    try:
        report = import_demonstrations(second, path, **arguments)
        assert report["already_imported"] and len(second) == 15
        assert second.checkpoint_state()["revision"] == before["revision"]
        changed = dict(arguments, expected_source_version="different")
        with pytest.raises(ValueError, match="Existing demonstration import"):
            import_demonstrations(second, path, **changed)
    finally:
        second.close()


@pytest.mark.parametrize("mutation,error", [
    ("sha", "SHA256"), ("source", "physical source"), ("grain", "physical source"),
    ("mujoco", "physical source"), ("capacity", "capacity"), ("horizon", "horizon"),
    ("manifest_split", "BC training split"), ("legacy", "Legacy|physical training-only"),
])
def test_import_rejects_incompatible_contracts_before_any_rows(accepted, mutation, error):
    replay, path, arguments = accepted
    if mutation == "sha":
        arguments["bc_manifest"]["dataset_sha256"] = "0" * 64
    elif mutation == "source":
        arguments["expected_source_hashes"] = {"other.py": "f" * 64}
    elif mutation == "grain":
        arguments["grain_count"] = 400
    elif mutation == "mujoco":
        arguments["mujoco_version"] = "unknown"
    elif mutation == "capacity":
        replay.contract["capacity"] = 14
    elif mutation == "horizon":
        arguments["max_episode_steps"] = 4
    elif mutation == "manifest_split":
        arguments["bc_manifest"]["train_episodes"] = ["demo_0"]
    elif mutation == "legacy":
        arguments["bc_manifest"]["legacy"] = True
    with pytest.raises(ValueError, match=error):
        import_demonstrations(replay, path, **arguments)
    assert_empty(replay)


@pytest.mark.parametrize("mutation,error", [
    ("camera", "discontinuous camera"), ("unordered", "ordered|prefix"),
    ("time_base", "discount time base"), ("action", "physical robot bounds"),
    ("origin", "merge provenance"), ("heldout", "held-out"), ("merge", "accepted"),
])
def test_late_dataset_validation_failure_rolls_back_entire_seed(accepted, mutation, error):
    replay, path, arguments = accepted
    with h5py.File(path, "r+") as handle:
        demo = handle["data/demo_4"]  # Four valid episodes can already be inserted.
        if mutation == "camera":
            demo["next_obs/top1"][3, 0, 0, 0] = 255
        elif mutation == "unordered":
            demo["subgoal_dones"][0] = [0, 1, 0]
            demo["subgoal_dones"][1:3] = [[1, 1, 0], [1, 1, 0]]
        elif mutation == "time_base":
            demo["physics/elapsed_s"][:] *= 2
        elif mutation == "action":
            arguments["action_high"] = [.01] * 14
        elif mutation == "origin":
            origin = json.loads(handle.attrs["merge_provenance"])
            origin[-1]["trajectory_sha256"] = "0" * 64
            handle.attrs["merge_provenance"] = json.dumps(origin)
        elif mutation == "heldout":
            demo.attrs["powder_id"] = 5
        elif mutation == "merge":
            handle.attrs["merge_validation"] = "unaccepted"
    refresh_digest(path, arguments)
    with pytest.raises(ValueError, match=error):
        import_demonstrations(replay, path, **arguments)
    assert_empty(replay)


def test_late_serialization_failure_is_atomic_and_allows_clean_retry(accepted, monkeypatch):
    replay, path, arguments = accepted
    encode = replay._encode_transition
    calls = 0
    def fail_on_last(transition):
        nonlocal calls
        calls += 1
        if calls == 15:
            raise RuntimeError("simulated serialization failure")
        return encode(transition)
    monkeypatch.setattr(replay, "_encode_transition", fail_on_last)
    with pytest.raises(RuntimeError, match="serialization"):
        import_demonstrations(replay, path, **arguments)
    assert calls == 15
    assert_empty(replay)
    monkeypatch.setattr(replay, "_encode_transition", encode)
    assert import_demonstrations(replay, path, **arguments)["transition_count"] == 15


def test_changed_input_during_import_rolls_back(accepted, monkeypatch):
    replay, path, arguments = accepted
    import chemlab_rl.vla_bc as bc
    original = bc.file_sha256
    calls = 0
    def changed_digest(filename):
        nonlocal calls
        calls += 1
        return original(filename) if calls == 1 else "f" * 64
    monkeypatch.setattr(bc, "file_sha256", changed_digest)
    with pytest.raises(ValueError, match="changed during"):
        import_demonstrations(replay, path, **arguments)
    assert_empty(replay)


def test_does_not_add_seed_to_nonempty_replay_and_default_cli_remains_opt_in(accepted):
    replay, path, arguments = accepted
    import_demonstrations(replay, path, **arguments)
    # Simulate a separately populated replay without the import marker.
    with replay.db:
        replay.db.execute("DELETE FROM metadata WHERE key='demonstration_import'")
    with pytest.raises(ValueError, match="empty replay"):
        import_demonstrations(replay, path, **arguments)
    assert len(replay) == 15
    from scripts.train_ogpo_vadar import parser
    options = parser().parse_args(["--checkpoint", "bc", "--output", "run"])
    assert options.demo_replay is None


def test_image_reads_use_bounded_windows_and_report_actual_payload_cost(accepted, monkeypatch):
    replay, path, arguments = accepted
    original = h5py.Dataset.__getitem__
    def bounded_image_window(dataset, index):
        if dataset.name.rsplit("/", 1)[-1] in CAMERAS:
            assert isinstance(index, slice) and index.start is not None and index.stop is not None
            assert index.step in (None, 1) and 0 < index.stop-index.start <= 32
        return original(dataset, index)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_image_window)
    progress = []
    report = import_demonstrations(replay, path, **arguments, on_progress=progress.append)
    assert report["transition_count"] == 15
    assert report["image_read_window_controls"] == 32
    assert report["max_image_window_bytes"] == 6 * 5 * 8 * 8 * 3
    assert report["encoded_payload_bytes"] == replay.db.execute("SELECT SUM(length(payload)) FROM transitions").fetchone()[0]
    assert len(progress) == 5 and progress[-1]["episodes"] == 5
    assert progress[-1]["event"].endswith("pending_commit")
    assert all(value >= 0 for value in report["phase_seconds"].values())
    assert report["elapsed_seconds"] >= report["elapsed_before_commit_seconds"]


def extend_last_episode(path, arguments, length=67):
    """Extend a file-contract fixture across multiple I/O boundaries, no physics."""
    from scripts.merge_granular_demos import _fingerprint
    with h5py.File(path, "r+") as handle:
        demo = handle["data/demo_4"]
        names = []
        demo.visititems(lambda name, value: names.append(name) if isinstance(value, h5py.Dataset)
                        and value.ndim > 0 and value.shape[0] == 5 else None)
        for name in names:
            old = demo[name][:]
            del demo[name]
            demo.create_dataset(name, data=np.concatenate([old, np.repeat(old[-1:], length-5, axis=0)]))
        demo.attrs.update(num_samples=length, physics_checks=length*400+1)
        demo["physics/checks"][:] = np.arange(1, length+1) * 400 + 1
        demo["physics/elapsed_s"][:] = np.arange(1, length+1) * .1
        demo["dones"][:] = False
        demo["dones"][-1] = True
        demo["subgoal_dones"][4:-1] = [1, 1, 0]
        value = np.arange((length+1)*14, dtype=np.float32).reshape(length+1, 14) / 1000
        demo["actions"][:] = value[:-1]
        balance = demo["physics/balance_g"][:]
        for prefix, offset, scale in (("obs", 0, np.r_[0, balance[:-1]]), ("next_obs", 1, balance)):
            demo[f"{prefix}/proprio"][:] = value[offset:offset+length]
            demo[f"{prefix}/scale_reading_g"][:] = scale.reshape(-1, 1)
            for camera in CAMERAS:
                demo[f"{prefix}/{camera}"][:] = np.broadcast_to(
                    np.arange(offset, offset+length).astype(np.uint8)[:, None, None, None], (length, 8, 8, 3))
        origins = json.loads(handle.attrs["merge_provenance"])
        origins[-1]["trajectory_sha256"] = _fingerprint(demo)
        handle.attrs["merge_provenance"] = json.dumps(origins)
    refresh_digest(path, arguments)


@pytest.mark.parametrize("edge", [31, 32, 63, 64])
def test_every_adjacent_pair_is_checked_across_window_boundaries(accepted, edge):
    replay, path, arguments = accepted
    extend_last_episode(path, arguments)
    with h5py.File(path, "r+") as handle:
        handle["data/demo_4/next_obs/top1"][edge, 0, 0, 0] = 255
    refresh_digest(path, arguments)
    with pytest.raises(ValueError, match=f"discontinuous camera transition top1/{edge}"):
        import_demonstrations(replay, path, **arguments)
    assert_empty(replay)


def test_chunk_alignment_and_terminal_tail_survive_multiple_io_windows(accepted):
    replay, path, arguments = accepted
    extend_last_episode(path, arguments)
    report = import_demonstrations(replay, path, **arguments)
    assert report["transition_count"] == 12 + 34
    assert report["max_image_window_bytes"] == 6 * 32 * 8 * 8 * 3
    trajectory = rows(replay)[12:]
    for index, transition in enumerate(trajectory):
        start, end = index*2, min(index*2+2, 67)
        assert transition.observation["top1"][0, 0, 0] == start
        assert transition.next_observation["top1"][0, 0, 0] == end
        assert transition.executed_steps == end-start
        assert transition.terminated == (end == 67)
    np.testing.assert_array_equal(trajectory[-1].actions[1], np.zeros(14))
