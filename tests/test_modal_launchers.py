"""Offline launcher contracts; all cloud calls and workspace checks are mocked."""
import ast
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import modal_collect_granular as c, modal_train as t
from modal.types import FileEntryType

class FakeCall:

    def __init__(self, owner, index, args, kwargs):
        self.owner, self.index, self.args, self.kwargs = (owner, index, args, kwargs)
        self.object_id = f'call-{index}'
        self.polls = 0
        self.canceled = False
        self.done = False

    def get(self, timeout=0):
        self.polls += 1
        if self.polls <= self.owner.delays.get(self.index, 0):
            raise c.modal.exception.TimeoutError()
        if not self.done:
            self.owner.active -= 1
            self.done = True
        value = self.owner.outcomes.get(self.index, {'accepted': True, 'exit_code': 0})
        if isinstance(value, Exception):
            raise value
        return value

    def cancel(self, **kwargs):
        self.canceled = True

class FakeSpawner:

    def __init__(self, outcomes=None, delays=None):
        self.calls = []
        self.outcomes = outcomes or {}
        self.delays = delays or {}
        self.active = 0
        self.peak = 0

    def spawn(self, *args, **kwargs):
        x = FakeCall(self, len(self.calls), args, kwargs)
        self.calls.append(x)
        self.active += 1
        self.peak = max(self.peak, self.active)
        return x

class ModalLauncherTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.tmp.cleanup()

    def manifest(self, count=5):
        return dict(batch_id='test', deadline_epoch=time.time() + 300, jobs=c.collection_plan('test', count), merges=[], expected_source_version='sha256:expected', reference_robot_result={'source_version': 'sha256:expected'})

    def wave(self, m, s, parallel=2):
        with patch.object(c, 'collect', s), patch.object(c.time, 'sleep', lambda _: None):
            c._collect_wave(m['jobs'], m, self.path, parallel, 120)

    def test_stdlib_source_inventory_matches_collector_and_snapshot(self):
        # Execute only the pure hash helper, never the collector module imports.
        source = (c.ROOT / 'scripts/collect_granular_demos.py').read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'source_identity')
        scope = dict(ROOT=c.ROOT, Path=Path, hashlib=hashlib, json=json)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<source_identity>', 'exec'), scope)
        expected = scope['source_identity']()
        self.assertEqual(c._source_identity(), expected)
        snapshot_path = c.ROOT / 'docs/pilot/physics_validation/granular-robot-cpu-003.local-source-snapshot.json'
        snapshot = json.loads(snapshot_path.read_text())
        self.assertEqual(len(snapshot['source_files_sha256']), 346)
        self.assertEqual(set(expected[1]), set(snapshot['source_files_sha256']))
        digest = hashlib.sha256(json.dumps(snapshot['source_files_sha256'], sort_keys=True).encode()).hexdigest()
        self.assertEqual(snapshot['source_version'], 'sha256:' + digest)

    def test_plan_balanced(self):
        p = c.collection_plan('batch', 30)
        self.assertEqual(len({j['seed'] for j in p}), 30)
        for material in range(5):
            values = [j['target'] for j in p if j['powder'] == material]
            self.assertEqual(sorted(values), [0.3, 0.3, 0.5, 0.5, 0.7, 0.7])
        self.assertEqual([j['target'] for j in p[:5]], [0.5] * 5)

    def test_plan_rejects_heldout_or_bad_protocol(self):
        for kw in ({'count': 6}, {'grains': 400}, {'seed_base': 0}, {'count': 5.0}):
            with self.assertRaises(ValueError):
                c.collection_plan('test', **kw)

    def test_batch_rejects_resource_limits_before_cloud_calls(self):
        invalid = ({'parallelism': 0}, {'parallelism': 31}, {'parallelism': True},
                   {'parallelism': 1.5}, {'max_minutes': 361}, {'job_minutes': 121})
        with patch.object(c, 'validate_reference_result', return_value={'source_version': 'fixture'}), \
                patch.object(t, 'verify_workspace') as verify, patch.object(c, 'collect') as collect:
            for options in invalid:
                with self.assertRaises(ValueError):
                    c.batch('limits', 'unused-reference.json', **options)
            verify.assert_not_called()
            collect.autoscaler.update.assert_not_called()
            collect.spawn.assert_not_called()

    def test_batch_selected_pool_cap_preserves_pilot_then_remaining_protocol(self):
        for selected in (8, 16):
            batch_id = f'cap-{selected}'
            events = []
            updates = []
            cloud = types.SimpleNamespace(autoscaler=types.SimpleNamespace(update=lambda **kw: updates.append(kw)))

            def wave(jobs, manifest, local, parallelism, job_seconds):
                events.append(('collect', len(jobs), parallelism, job_seconds))
                for job in jobs:
                    job['status'] = 'accepted'

            def merge(manifest, local, jobs, expected):
                events.append(('merge', len(jobs), expected))

            options = {} if selected == 8 else {'parallelism': 16}
            with patch.object(c, 'ROOT', self.path), patch.object(c, 'collect', cloud), \
                    patch.object(c, 'validate_reference_result', return_value={'source_version': 'fixture'}), \
                    patch.object(t, 'verify_workspace'), patch.object(c, '_collect_wave', wave), \
                    patch.object(c, '_merge_batch', merge):
                c.batch(batch_id, 'unused-reference.json', count=30, **options)
            manifest = json.loads((self.path / 'docs/pilot/granular_batches' / batch_id / 'batch_manifest.json').read_text())
            self.assertEqual(updates, [dict(max_containers=selected, min_containers=0, buffer_containers=0)])
            self.assertEqual(events, [('collect', 5, selected, 7200), ('merge', 5, 5),
                                      ('collect', 25, selected, 7200), ('merge', 30, 30)])
            self.assertEqual(manifest['max_cpu_cores'], selected * 2)
            self.assertEqual(manifest['max_memory_gib'], selected * 12)
            self.assertTrue(manifest['resource_limits_enforced'])
            self.assertEqual(manifest['deadline_epoch'] - manifest['started_epoch'], 21600)
            self.assertEqual(manifest['retries'], 0)
            self.assertIsNone(manifest['gpu'])
            self.assertFalse(manifest['reuse_reference'])

    def test_shared_pool_decorator_enforces_cpu_memory_and_container_limits(self):
        tree = ast.parse((c.ROOT / 'scripts/modal_collect_granular.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'collect')
        values = {arg.arg: eval(compile(ast.Expression(arg.value), '<resource-limit>', 'eval'), vars(c))
                  for arg in function.decorator_list[0].keywords
                  if arg.arg in {'cpu', 'memory', 'max_containers', 'timeout', 'retries'}}
        self.assertEqual(values, dict(cpu=(2, 2), memory=(12288, 12288), max_containers=30,
                                      timeout=7200, retries=0))

    def overlap(self, manifest, spawner, parallel=30):
        with patch.object(c, 'collect', spawner), patch.object(c.time, 'sleep', lambda _: None):
            c._collect_overlap(manifest, self.path, parallel, 7200)

    def test_overlap_submits_29_before_pilot_and_merges_early_in_shared_pool(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        snapshots = []
        spawner = FakeSpawner(delays={index: 5 for index in range(4, 29)})
        original = spawner.spawn

        def spawn(*args, **kwargs):
            if 'merge_expected_episodes' in kwargs:
                snapshots.append((kwargs['merge_expected_episodes'],
                                  [job['status'] for job in manifest['jobs']], spawner.active))
            return original(*args, **kwargs)

        spawner.spawn = spawn
        self.overlap(manifest, spawner)
        self.assertEqual(len(spawner.calls), 31)
        self.assertTrue(all('merge_run_ids' not in call.kwargs for call in spawner.calls[:29]))
        self.assertEqual([s[0] for s in snapshots], [5, 30])
        self.assertEqual(snapshots[0][1][:5], ['accepted'] * 5)
        self.assertEqual(snapshots[0][1][5:], ['running'] * 25)
        self.assertEqual(snapshots[0][2], 25)
        self.assertLessEqual(spawner.peak, 30)
        self.assertTrue(all(call.done and not call.canceled for call in spawner.calls))
        self.assertEqual([row['status'] for row in manifest['merges']], ['accepted', 'accepted'])
        for call in spawner.calls[:29]:
            self.assertEqual(call.kwargs['expected_source_version'], 'sha256:expected')
            self.assertEqual(call.args[-1], 7200)

    def test_overlap_outside_pilot_failure_before_pilot_finish_preserves_pilot(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(outcomes={4: {'accepted': False, 'exit_code': 2}},
                              delays={index: 4 for index in range(4)})
        with self.assertRaises(RuntimeError):
            self.overlap(manifest, spawner)
        self.assertEqual(len(spawner.calls), 30)  #29 collections plus only pilot merge.
        self.assertEqual(manifest['jobs'][5]['status'], 'failed')
        self.assertEqual([job['status'] for job in manifest['jobs'][:5]], ['accepted'] * 5)
        self.assertEqual([(row['expected_episodes'], row['status']) for row in manifest['merges']], [(5, 'accepted')])
        self.assertTrue(json.loads((self.path / 'merge_5.json').read_text())['accepted'])
        self.assertFalse((self.path / 'merge_30.json').exists())
        self.assertTrue(all(call.done and not call.canceled for call in spawner.calls))

    def test_overlap_late_failure_does_not_remove_already_accepted_pilot(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(outcomes={28: {'accepted': False, 'exit_code': 2}}, delays={28: 6})
        with self.assertRaises(RuntimeError):
            self.overlap(manifest, spawner)
        self.assertEqual([row['expected_episodes'] for row in manifest['merges']], [5])
        self.assertEqual(manifest['merges'][0]['status'], 'accepted')
        self.assertFalse(any(call.canceled for call in spawner.calls))

    def test_overlap_pilot_failure_never_merges_incomplete_data(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(outcomes={0: {'accepted': False, 'exit_code': 2}})
        with self.assertRaises(RuntimeError):
            self.overlap(manifest, spawner)
        self.assertEqual(len(spawner.calls), 29)
        self.assertEqual(manifest['merges'], [])
        self.assertTrue(all(call.done for call in spawner.calls))

    def test_overlap_rejected_pilot_merge_is_not_retried_or_used_for_full_merge(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(outcomes={29: {'accepted': False, 'exit_code': 1}},
                              delays={index: 4 for index in range(4, 29)})
        with self.assertRaises(RuntimeError):
            self.overlap(manifest, spawner)
        self.assertEqual(len(spawner.calls), 30)
        self.assertEqual([(row['expected_episodes'], row['status']) for row in manifest['merges']], [(5, 'failed')])
        self.assertTrue(all(call.done and not call.canceled for call in spawner.calls))

    def test_overlap_shared_cap_counts_active_merge_with_collectors(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(delays={index: 2 for index in range(31)})
        self.overlap(manifest, spawner, parallel=3)
        self.assertLessEqual(spawner.peak, 3)
        self.assertEqual(len(spawner.calls), 31)
        self.assertEqual([row['expected_episodes'] for row in manifest['merges']], [5, 30])

    def test_overlap_deadline_cancels_own_collections_and_merge(self):
        manifest = self.manifest(30)
        manifest['jobs'][0]['status'] = 'accepted'
        spawner = FakeSpawner(delays={index: 20 for index in range(4, 30)})
        loops = 0

        def sleep(_):
            nonlocal loops
            loops += 1
            if loops == 2:
                manifest['deadline_epoch'] = time.time() - 1

        with patch.object(c, 'collect', spawner), patch.object(c.time, 'sleep', sleep), self.assertRaises(TimeoutError):
            c._collect_overlap(manifest, self.path, 30, 7200)
        self.assertEqual(len(spawner.calls), 30)
        self.assertTrue(all(call.canceled for call in spawner.calls[4:]))
        self.assertFalse(any(call.canceled for call in spawner.calls[:4]))
        self.assertEqual(manifest['merges'][0]['status'], 'canceled')

    def test_batch_overlap_is_explicit_and_records_30_container_resource_cap(self):
        cloud = types.SimpleNamespace(autoscaler=types.SimpleNamespace(update=lambda **kw: None))
        with patch.object(c, 'ROOT', self.path), patch.object(c, 'collect', cloud), \
                patch.object(c, 'validate_reference_result', return_value={'source_version': 'fixture'}), \
                patch.object(t, 'verify_workspace'), patch.object(c, '_collect_overlap') as overlap, \
                patch.object(c, '_collect_wave') as wave:
            c.batch('overlap', 'unused', count=30, parallelism=30, overlap_waves=True)
            overlap.assert_called_once()
            wave.assert_not_called()
            with self.assertRaises(ValueError):
                c.batch('bad-overlap', 'unused', count=5, overlap_waves=True)
        manifest = json.loads((self.path / 'docs/pilot/granular_batches/overlap/batch_manifest.json').read_text())
        self.assertEqual(manifest['schedule'], 'overlapping_after_reference')
        self.assertTrue(manifest['overlap_waves'])
        self.assertEqual(manifest['max_cpu_cores'], 60)
        self.assertEqual(manifest['max_memory_gib'], 360)
        self.assertEqual(manifest['deadline_epoch'] - manifest['started_epoch'], 21600)

    def continuation_fixture(self):
        import numpy as np
        source_hashes = {'scripts/collect_granular_demos.py': 'b' * 64}
        current = 'sha256:' + hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
        parent_hashes = {'scripts/collect_granular_demos.py': 'a' * 64}
        parent_version = 'sha256:' + hashlib.sha256(json.dumps(parent_hashes, sort_keys=True).encode()).hexdigest()
        directory = self.path / 'docs/pilot/physics_validation'
        directory.mkdir(parents=True)
        ancestor_id = 'older-diagnostic'
        ancestor_snapshot = directory / f'{ancestor_id}.final.npz'
        ctrl = np.zeros(14)
        ctrl[6] = 255
        arrays = dict(qpos=np.zeros(1459), qvel=np.zeros(1255), ctrl=ctrl)
        np.savez(ancestor_snapshot, **arrays)
        ancestor_result = directory / f'{ancestor_id}.remote.json'
        ancestor_result.write_text(json.dumps(dict(run_id=ancestor_id, accepted=False, exit_code=1, timed_out=False)))
        ancestor_sources = directory / f'{ancestor_id}.local-source-snapshot.json'
        ancestor_sources.write_text(json.dumps(dict(run_id=ancestor_id, source_version=parent_version,
                                                    source_files_sha256=parent_hashes, remote_source_verified=False)))
        artifact = directory / 'physical-regrasp'
        artifact.mkdir()
        snapshot = artifact / 'final.npz'
        np.savez(snapshot, **arrays)
        outcome = artifact / 'outcome.json'
        initial = dict(success=False, physical_valid=True, continuation_passed=True, failure='', grain_count=200,
                       grain_radius_m=.0015, physics_checks=5000, max_grain_contact_penetration_m=.00002,
                       conservation_error_g=0., spilled_g=0., source_g=4., tool_g=0., mass_g=0., inflight_g=0., total_mass_g=4.,
                       physical_validation_scope='rigid_bodies_and_grains', powder_dynamics='physical_grains_v1',
                       no_state_writes_after_restore=True, scope='Diagnostic suffix; not a from-reset rollout',
                       source_snapshot_sha256=hashlib.sha256(ancestor_snapshot.read_bytes()).hexdigest(), source_snapshot_unchanged=True,
                       powder_id=0, seed=None, target_g=.5,
                       stages=[dict(stage='regrasped', source=[.02,.08,.85], left_tool=[.02,.08,.88], failure=None),
                               dict(stage='lifted_regrasped_source', source=[.02,.08,.91], left_tool=[.02,.08,.94], failure=None)])
        outcome.write_text(json.dumps(initial))
        (artifact / 'fixture.mjb').write_bytes(b'native model fixture')
        (artifact / 'physical_probe.py').write_text('# Physical actuator-only suffix fixture\n')
        artifact_manifest = artifact / 'sha256_manifest.json'
        artifact_manifest.write_text(json.dumps({name:hashlib.sha256((artifact/name).read_bytes()).hexdigest()
                                                for name in ('final.npz','outcome.json','fixture.mjb','physical_probe.py')}))
        parent_id = 'failed-position-guard'
        parent = dict(run_id=parent_id, accepted=False, exit_code=1, timed_out=False, dataset_written=False,
                      source_version=parent_version, source_files_sha256=parent_hashes,
                      outcome=dict(failure='loading depth miss: fixture', powder_id=0, seed=1001, target_g=.5, grain_count=200))
        motivation = directory / f'{parent_id}.remote.json'
        motivation.write_text(json.dumps(parent))
        script = self.path / 'scripts/collect_granular_demos.py'
        script.parent.mkdir()
        script.write_text('DEPTH_TOLERANCE_M = 0.00025\nLATERAL_TOLERANCE_M = 0.001\n')
        payload = dict(scope='saved_state_corrective_continuation', full_episode_accepted=False,
                       candidate_data_eligible=False, stage_physical_valid=True, correction_passed=True,
                       lift_passed=True, pour_passed=True, failure=None, grain_count=200,
                       no_state_writes_after_initialization=True, initialization_contact_model_override=None,
                       powder_id=0, seed=None, target_g=.5, source_version=current,
                       source_files_sha256=source_hashes,
                       motivation=dict(run_id=parent_id,result_path=str(motivation.relative_to(self.path)),result_sha256=hashlib.sha256(motivation.read_bytes()).hexdigest()),
                       stage_summary=dict(physics_checks=5000, grain_radius_m=.0015,
                                          max_grain_contact_penetration_m=.0001, conservation_error_g=0.,
                                          spilled_g=0., transferred_g=.38, mass_g=.38, raw_net_mass_g=.379,
                                          depth_error_m=.0001, lateral_error_m=.0004,
                                          depth_tolerance_m=.00025, lateral_tolerance_m=.001,
                                          settled=True, finite_state=True, forbidden_contact_failure=False,
                                          rigid_contact_valid=True))
        evidence = dict(checks=dict(source_held=True,tool_on_bench=True,finite_state=True,physics_valid=True,
                                    right_tool_held=False,bench_support_gap_m=0.))
        for name,file in dict(snapshot=snapshot,outcome=outcome,manifest=artifact_manifest,ancestor_snapshot=ancestor_snapshot,
                              ancestor_result=ancestor_result,ancestor_source_manifest=ancestor_sources).items():
            evidence[name+'_path']=str(file.relative_to(self.path))
            evidence[name+'_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
        payload['initial_state_provenance'] = evidence
        path = self.path / 'stage.json'
        path.write_text(json.dumps(payload))
        return path, payload, (current, source_hashes), snapshot

    def test_continuation_gate_preserves_partial_scope_and_independent_force_evidence(self):
        path, payload, source, _ = self.continuation_fixture()
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source):
            gate = c.validate_stage_validation(path)
        self.assertEqual(gate['kind'], 'saved_state_corrective_continuation')
        self.assertFalse(gate['full_episode_accepted'])
        self.assertFalse(gate['candidate_data_eligible'])
        self.assertEqual(gate['accepted_demonstrations_from_gate'], 0)
        self.assertEqual(gate['stage_summary']['mass_g'], .38)  #Does not require the full .5g target.
        self.assertIsNone(gate['initial_state_provenance']['initial_seed'])
        self.assertFalse(gate['initial_state_provenance']['ancestor_remote_source_verified'])
        self.assertEqual(gate['initial_state_provenance']['ancestor_run_id'], 'older-diagnostic')
        self.assertEqual(gate['motivation']['run_id'], 'failed-position-guard')

    def test_candidate_rejects_incomplete_failed_or_mislabeled_continuation_before_cloud(self):
        path, payload, source, _ = self.continuation_fixture()
        changes = [('scope', 'full_reference'), ('correction_passed', False), ('lift_passed', False),
                   ('pour_passed', False), ('full_episode_accepted', True), ('candidate_data_eligible', True),
                   ('stage_physical_valid', False), ('failure', 'contact failure'), ('grain_count', 400),
                   ('no_state_writes_after_initialization',False),('initialization_contact_model_override',{'pad_condim':4})]
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source), \
                patch.object(t, 'verify_workspace') as verify, patch.object(c, 'collect') as cloud:
            for key, value in changes:
                with self.subTest(key=key):
                    path.write_text(json.dumps({**payload, key: value}))
                    with self.assertRaises(ValueError):
                        c.candidate_batch('candidate', str(path))
            verify.assert_not_called()
            cloud.spawn.assert_not_called()
            cloud.autoscaler.update.assert_not_called()

    def test_continuation_rejects_loose_guards_transient_force_spill_and_insufficient_transfer(self):
        path, payload, source, _ = self.continuation_fixture()
        changes = [('physics_checks', 0), ('max_grain_contact_penetration_m', .00031),
                   ('conservation_error_g', .0001), ('spilled_g', .02), ('transferred_g', .05),
                   ('raw_net_mass_g', .1), ('depth_error_m', .00026), ('lateral_error_m', .0011),
                   ('depth_tolerance_m', .0005), ('lateral_tolerance_m', .002), ('settled', False),
                   ('finite_state', False), ('rigid_contact_valid', False), ('forbidden_contact_failure', True)]
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source):
            for key, value in changes:
                with self.subTest(key=key):
                    path.write_text(json.dumps({**payload, 'stage_summary': {**payload['stage_summary'], key: value}}))
                    with self.assertRaises(ValueError):
                        c.validate_stage_validation(path)
            path.write_text(json.dumps(payload))
            (self.path / 'scripts/collect_granular_demos.py').write_text('DEPTH_TOLERANCE_M = .0005\nLATERAL_TOLERANCE_M = .001\n')
            with self.assertRaises(ValueError):
                c.validate_stage_validation(path)

    def test_candidate_requires_current_source_and_unchanged_parent_snapshot(self):
        path, payload, source, snapshot = self.continuation_fixture()
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=('sha256:changed', {})), \
                patch.object(t, 'verify_workspace') as verify:
            with self.assertRaises(ValueError):
                c.candidate_batch('candidate', str(path))
            verify.assert_not_called()
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source):
            snapshot.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                c.validate_stage_validation(path)

    def test_initial_provenance_rejects_wrong_hold_support_or_invented_seed(self):
        path, payload, source, _ = self.continuation_fixture()
        checks = payload['initial_state_provenance']['checks']
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source):
            for key,value in [('source_held',False),('tool_on_bench',False),('physics_valid',False),
                              ('right_tool_held',True),('bench_support_gap_m',.002)]:
                with self.subTest(key=key):
                    changed = {**payload,'initial_state_provenance':{**payload['initial_state_provenance'],'checks':{**checks,key:value}}}
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError): c.validate_stage_validation(path)
            path.write_text(json.dumps({**payload,'seed':1001}))
            with self.assertRaises(ValueError): c.validate_stage_validation(path)

    def test_initial_provenance_rejects_repo_escape_and_bad_motivation_identity(self):
        path, payload, source, _ = self.continuation_fixture()
        with tempfile.TemporaryDirectory() as external, patch.object(c, 'ROOT', self.path), \
                patch.object(c, '_source_identity', return_value=source):
            file = Path(external) / 'outside.json'
            file.write_text('{}')
            changed = {**payload,'initial_state_provenance':{**payload['initial_state_provenance'],
                       'outcome_path':str(file),'outcome_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}}
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): c.validate_stage_validation(path)
            path.write_text(json.dumps({**payload,'motivation':{**payload['motivation'],'run_id':'wrong-parent'}}))
            with self.assertRaises(ValueError): c.validate_stage_validation(path)

    def test_initial_provenance_rejects_nonfinite_state_even_with_consistent_file_hashes(self):
        import numpy as np
        path, payload, source, snapshot = self.continuation_fixture()
        with np.load(snapshot) as values: arrays = {key:values[key].copy() for key in values.files}
        arrays['qvel'][5] = np.nan
        np.savez(snapshot, **arrays)
        evidence = payload['initial_state_provenance']
        evidence['snapshot_sha256'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        manifest_path = self.path / evidence['manifest_path']
        inventory = json.loads(manifest_path.read_text())
        inventory['final.npz'] = evidence['snapshot_sha256']
        manifest_path.write_text(json.dumps(inventory))
        evidence['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload))
        with patch.object(c, 'ROOT', self.path), patch.object(c, '_source_identity', return_value=source):
            with self.assertRaises(ValueError): c.validate_stage_validation(path)

    def test_initial_provenance_rejects_native_physics_changes_from_recorded_ancestor(self):
        _, payload, _, _ = self.continuation_fixture()
        payload['source_files_sha256']['chemlab_rl/granular_env.py'] = 'c' * 64
        with patch.object(c, 'ROOT', self.path):
            with self.assertRaises(ValueError): c._validate_initial_state_provenance(payload)

    def chain_fixture(self):
        import tarfile
        import numpy as np
        path, payload, _, initial_snapshot = self.continuation_fixture()
        old_move = '''def _move_cup(demo, env, hold, target, rotation=None):
    count = 1
    for index in range(1, count + 1):
        tip = target
        mat = rotation
        rot = mat.reshape(3, 3)
        demo.scene.step(None)
        if env.scene.bad:
            return False
    error = float(np.linalg.norm(legacy._tip(env.data) - target))
    if error > 0.004:
        env.scene.bad = f"scoop tracking error {error * 1000:.2f} mm"
        return False
    return True
'''
        new_move = '''def _move_cup(demo, env, hold, target, rotation=None):
    count = 1
    def command_pose(tip, rot):
        demo.scene.step(None)
        return not env.scene.bad
    for index in range(1, count + 1):
        tip = target
        mat = rotation
        if not command_pose(tip, mat.reshape(3, 3)):
            return False
    error = float(np.linalg.norm(legacy._tip(env.data) - target))
    if error > 0.004:
        for _ in range(5):
            if not command_pose(target, rotation):
                return False
            error = float(np.linalg.norm(legacy._tip(env.data) - target))
            if error <= 0.001:
                break
    if error > 0.004:
        env.scene.bad = f"scoop tracking error {error * 1000:.2f} mm"
        return False
    return True
'''
        prelude = 'DEPTH_TOLERANCE_M = .00025\nLATERAL_TOLERANCE_M = .001\ndef _load_batch():\n    return True\n'
        old_source, new_source = (prelude + old_move).encode(), (prelude + new_move).encode()
        source_name = 'scripts/collect_granular_demos.py'
        old_hashes = {source_name: hashlib.sha256(old_source).hexdigest()}
        new_hashes = {source_name: hashlib.sha256(new_source).hexdigest()}
        version = lambda h: 'sha256:' + hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
        payload.update(scope='saved_state_chained_corrective_continuation_v1', source_version=version(new_hashes), source_files_sha256=new_hashes)
        (self.path / source_name).write_bytes(new_source)
        with np.load(initial_snapshot) as z: initial_arrays = {k:z[k].copy() for k in z.files}
        np.savez(initial_snapshot, **initial_arrays, time=.8)
        evidence = payload['initial_state_provenance']
        evidence['snapshot_sha256'] = hashlib.sha256(initial_snapshot.read_bytes()).hexdigest()
        initial_manifest = self.path / evidence['manifest_path']
        inventory = json.loads(initial_manifest.read_text()); inventory['final.npz'] = evidence['snapshot_sha256']
        initial_manifest.write_text(json.dumps(inventory)); evidence['manifest_sha256'] = hashlib.sha256(initial_manifest.read_bytes()).hexdigest()
        chain_dir = self.path / 'chain'; chain_dir.mkdir()
        boundary = chain_dir / 'loaded_and_lifted.npz'
        arrays = {**initial_arrays, 'ctrl':np.full(14,255.), 'qacc_warmstart':np.zeros(1255),
                  'integration_state':np.zeros(6811), 'integration_state_spec':np.array(16383),
                  'q_cmd_left':np.zeros(6),'q_cmd_right':np.zeros(6),'grip_cmd_left':np.array(255.),'grip_cmd_right':np.array(255.)}
        np.savez(boundary, **arrays, time=48.3)
        endpoint = chain_dir / 'post_pour_settled.npz'; np.savez(endpoint, **arrays, time=70.)
        cut = dict(stage='loaded_and_lifted', sim_seconds=47.5, failure=None, source_g=3.84, tool_g=.16, receiver_g=0.,inflight_g=0.,spill_g=0.)
        failure = 'scoop IK miss 0.0146 m/0.011 rad'
        common = dict(grain_count=200,powder_id=0,seed=None,target_g=.5,physical_validation_scope='rigid_bodies_and_grains',
                      powder_dynamics='physical_grains_v1',physics_validation='rigid_contact_v1',grain_radius_m=.0015,
                      grain_mass_g=.02,timestep_s=.00025,max_contact_penetration_m=.0002,
                      max_grain_contact_penetration_m=.00007,conservation_error_g=0.,total_mass_g=4.,
                      snapshot_unchanged=True,no_state_writes_after_initialization=True,initialization_contact_model_override=None)
        p = dict(**common, source_version=version(old_hashes),snapshot_sha256=evidence['snapshot_sha256'],variant='pickup_suffix',
                 correction_passed=True,lift_passed=True,pour_passed=False,continuation_passed=False,physical_valid=False,failure=failure,
                 physics_checks=220000,stages=[cut,dict(stage='first_pour_completed',failure=failure)],
                 expert_trace=[dict(phase='depth_check',sim_seconds=44.3,z_error_m=.000049,lateral_error_m=.000101),
                               dict(phase='retained_batch',sim_seconds=48.3,batch_fits=True,tool_g=.16)])
        s = dict(**common,source_version=version(new_hashes),snapshot_sha256=hashlib.sha256(boundary.read_bytes()).hexdigest(),
                 variant='receiver_transit_suffix',pour_passed=True,continuation_passed=True,physical_valid=True,failure=None,
                 physics_checks=80000,settled=True,mass_force_agreement=True,tool_g=0.,inflight_g=0.,spilled_g=0.,mass_g=.16,raw_net_mass_g=.159,
                 stages=[dict(stage='post_pour_settled',sim_seconds=21.7,failure=None)])
        chain = dict(schema_version=1)
        for role, source, hashes, outcome, input_file, output_file in [
                ('prefix',old_source,old_hashes,p,initial_snapshot,boundary),('suffix',new_source,new_hashes,s,boundary,endpoint)]:
            folder=chain_dir/role; folder.mkdir()
            probe=folder/'probe.py';probe.write_text('# Immutable executed probe fixture\n')
            manifest=folder/'source.json';manifest.write_text(json.dumps(dict(source_version=version(hashes),source_files_sha256=hashes,probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())))
            archive=folder/'source-python.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                entry=tarfile.TarInfo(source_name);entry.size=len(source);t.addfile(entry,io.BytesIO(source))
            result=folder/'outcome.json';result.write_text(json.dumps(outcome))
            chain[role]={}
            for key,file in dict(source_manifest=manifest,source_archive=archive,probe=probe,outcome=result,input_snapshot=input_file,output_snapshot=output_file).items():
                chain[role][key+'_path']=str(file.relative_to(self.path));chain[role][key+'_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            # Synthetic bytes exercise the offline contract; these fixtures
            # are never represented as captured MuJoCo physics evidence.
            captures={}
            for name,phase in [('expected_production','fresh_production_before_overrides'),
                               ('effective_initial','after_initialization_and_overrides_before_first_step'),
                               ('effective_final','after_last_step_before_cleanup')]:
                model=folder/(name+'.mjb');model.write_bytes(b'unit-test compiled native model: impratio=1')
                captures[name]=dict(capture_phase=phase,model_path=str(model.relative_to(self.path)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
            physics=folder/'effective_physics.json'
            physics.write_text(json.dumps(dict(schema_version=1,capture_scope='actual_segment_runtime',
                  serialization='mujoco_mjb_copy_zero_geom_rgba_v1',excluded_model_fields=['geom_rgba'],mujoco_version='3.16.0',
                  source_version=version(hashes),probe_sha256=chain[role]['probe_sha256'],captures=captures)))
            chain[role]['effective_physics_path']=str(physics.relative_to(self.path))
            chain[role]['effective_physics_sha256']=hashlib.sha256(physics.read_bytes()).hexdigest()
            outcome.update(effective_physics_sha256=chain[role]['effective_physics_sha256'],
                           checkpoint_sha256={output_file.name:chain[role]['output_snapshot_sha256']})
            result.write_text(json.dumps(outcome));chain[role]['outcome_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        audit=chain_dir/'boundary.json';audit.write_text(json.dumps(dict(snapshot_sha256=chain['prefix']['output_snapshot_sha256'],source_version=version(new_hashes),
              integration_state_restored_exactly=True,controller_state_restored_exactly=True,finite_state=True,source_held=True,right_tool_held=True,
              receiver_empty_before_tare=True,integration_state_size_valid=True,snapshot_redundant_fields_match_integration_state=True,
              normalized_model_sha256=hashlib.sha256(b'unit-test compiled native model: impratio=1').hexdigest(),
              source_g=3.84,tool_g=.16,receiver_g=0.,inflight_g=0.,spilled_g=0.,total_mass_g=4.,conservation_error_g=0.)))
        chain.update(boundary_audit_path=str(audit.relative_to(self.path)),boundary_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
        payload['continuation_chain']=chain
        payload['stage_summary'].update(physics_checks=300000,physics_checks_scope='prefix_run_including_excluded_tail_plus_suffix_upper_bound',
              max_grain_contact_penetration_m=.00007,transferred_g=.16,mass_g=.16,raw_net_mass_g=.159,depth_error_m=.000049,lateral_error_m=.000101)
        path.write_text(json.dumps(payload))
        return path,payload,(version(new_hashes),new_hashes),old_source,new_source

    def test_chained_partial_gate_preserves_failed_tail_and_counts_zero_demonstrations(self):
        path,payload,source,_,_=self.chain_fixture()
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            gate=c.validate_stage_validation(path)
        self.assertEqual(gate['accepted_demonstrations_from_gate'],0)
        self.assertFalse(gate['candidate_data_eligible'])
        self.assertEqual(gate['continuation_chain']['prefix_cut'],'loaded_and_lifted')
        self.assertTrue(gate['continuation_chain']['prefix_later_failure_preserved'].startswith('scoop IK miss'))
        self.assertEqual(gate['continuation_chain']['source_compatibility'],'normal_path_ast_equal_old_success_implies_retry_inactive')
        self.assertIn('upper_bound',gate['stage_summary']['physics_checks_scope'])

    def test_chained_gate_rejects_state_splice_tamper_missing_archive_and_false_exact_count(self):
        import copy
        path,payload,source,_,_=self.chain_fixture()
        variants=[]
        bad=copy.deepcopy(payload);bad['continuation_chain']['suffix']['input_snapshot_sha256']='0'*64;variants.append(bad)
        bad=copy.deepcopy(payload);bad['continuation_chain']['prefix']['source_archive_sha256']='0'*64;variants.append(bad)
        bad=copy.deepcopy(payload);bad['stage_summary']['physics_checks_scope']='exact_prefix_plus_suffix';variants.append(bad)
        bad=copy.deepcopy(payload);bad['stage_summary']['physics_checks']=299999;variants.append(bad)
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for bad in variants:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_chained_gate_rejects_failed_unsettled_spilled_or_incomplete_suffix_even_with_updated_file_hash(self):
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['suffix'];file=self.path/record['outcome_path'];original=json.loads(file.read_text())
        cases=[('pour_passed',False),('failure','deep grain contact'),('settled',False),('spilled_g',.02),('tool_g',.02),
               ('inflight_g',.02),('mass_g',.06),('raw_net_mass_g',.1),('max_grain_contact_penetration_m',.000656),
               ('initialization_contact_model_override',{'pad_condim':4}),('source_version','sha256:wrong')]
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for key,value in cases:
                with self.subTest(key=key):
                    file.write_text(json.dumps({**original,key:value}));record['outcome_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_chained_gate_rejects_prefix_relabel_or_earlier_failure(self):
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['prefix'];file=self.path/record['outcome_path'];original=json.loads(file.read_text())
        cases=[{**original,'continuation_passed':True},{**original,'physical_valid':True},
               {**original,'stages':[{**original['stages'][0],'failure':'forbidden contact'},original['stages'][1]]},
               {**original,'stages':[{**original['stages'][0],'sim_seconds':48.3},original['stages'][1]]}]
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for bad in cases:
                file.write_text(json.dumps(bad));record['outcome_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_chained_gate_rejects_false_restore_and_nonempty_tare_audit(self):
        path,payload,source,_,_=self.chain_fixture()
        chain=payload['continuation_chain'];file=self.path/chain['boundary_audit_path'];original=json.loads(file.read_text())
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for key,value in [('integration_state_restored_exactly',False),('controller_state_restored_exactly',False),
                              ('integration_state_size_valid',False),('snapshot_redundant_fields_match_integration_state',False),
                              ('receiver_empty_before_tare',False),('receiver_g',.02),('source_g',3.8)]:
                file.write_text(json.dumps({**original,key:value}));chain['boundary_audit_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_prefix_ast_equivalence_rejects_guard_changes_and_behavior_before_boundary(self):
        _,_,_,old,new=self.chain_fixture()
        self.assertEqual(c._verify_prefix_controller_compatibility(old,old),'identical_prefix_controller')
        for changed in [new.replace(b'range(5)',b'range(6)'),new.replace(b'error > 0.004',b'error > 0.005'),
                        new.replace(b'count = 1',b'count = 2'),new.replace(b'return not env.scene.bad',b'return True'),
                        new.replace(b'def _load_batch():\n    return True',b'def _load_batch():\n    return False')]:
            with self.assertRaises(ValueError):c._verify_prefix_controller_compatibility(old,changed)

    def test_chained_gate_rejects_changed_native_asset_even_with_self_consistent_source_hashes(self):
        _,payload,_,_,_=self.chain_fixture()
        payload['source_files_sha256']['native/model.xml']='a'*64
        for role in ('prefix','suffix'):
            record=payload['continuation_chain'][role];file=self.path/record['source_manifest_path'];manifest=json.loads(file.read_text())
            manifest['source_files_sha256']['native/model.xml']=('b' if role=='prefix' else 'a')*64
            manifest['source_version']='sha256:'+hashlib.sha256(json.dumps(manifest['source_files_sha256'],sort_keys=True).encode()).hexdigest()
            file.write_text(json.dumps(manifest));record['source_manifest_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
        with patch.object(c,'ROOT',self.path),self.assertRaisesRegex(ValueError,'identical physics'):
            c._validate_continuation_chain(payload)

    def test_chained_gate_rejects_incomplete_native_final_state_even_with_updated_snapshot_hash(self):
        import numpy as np
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['suffix'];file=self.path/record['output_snapshot_path']
        with np.load(file) as saved:arrays={key:saved[key].copy() for key in saved.files}
        arrays['integration_state']=arrays['integration_state'][:-1]
        np.savez(file,**arrays);record['output_snapshot_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        result_path=self.path/record['outcome_path'];result=json.loads(result_path.read_text());result['checkpoint_sha256'][file.name]=record['output_snapshot_sha256']
        result_path.write_text(json.dumps(result));record['outcome_sha256']=hashlib.sha256(result_path.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source),self.assertRaisesRegex(ValueError,'snapshot is incomplete'):
            c.validate_stage_validation(path)

    def test_chained_gate_rejects_missing_or_backdated_effective_model_capture(self):
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['prefix'];file=self.path/record['effective_physics_path'];original=json.loads(file.read_text())
        variants=[{**original,'capture_scope':'reconstructed_after_execution'},
                  {**original,'excluded_model_fields':['geom_rgba','geom_friction']},
                  {**original,'captures':{k:v for k,v in original['captures'].items() if k!='effective_final'}}]
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for value in variants:
                file.write_text(json.dumps(value));record['effective_physics_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
                outcome_file=self.path/record['outcome_path'];outcome=json.loads(outcome_file.read_text());outcome['effective_physics_sha256']=record['effective_physics_sha256']
                outcome_file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(outcome_file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_runtime_probe_override_rejected_despite_updated_probe_and_source_manifest_hashes(self):
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['suffix'];probe=self.path/record['probe_path']
        probe.write_text('env.model.opt.impratio = 10\n');record['probe_sha256']=hashlib.sha256(probe.read_bytes()).hexdigest()
        manifest_file=self.path/record['source_manifest_path'];manifest=json.loads(manifest_file.read_text());manifest['probe_sha256']=record['probe_sha256']
        manifest_file.write_text(json.dumps(manifest));record['source_manifest_sha256']=hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        physics_file=self.path/record['effective_physics_path'];physics=json.loads(physics_file.read_text());physics['probe_sha256']=record['probe_sha256']
        for name in ('effective_initial','effective_final'):
            capture=physics['captures'][name];file=self.path/capture['model_path'];file.write_bytes(b'unit-test compiled native model: impratio=10');capture['model_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
        physics_file.write_text(json.dumps(physics));record['effective_physics_sha256']=hashlib.sha256(physics_file.read_bytes()).hexdigest()
        outcome_file=self.path/record['outcome_path'];outcome=json.loads(outcome_file.read_text());outcome['effective_physics_sha256']=record['effective_physics_sha256']
        outcome_file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(outcome_file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source),self.assertRaisesRegex(ValueError,'Runtime physics/model'):
            c.validate_stage_validation(path)

    def test_runtime_pour_helper_override_rejected_despite_compatible_prefix_ast_and_updated_source(self):
        import tarfile
        path,payload,_,old,new=self.chain_fixture()
        new+=b'\ndef _tilt_complete_batch(env):\n    env.model.opt.impratio = 10\n'
        c._verify_prefix_controller_compatibility(old,new)  # The mutation is after the saved prefix cut.
        source_name='scripts/collect_granular_demos.py';(self.path/source_name).write_bytes(new)
        hashes={source_name:hashlib.sha256(new).hexdigest()};version='sha256:'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        payload.update(source_version=version,source_files_sha256=hashes)
        record=payload['continuation_chain']['suffix'];manifest_file=self.path/record['source_manifest_path'];manifest=json.loads(manifest_file.read_text())
        manifest.update(source_version=version,source_files_sha256=hashes);manifest_file.write_text(json.dumps(manifest));record['source_manifest_sha256']=hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        archive=self.path/record['source_archive_path']
        with tarfile.open(archive,'w:gz') as t:
            entry=tarfile.TarInfo(source_name);entry.size=len(new);t.addfile(entry,io.BytesIO(new))
        record['source_archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
        physics_file=self.path/record['effective_physics_path'];physics=json.loads(physics_file.read_text());physics['source_version']=version
        capture=physics['captures']['effective_final'];file=self.path/capture['model_path'];file.write_bytes(b'unit-test compiled native model: impratio=10');capture['model_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
        physics_file.write_text(json.dumps(physics));record['effective_physics_sha256']=hashlib.sha256(physics_file.read_bytes()).hexdigest()
        outcome_file=self.path/record['outcome_path'];outcome=json.loads(outcome_file.read_text());outcome.update(source_version=version,effective_physics_sha256=record['effective_physics_sha256'])
        outcome_file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(outcome_file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=(version,hashes)),self.assertRaisesRegex(ValueError,'Runtime physics/model'):
            c.validate_stage_validation(path)

    def test_settled_checkpoint_time_must_match_bound_executed_stage(self):
        path,payload,source,_,_=self.chain_fixture()
        record=payload['continuation_chain']['suffix'];file=self.path/record['outcome_path'];outcome=json.loads(file.read_text())
        outcome['stages'][0]['sim_seconds']=20.
        file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source),self.assertRaisesRegex(ValueError,'snapshot is incomplete'):
            c.validate_stage_validation(path)

    def loaded_transfer_fixture(self):
        path,payload,source,_,_=self.chain_fixture()
        chain=payload.pop('continuation_chain');payload.pop('initial_state_provenance');payload.pop('motivation');payload.pop('stage_summary')
        payload.update(scope='saved_state_loaded_transfer_feasibility_v1',historical_prefix_validated=False,
                       accepted_demo_count=0,correction_passed=False,lift_passed=False)
        record=chain['suffix'];record.update(boundary_audit_path=chain['boundary_audit_path'],boundary_audit_sha256=chain['boundary_audit_sha256'])
        payload['transfer_segment']=record
        file=self.path/record['outcome_path'];outcome=json.loads(file.read_text())
        outcome.update(source_g=3.84,physics_checks=90000,
            stages=[dict(stage='initial',sim_seconds=0.,failure=None,source_g=3.84,tool_g=.16,receiver_g=0.,inflight_g=0.,spill_g=0.),
                    dict(stage='post_pour_settled',sim_seconds=21.7,failure=None,source_g=3.84,tool_g=0.,receiver_g=.16,inflight_g=0.,spill_g=0.)],
            expert_trace=[dict(phase=p) for p in ('receiver_transit','pre_pour_guard','complete_pour','drain_hold','return_upright','receiver_retract')])
        file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
        return path,payload,source

    def test_loaded_transfer_accepts_only_forward_scope_without_a_historical_prefix_claim(self):
        path,payload,source=self.loaded_transfer_fixture()
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            gate=c.validate_stage_validation(path)
        self.assertEqual(gate['kind'],'saved_state_loaded_transfer_feasibility_v1')
        self.assertFalse(gate['historical_prefix_validated'])
        self.assertFalse(gate['full_episode_accepted'])
        self.assertFalse(gate['candidate_data_eligible'])
        self.assertEqual(gate['accepted_demo_count'],0)
        self.assertEqual(gate['stage_summary']['physics_checks_scope'],'forward_segment_only')
        self.assertNotIn('initial_state_provenance',gate)
        self.assertNotIn('continuation_chain',gate)

    def test_loaded_transfer_rejects_missing_evidence_prefix_claim_or_changed_physics(self):
        import copy
        path,payload,source=self.loaded_transfer_fixture()
        changes=[('historical_prefix_validated',True),('accepted_demo_count',1),('correction_passed',True),
                 ('lift_passed',True),('full_episode_accepted',True),('candidate_data_eligible',True),('stage_physical_valid',False)]
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for key,value in changes:
                path.write_text(json.dumps({**payload,key:value}))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)
            bad=copy.deepcopy(payload);bad['transfer_segment'].pop('effective_physics_sha256');path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):c.validate_stage_validation(path)
            record=payload['transfer_segment'];file=self.path/record['effective_physics_path'];physics=json.loads(file.read_text())
            capture=physics['captures']['effective_initial'];model=self.path/capture['model_path'];model.write_bytes(b'unit-test changed impratio=10')
            capture['model_sha256']=hashlib.sha256(model.read_bytes()).hexdigest();file.write_text(json.dumps(physics));record['effective_physics_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            result_file=self.path/record['outcome_path'];outcome=json.loads(result_file.read_text());outcome['effective_physics_sha256']=record['effective_physics_sha256']
            result_file.write_text(json.dumps(outcome));record['outcome_sha256']=hashlib.sha256(result_file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'Runtime physics/model'):c.validate_stage_validation(path)

    def test_loaded_transfer_rejects_bad_native_state_or_nonempty_initial_receiver(self):
        path,payload,source=self.loaded_transfer_fixture();record=payload['transfer_segment']
        file=self.path/record['boundary_audit_path'];original=json.loads(file.read_text())
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for key,value in [('integration_state_size_valid',False),('snapshot_redundant_fields_match_integration_state',False),
                              ('normalized_model_sha256','a'*64),('receiver_g',.02),('spilled_g',.02),('inflight_g',.02),('tool_g',.04)]:
                file.write_text(json.dumps({**original,key:value}));record['boundary_audit_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_loaded_transfer_rejects_spill_incomplete_drain_force_and_unchecked_substeps(self):
        path,payload,source=self.loaded_transfer_fixture();record=payload['transfer_segment']
        file=self.path/record['outcome_path'];original=json.loads(file.read_text())
        cases=[('spilled_g',.02),('tool_g',.02),('inflight_g',.02),('raw_net_mass_g',.1),('mass_g',.06),
               ('settled',False),('failure','forbidden contact'),('physics_checks',100),('max_grain_contact_penetration_m',.000656),
               ('max_contact_penetration_m',.0011),('no_state_writes_after_initialization',False),
               ('expert_trace',[row for row in original['expert_trace'] if row['phase']!='receiver_retract'])]
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source):
            for key,value in cases:
                with self.subTest(key=key):
                    file.write_text(json.dumps({**original,key:value}));record['outcome_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):c.validate_stage_validation(path)

    def test_loaded_transfer_candidates_start_from_reset_and_never_reuse_stage_data(self):
        path,payload,source=self.loaded_transfer_fixture()
        with patch.object(c,'ROOT',self.path),patch.object(c,'_source_identity',return_value=source), \
                patch.object(t,'verify_workspace'),patch.object(c,'_run_batch') as run:
            c.candidate_batch('transfer-candidates',str(path))
        jobs=run.call_args.args[1];gate=run.call_args.args[2]
        self.assertEqual(len(jobs),30);self.assertTrue(all(job['status']=='planned' for job in jobs))
        self.assertEqual(gate['accepted_demo_count'],0);self.assertTrue(run.call_args.kwargs['candidate_mode'])
        with self.assertRaises(TypeError):c.candidate_batch('transfer-candidates',str(path),reuse_reference=True)
        with self.assertRaises(ValueError):c._run_batch('illegal',jobs,gate,30,360,120,overlap_waves=True,candidate_mode=True,reused={'accepted':True})

    def test_candidate_submits_all_30_from_reset_with_no_reference_or_stage_reuse(self):
        gate = dict(kind='saved_state_corrective_continuation', source_version='sha256:new', full_episode_accepted=False)
        spawner = FakeSpawner()
        spawner.autoscaler = types.SimpleNamespace(update=lambda **kw: None)
        with patch.object(c, 'ROOT', self.path), patch.object(c, 'validate_stage_validation', return_value=gate), \
                patch.object(t, 'verify_workspace'), patch.object(c, 'collect', spawner), \
                patch.object(c.time, 'sleep', lambda _: None):
            c.candidate_batch('candidate', 'scoped-stage.json')
        manifest = json.loads((self.path / 'docs/pilot/granular_batches/candidate/batch_manifest.json').read_text())
        self.assertNotIn('reference_robot_result', manifest)
        self.assertTrue(manifest['candidate_mode'])
        self.assertFalse(manifest['reuse_reference'])
        self.assertEqual(manifest['validation_gate'], gate)
        self.assertEqual(manifest['expected_source_version'], 'sha256:new')
        self.assertEqual(len(spawner.calls), 32)
        self.assertEqual(spawner.peak, 30)
        self.assertTrue(all('merge_run_ids' not in call.kwargs for call in spawner.calls[:30]))
        self.assertTrue(all(call.args[5:7] == (False, False) for call in spawner.calls[:30]))
        self.assertEqual([call.kwargs['merge_expected_episodes'] for call in spawner.calls[30:]], [5, 30])
        self.assertTrue(all(call.kwargs['expected_source_version'] == 'sha256:new' for call in spawner.calls[:30]))
        with self.assertRaises(TypeError):
            c.candidate_batch('candidate', 'stage.json', reuse_reference=True)
        with self.assertRaises(ValueError):
            c._run_batch('illegal', [{'status': 'accepted'}], gate, 30, 360, 120, overlap_waves=True, candidate_mode=True)

    def test_wave_bounded_and_source_pinned(self):
        m = self.manifest()
        s = FakeSpawner(delays={0: 2, 1: 1})
        self.wave(m, s)
        self.assertLessEqual(s.peak, 2)
        self.assertEqual([j['status'] for j in m['jobs']], ['accepted'] * 5)
        self.assertTrue(all((x.kwargs['expected_source_version'] == 'sha256:expected' for x in s.calls)))

    def test_failure_drains_running_without_more_submissions(self):
        m = self.manifest()
        s = FakeSpawner(outcomes={0: {'accepted': False, 'exit_code': 2}}, delays={1: 3})
        with self.assertRaises(RuntimeError):
            self.wave(m, s)
        self.assertEqual(len(s.calls), 2)
        self.assertTrue(s.calls[1].done)
        self.assertFalse(any((x.canceled for x in s.calls)))
        self.assertEqual([j['status'] for j in m['jobs']], ['failed', 'accepted', 'not_started', 'not_started', 'not_started'])
        self.assertTrue((self.path / (m['jobs'][1]['run_id'] + '.json')).is_file())

    def test_terminal_timeout_drains_other_call(self):
        m = self.manifest()
        s = FakeSpawner(outcomes={0: c.modal.exception.FunctionTimeoutError('terminal')}, delays={1: 1})
        with self.assertRaises(RuntimeError):
            self.wave(m, s)
        self.assertEqual(len(s.calls), 2)
        self.assertEqual(m['jobs'][1]['status'], 'accepted')
        self.assertFalse(s.calls[1].canceled)

    def test_deadline_cancels_only_batch_calls(self):
        m = self.manifest()
        s = FakeSpawner(delays={0: 10, 1: 10})

        def expire(_):
            m['deadline_epoch'] = time.time() - 1
        with patch.object(c, 'collect', s), patch.object(c.time, 'sleep', expire), self.assertRaises(TimeoutError):
            c._collect_wave(m['jobs'], m, self.path, 2, 120)
        self.assertEqual(len(s.calls), 2)
        self.assertTrue(all((x.canceled for x in s.calls)))

    def test_merge_uses_shared_pool_exact_ids(self):
        m = self.manifest()
        s = FakeSpawner()
        for j in m['jobs']:
            j['status'] = 'accepted'
        with patch.object(c, 'collect', s):
            c._merge_batch(m, self.path, m['jobs'], 5)
        self.assertEqual(s.calls[0].kwargs['merge_run_ids'], [j['run_id'] for j in m['jobs']])
        self.assertEqual(s.calls[0].kwargs['merge_expected_episodes'], 5)
        m['jobs'][0]['status'] = 'failed'
        with patch.object(c, 'collect', s), self.assertRaises(ValueError):
            c._merge_batch(m, self.path, m['jobs'], 5)
        self.assertEqual(len(s.calls), 1)

    def reference(self):
        hashes = {'scripts/collect_granular_demos.py': 'a' * 64}
        version = 'sha256:' + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        outcome = dict(success=True, physical_valid=True, mass_force_agreement=True, failure=None, grain_count=200, powder_dynamics='physical_grains_v1', physical_validation_scope='rigid_bodies_and_grains', physics_checks=4, mass_g=0.5, raw_net_mass_g=0.5, conservation_error_g=0, grain_radius_m=0.0015, max_grain_contact_penetration_m=0.0002, spilled_g=0, target_g=0.5, balance_g=0.5)
        result = {'run_id': 'ref', 'exit_code': 0, 'timed_out': False, 'outcome': outcome}
        src = {'run_id': 'ref', 'source_version': version, 'source_files_sha256': hashes, 'capture_note': 'paired local snapshot'}
        (self.path / 'result.json').write_text(json.dumps(result))
        (self.path / 'source.json').write_text(json.dumps(src))
        module = types.ModuleType('scripts.collect_granular_demos')
        module.source_identity = lambda: (version, hashes)
        return (result, src, module)

    def recorded_reference(self, batch_id='test'):
        result, source, module = self.reference()
        job = c.collection_plan(batch_id)[0]
        result.update(run_id=job['run_id'], accepted=True, diagnostic=False, approach_only=False,
                      dataset_written=True, dataset_path=f"/datasets/{job['run_id']}.hdf5",
                      powder_id=job['powder'], seed=job['seed'], target_g=job['target'], grain_count=job['grains'],
                      source_version=source['source_version'], source_files_sha256=source['source_files_sha256'],
                      episode_identity=[job['powder'], job['seed'], job['target']], trajectory_sha256='b' * 64,
                      validation='physical_grain_episode_audit_v1')
        result['outcome'].update(powder_id=job['powder'], seed=job['seed'])
        path = self.path / 'recorded-reference.json'
        path.write_text(json.dumps(result))
        with patch.object(c, '_source_identity', module.source_identity):
            reference = c.validate_reference_result(path)
        return result, reference, module

    def test_reused_recorded_reference_submits_only_four_new_pilot_jobs(self):
        result, reference, _ = self.recorded_reference()
        manifest = self.manifest()
        manifest['reference_robot_result'] = reference
        manifest['expected_source_version'] = reference['source_version']
        adopted = c._reuse_reference(manifest['jobs'][0], reference)
        spawner = FakeSpawner()
        self.wave(manifest, spawner, parallel=16)
        self.assertEqual(adopted, result)
        self.assertEqual(len(spawner.calls), 4)
        self.assertEqual([call.args[0] for call in spawner.calls], [job['run_id'] for job in manifest['jobs'][1:]])
        self.assertEqual(manifest['jobs'][0]['result'], result)
        self.assertTrue(manifest['jobs'][0]['reused_reference'])
        self.assertEqual(manifest['jobs'][0]['reference_result_sha256'], reference['sha256'])
        self.assertEqual([job['status'] for job in manifest['jobs']], ['accepted'] * 5)

    def test_reference_reuse_rejects_mismatch_diagnostics_and_missing_dataset_metadata(self):
        mutations = [dict(run_id='another-run'), dict(powder_id=1), dict(seed=1002), dict(target_g=.3),
                     dict(grain_count=400), dict(source_version='sha256:other'), dict(diagnostic=True),
                     dict(approach_only=True), dict(accepted=False), dict(dataset_written=False),
                     dict(dataset_path='/datasets/other.hdf5'), dict(episode_identity=[0, 1002, .5])]
        for changes in [*mutations, {'dataset_path': None}, {'validation': None}]:
            with self.subTest(changes=changes):
                result, reference, _ = self.recorded_reference()
                result.update(changes)
                if changes.get('dataset_path', '') is None:
                    result.pop('dataset_path')
                raw = json.dumps(result).encode()
                Path(reference['path']).write_bytes(raw)
                reference['sha256'] = hashlib.sha256(raw).hexdigest()
                job = c.collection_plan('test')[0]
                with self.assertRaises(ValueError):
                    c._reuse_reference(job, reference)
                self.assertEqual(job['status'], 'planned')

    def test_reference_reuse_rejects_changed_result_or_nonfirst_job(self):
        _, reference, _ = self.recorded_reference()
        with self.assertRaises(ValueError):
            c._reuse_reference(c.collection_plan('test')[1], reference)
        Path(reference['path']).write_text('{}')
        with self.assertRaises(ValueError):
            c._reuse_reference(c.collection_plan('test')[0], reference)

    def test_wave_does_not_resume_failed_or_unknown_jobs(self):
        for status in ('failed', 'submission_unknown', 'canceled', 'running', 'not_started'):
            manifest = self.manifest()
            manifest['jobs'][0]['status'] = status
            spawner = FakeSpawner()
            with self.assertRaises(ValueError):
                self.wave(manifest, spawner)
            self.assertFalse(spawner.calls)

    def test_batch_reference_reuse_is_opt_in_and_rechecks_remote_dataset_metadata(self):
        result, reference, module = self.recorded_reference('reused-batch')
        spawner = FakeSpawner()
        spawner.autoscaler = types.SimpleNamespace(update=lambda **_: None)
        merged = []
        checked = []
        with patch.object(c, 'ROOT', self.path), patch.object(c, 'collect', spawner), \
                patch.object(c, '_source_identity', module.source_identity), patch.object(t, 'verify_workspace'), \
                patch.object(t, '_require_remote_dataset', checked.append), patch.object(c.time, 'sleep', lambda _: None), \
                patch.object(c, '_merge_batch', lambda manifest, local, jobs, expected: merged.append((len(jobs), expected, len(spawner.calls)))):
            c.batch('reused-batch', reference['path'], count=30, reuse_reference=True)
        manifest_path = self.path / 'docs/pilot/granular_batches/reused-batch/batch_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(spawner.calls), 29)
        self.assertEqual(checked, [Path(result['dataset_path']).name])
        self.assertEqual(merged, [(5, 5, 4), (30, 30, 29)])
        self.assertTrue(manifest['reuse_reference'])
        self.assertEqual(manifest['jobs'][0]['result'], result)
        saved = manifest_path.parent / f"{result['run_id']}.json"
        self.assertEqual(json.loads(saved.read_text()), result)

    def test_reuse_missing_remote_file_stops_before_autoscaler_or_submission(self):
        _, reference, module = self.recorded_reference('missing-file')
        with patch.object(c, 'collect') as cloud, patch.object(c, '_source_identity', module.source_identity), \
                patch.object(t, 'verify_workspace'), \
                patch.object(t, '_require_remote_dataset', side_effect=FileNotFoundError('reference file missing')):
            with self.assertRaises(FileNotFoundError):
                c.batch('missing-file', reference['path'], reuse_reference=True)
            cloud.autoscaler.update.assert_not_called()
            cloud.spawn.assert_not_called()

    def test_reference_requires_and_pairs_source(self):
        result, src, module = self.reference()
        with patch.object(c, '_source_identity', module.source_identity):
            got = c.validate_reference_result(self.path / 'result.json', self.path / 'source.json')
            self.assertEqual(got['source_version'], src['source_version'])
            with self.assertRaises(ValueError):
                c.validate_reference_result(self.path / 'result.json')
            src['run_id'] = 'wrong'
            (self.path / 'source.json').write_text(json.dumps(src))
            with self.assertRaises(ValueError):
                c.validate_reference_result(self.path / 'result.json', self.path / 'source.json')

    def test_reference_rejects_changed_source_and_physics(self):
        result, src, module = self.reference()
        module.source_identity = lambda: ('sha256:new', {})
        with patch.object(c, '_source_identity', module.source_identity), self.assertRaises(ValueError):
            c.validate_reference_result(self.path / 'result.json', self.path / 'source.json')
        result['outcome']['raw_net_mass_g'] = 0.52
        (self.path / 'result.json').write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            c.validate_reference_result(self.path / 'result.json', self.path / 'source.json')

    def test_external_source_cannot_override_embedded_version(self):
        result, src, module = self.reference()
        result['source_version'] = 'sha256:different-code'
        (self.path / 'result.json').write_text(json.dumps(result))
        with patch.object(c, '_source_identity', module.source_identity), self.assertRaises(ValueError):
            c.validate_reference_result(self.path / 'result.json', self.path / 'source.json')

    def test_submission_failure_drains_already_running_call(self):
        manifest = self.manifest()
        spawner = FakeSpawner(delays={0: 2})
        original_spawn = spawner.spawn

        def fail_second_submission(*args, **kwargs):
            if spawner.calls:
                raise ConnectionError('submission response lost')
            return original_spawn(*args, **kwargs)

        spawner.spawn = fail_second_submission
        with self.assertRaises(RuntimeError):
            self.wave(manifest, spawner)
        self.assertEqual(len(spawner.calls), 1)
        self.assertEqual(manifest['jobs'][0]['status'], 'accepted')
        self.assertEqual(manifest['jobs'][1]['status'], 'submission_unknown')
        self.assertFalse(spawner.calls[0].canceled)

    def test_remote_training_uses_no_local_dataset(self):
        calls = []
        checks = []
        with patch.object(t, 'verify_workspace'), patch.object(t, '_require_remote_dataset', checks.append), patch.object(t, 'execute', types.SimpleNamespace(remote=lambda *a: calls.append(a) or {'exit_code': 0})):
            t.main(dataset='/missing/local.hdf5', remote_dataset='pilot5.hdf5', run_id='new-pilot')
        self.assertEqual(checks, ['pilot5.hdf5'])
        cmd = calls[0][1]
        self.assertEqual(cmd[cmd.index('--dataset') + 1], '/datasets/pilot5.hdf5')

    def test_upload_remote_mutually_exclusive_and_basename(self):
        with patch.object(t, 'verify_workspace'), self.assertRaises(ValueError):
            t.main(remote_dataset='pilot5.hdf5', upload=True)
        for name in ('../x.hdf5', 'x/y.hdf5', '.', 'x.txt', '/x.hdf5'):
            with self.assertRaises(ValueError):
                t._dataset_basename(name)

    def test_volume_stat_rejects_directory(self):
        fake = types.SimpleNamespace(listdir=lambda _: [types.SimpleNamespace(path='pilot.hdf5', type=FileEntryType.DIRECTORY)])
        with patch.object(t, 'data_volume', fake), self.assertRaises(FileNotFoundError):
            t._require_remote_dataset('pilot.hdf5')
        fake.listdir = lambda _: [types.SimpleNamespace(path='/pilot.hdf5', type=FileEntryType.FILE)]
        with patch.object(t, 'data_volume', fake):
            t._require_remote_dataset('pilot.hdf5')

    def test_evaluate_target_seed(self):
        calls = []
        with patch.object(t, 'verify_workspace'), patch.object(t, 'execute_evaluation', types.SimpleNamespace(remote=lambda *a: calls.append(a) or {})):
            t.evaluate('run', 100)
            t.evaluate('run', 100, target=0.7, seed=123)
        for command, target, seed in ((calls[0][1], '.5', '10000'), (calls[1][1], '.7', '123')):
            self.assertEqual(float(command[command.index('--targets') + 1]), float(target))
            self.assertEqual(command[command.index('--seed') + 1], seed)
        with patch.object(t, 'verify_workspace'), self.assertRaises(ValueError):
            t.evaluate('run', 100, target=float('nan'))

    def test_full_evaluation_defaults_and_limits(self):
        calls = []
        with patch.object(t, 'verify_workspace'), patch.object(t.time, 'time', return_value=1000), \
                patch.object(t, 'execute_evaluation', types.SimpleNamespace(remote=lambda *args: calls.append(args) or {})):
            t.evaluate('run', 100)
            for invalid in ({'max_minutes': 121}, {'max_steps': 5001}, {'max_minutes': 4}, {'max_steps': 0}):
                with self.assertRaises(ValueError):
                    t.evaluate('run', 100, **invalid)
        self.assertEqual(len(calls), 1)
        command = calls[0][1]
        self.assertEqual(command[command.index('--max-steps') + 1], '5000')
        self.assertEqual(command[command.index('--max-seconds') + 1], '7080')
        self.assertEqual(command[command.index('--episodes-per-powder') + 1], '1')
        self.assertEqual(calls[0][2], 8200)

    def test_evaluation_hard_deadline_and_single_container_limit(self):
        command = ['-m', 'scripts.evaluate_vla', '--max-steps', '5000', '--max-seconds', '7080']
        with patch.object(t.time, 'time', return_value=1000), patch.object(t, '_execute_command', return_value={}) as execute:
            t._execute_evaluation('bounded-eval', command, 99999)
            self.assertEqual(execute.call_args.args[2], 8185)
            self.assertEqual(execute.call_args.kwargs['termination_grace_seconds'], 5)
            t._execute_evaluation('bounded-eval', command, 1100)
            self.assertEqual(execute.call_args.args[2], 1085)
            with self.assertRaises(TimeoutError):
                t._execute_evaluation('bounded-eval', command, 1010)
            with self.assertRaises(ValueError):
                t._execute_evaluation('bounded-eval', command[:-1] + ['7201'], 99999)
        tree = ast.parse((t.ROOT / 'scripts/modal_train.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute_evaluation')
        values = {arg.arg: ast.literal_eval(arg.value) for arg in function.decorator_list[0].keywords
                  if arg.arg in {'timeout', 'max_containers', 'retries', 'gpu'}}
        self.assertEqual(values, dict(timeout=7200, max_containers=1, retries=0, gpu='A100-80GB'))

    def test_evaluation_completion_labels_do_not_treat_exit_zero_as_a_finished_episode(self):
        report = self.path / 'evaluation.json'
        self.assertEqual(t._evaluation_completion(self.path, exit_code=0, timed_out=False)['status'], 'incomplete')
        report.write_text('{')
        self.assertEqual(t._evaluation_completion(self.path, exit_code=0, timed_out=False)['status'], 'incomplete')
        for completed, reason in ((False, 'wall_time_limit'), (True, 'success'), (True, 'step_limit')):
            report.write_text(json.dumps(dict(completed_episodes=int(completed), episodes=[
                dict(completed_evaluation=completed, termination_reason=reason)])))
            result = t._evaluation_completion(self.path, exit_code=0, timed_out=False)
            self.assertEqual(result['status'], 'completed' if completed else 'incomplete')
            self.assertEqual(result['termination_reasons'], [reason])
            self.assertEqual(t._evaluation_completion(self.path, exit_code=1, timed_out=False)['status'], 'incomplete')
            self.assertEqual(t._evaluation_completion(self.path, exit_code=0, timed_out=True)['status'], 'incomplete')

    def test_merge_one_hour_cap_never_extends_overall_deadline(self):
        with patch.object(c.time, 'time', return_value=1000):
            self.assertEqual(c._merge_timeout(1000 + 21600), 3600)
            self.assertEqual(c._merge_timeout(1100), 85)
            with self.assertRaises(TimeoutError):
                c._merge_timeout(1015)
            with self.assertRaises(TimeoutError):
                c._merge_timeout(999)

    def ogpo_request(self, mode='engineering', demo_replay=False):
        return dict(checkpoint_run='physical-bc5', checkpoint_step=2000, dataset='batch-pilot5.hdf5',
                    run_id='ogpo-' + mode, mode=mode, demo_replay=demo_replay, seed=123, target=.5)

    def test_ogpo_profiles_match_trainer_parser_and_keep_replay_opt_in(self):
        from scripts.train_ogpo_vadar import parser
        for mode, expected in (('engineering', (2, 2, 1, 20, 4, 600)),
                               ('learning', (4, 10, 100, 2000, 1000, 3600))):
            for seeded in (False, True):
                options = parser().parse_args(t._ogpo_command(self.ogpo_request(mode, seeded))[2:])
                self.assertEqual((options.group_size, options.flow_steps, options.critic_warmup_updates,
                                  options.max_environment_steps, options.max_updates, options.max_seconds), expected)
                self.assertEqual((options.batch_size, options.actor_microbatch_size), (2, 2))
                self.assertEqual((options.grain_count, options.max_episode_steps, options.target_g), (200, 5000, .5))
                self.assertEqual(options.powder_ids, [0, 1, 2, 3, 4])
                self.assertEqual((options.discount, options.r_step, options.r_done, options.value_cap), (.999, 0, 5, 15))
                self.assertEqual(options.demo_replay, Path('/datasets/batch-pilot5.hdf5') if seeded else None)
                self.assertEqual(options.bc_anchor_weight, 0)
                self.assertFalse(options.gradient_smoke)

    def test_ogpo_profile_rejects_unbounded_or_invalid_requests(self):
        for mutation in ({'mode': 'unbounded'}, {'checkpoint_step': 0}, {'run_id': '../overwrite'},
                         {'dataset': '../data.hdf5'}, {'seed': -1}, {'target': float('nan')},
                         {'target': 4.0}, {'demo_replay': 'yes'}):
            request = self.ogpo_request() | mutation
            with self.assertRaises(ValueError):
                t._ogpo_command(request)

    def physical_bc_manifest(self):
        from chemlab_rl.vla_bc import PROPRIO_FEATURES, STATE_KEY, ACTION_KEY
        statistics = {key: {stat: [1. if stat in ('std', 'max') else 0.] * size
                            for stat in ('mean', 'std', 'min', 'max')}
                      for key, size in ((STATE_KEY, 15), (ACTION_KEY, 14))}
        return dict(manifest_version=2, include_measured_scale=True, state_dim=15,
                    state_features=PROPRIO_FEATURES + ['scale_reading_g'], legacy=False,
                    schema_version='scrape_v2', use='synthetic_coarse_grain_control_data',
                    grain_count=200, heldout_powder_ids=[5], heldout_episodes=[],
                    train_episodes=[f'demo_{i}' for i in range(5)],
                    normalization_fitted_on='train_episodes_only', statistics=statistics,
                    dataset_sha256='a' * 64, source_version='sha256:physical-fixture')

    def checkpoint_volume(self, manifest, missing=None, existing_run=None):
        directory = 'physical-bc5/checkpoint_002000'
        files = ['config.json', 'model.safetensors', 'data_manifest.json', 'policy_preprocessor.json',
                 'policy_postprocessor.json', 'run_config.json', 'training_state.pt']
        raw = json.dumps(manifest).encode()
        entries = [types.SimpleNamespace(path=directory + '/' + name, type=FileEntryType.FILE,
                                         size=len(raw) if name == 'data_manifest.json' else 100)
                   for name in files if name != missing]
        def listdir(path):
            if path == '/':
                return [types.SimpleNamespace(path=existing_run)] if existing_run else []
            self.assertEqual(path, '/' + directory)
            return entries
        def read_file(path):
            self.assertEqual(path, '/' + directory + '/data_manifest.json')
            return iter([raw[:10], raw[10:]])
        return types.SimpleNamespace(listdir=listdir, read_file=read_file), raw

    def test_ogpo_preflight_reads_metadata_only_and_requires_physical_bc(self):
        manifest = self.physical_bc_manifest()
        volume, raw = self.checkpoint_volume(manifest)
        checked = []
        with patch.object(t, 'run_volume', volume), patch.object(t, '_require_remote_dataset', checked.append), \
                patch.object(c, '_source_identity', return_value=(manifest['source_version'], {})):
            report = t._preflight_ogpo(self.ogpo_request())
        self.assertEqual(report['manifest_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report['checkpoint_path'], '/runs/physical-bc5/checkpoint_002000')
        self.assertEqual(report['expected_dataset_sha256'], 'a' * 64)
        self.assertEqual(checked, ['batch-pilot5.hdf5'])
        for mutation in ({'legacy': True}, {'grain_count': 400}, {'heldout_powder_ids': []},
                         {'normalization_fitted_on': 'all'}, {'source_version': 'sha256:changed'}):
            bad_volume, _ = self.checkpoint_volume(manifest | mutation)
            with patch.object(t, 'run_volume', bad_volume), patch.object(t, '_require_remote_dataset'), \
                    patch.object(c, '_source_identity', return_value=(manifest['source_version'], {})), \
                    self.assertRaises(ValueError):
                t._preflight_ogpo(self.ogpo_request())

    def test_ogpo_missing_checkpoint_or_dataset_and_existing_run_fail_before_gpu(self):
        manifest = self.physical_bc_manifest()
        volume, _ = self.checkpoint_volume(manifest, missing='model.safetensors')
        with patch.object(t, 'run_volume', volume), patch.object(t, 'verify_workspace'), \
                patch.object(t, 'execute_ogpo_engineering') as gpu, self.assertRaises(FileNotFoundError):
            request = self.ogpo_request()
            t.ogpo_pilot(**{key: value for key, value in request.items()})
        gpu.remote.assert_not_called()
        for dataset_missing in (False, True):
            volume, _ = self.checkpoint_volume(manifest, existing_run=None if dataset_missing else 'ogpo-engineering')
            with patch.object(t, 'run_volume', volume), patch.object(t, 'verify_workspace'), \
                    patch.object(c, '_source_identity', return_value=(manifest['source_version'], {})), \
                    patch.object(t, '_require_remote_dataset', side_effect=FileNotFoundError() if dataset_missing else None), \
                    patch.object(t, 'execute_ogpo_engineering') as gpu, self.assertRaises(FileNotFoundError if dataset_missing else FileExistsError):
                t.ogpo_pilot(**self.ogpo_request())
            gpu.remote.assert_not_called()

    def test_ogpo_wrapper_selects_one_profile_and_persists_explicit_seed_mode(self):
        for mode, cap, seeded in (('engineering', 1200, False), ('learning', 3900, True)):
            calls = []
            unused = []
            selected = types.SimpleNamespace(remote=lambda *args: calls.append(args) or {'exit_code': 0})
            other = types.SimpleNamespace(remote=lambda *args: unused.append(args))
            request = self.ogpo_request(mode, seeded)
            with patch.object(t, 'ROOT', self.path), patch.object(t, 'verify_workspace'), \
                    patch.object(t, '_preflight_ogpo', return_value={'checked': True}), \
                    patch.object(t, 'execute_ogpo_engineering', selected if mode == 'engineering' else other), \
                    patch.object(t, 'execute_ogpo_learning', selected if mode == 'learning' else other), \
                    patch.object(t.time, 'time', return_value=1000):
                t.ogpo_pilot(**request)
            self.assertEqual(len(calls), 1)
            self.assertFalse(unused)
            self.assertEqual(calls[0][2], 1000 + cap)
            launch = json.loads((self.path / 'docs/pilot' / request['run_id'] / 'wrapper_launch.json').read_text())
            self.assertEqual(launch['replay_initialization'], 'demonstration_seeded_off_policy' if seeded else 'online_only')
            self.assertEqual(launch['limits']['total_seconds'], cap)

    def test_ogpo_server_functions_have_fixed_timeout_and_one_gpu_container(self):
        tree = ast.parse((t.ROOT / 'scripts/modal_train.py').read_text())
        for name, cap in (('execute_ogpo_engineering', 1200), ('execute_ogpo_learning', 3900)):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            values = {arg.arg: ast.literal_eval(arg.value) for arg in function.decorator_list[0].keywords
                      if arg.arg in {'timeout', 'max_containers', 'retries', 'gpu', 'cpu', 'memory'}}
            self.assertEqual(values, dict(timeout=cap, max_containers=1, retries=0, gpu='A100-80GB',
                                          cpu=(8, 8), memory=(32768, 32768)))

    def test_ogpo_server_enforces_absolute_deadline_and_preflight_source(self):
        checkpoint = self.path / 'checkpoint'
        checkpoint.mkdir()
        raw = b'{"physical_manifest_fixture":true}'
        (checkpoint / 'data_manifest.json').write_bytes(raw)
        dataset = self.path / 'data.hdf5'
        dataset.write_bytes(b'file-existence fixture')
        module = types.ModuleType('scripts.collect_granular_demos')
        module.source_identity = lambda: ('sha256:fixed', {})
        for mode, limit in (('engineering', 1200), ('learning', 3900)):
            request = self.ogpo_request(mode)
            request['preflight'] = dict(checkpoint_path=str(checkpoint), dataset_path=str(dataset),
                                        manifest_sha256=hashlib.sha256(raw).hexdigest(), source_version='sha256:fixed')
            with patch.dict(sys.modules, {'scripts.collect_granular_demos': module}), \
                    patch.object(t.time, 'time', return_value=1000), patch.object(t, '_execute_command', return_value={}) as execute:
                t._execute_ogpo_pilot(request['run_id'], request, 1000 + limit + 100, mode)
                self.assertEqual(execute.call_args.args[2], 1000 + limit - 15)
                self.assertEqual(execute.call_args.kwargs['termination_grace_seconds'], 5)
                self.assertEqual(execute.call_args.kwargs['launcher_context']['hard_deadline_epoch'], 1000 + limit)
                t._execute_ogpo_pilot(request['run_id'], request, 1100, mode)
                self.assertEqual(execute.call_args.args[2], 1085)
                with self.assertRaises(TimeoutError):
                    t._execute_ogpo_pilot(request['run_id'], request, 1010, mode)
                request['preflight']['source_version'] = 'sha256:changed'
                with self.assertRaises(ValueError):
                    t._execute_ogpo_pilot(request['run_id'], request, 2000, mode)
if __name__ == '__main__':
    unittest.main()
