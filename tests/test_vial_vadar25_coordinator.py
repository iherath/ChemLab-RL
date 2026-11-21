"""Full-run continuation must survive boundaries and transient worker failures."""
from pathlib import Path
from unittest.mock import Mock,patch
import pytest
from scripts import modal_vial_vadar25_full as launcher

def pilot_result(stochastic=6,native=5):
    return dict(status='completed',density_check={'passed':True},post_density_check={'passed':True},
        training={'actor_updates':128},finetuned_stochastic={'trials':10,'successes':stochastic},
        finetuned_native_ode={'trials':10,'successes':native})

def run_results(tmp_path,outcomes):
    worker=Mock();calls=[]
    for outcome in outcomes:
        call=Mock()
        if isinstance(outcome,Exception):call.get.side_effect=outcome
        else:call.get.return_value=outcome
        calls.append(call)
    worker.spawn.side_effect=calls
    with patch.object(launcher,'segment',worker),patch.object(launcher,'run_volume',Mock()), \
         patch.object(launcher,'Path',side_effect=lambda _:tmp_path), \
         patch.object(launcher.modal.FunctionCall,'from_id',return_value=Mock(get=Mock(return_value=pilot_result()))), \
         patch.object(launcher.time,'time',return_value=0):
        result=launcher.coordinate.local('pilot-call','pilot','full')
    return result,worker

def test_score_dip_does_not_cancel_full_run():
    assert launcher.pilot_gate(pilot_result(6,5))[0]
    assert launcher.pilot_gate(pilot_result(1,0))[0]
    assert not launcher.pilot_gate(pilot_result(0,0))[0]

def test_invalid_density_cannot_start_training():
    result=pilot_result();result['post_density_check']['passed']=False
    assert not launcher.pilot_gate(result)[0]

def test_boundaries_and_worker_failure_continue(tmp_path):
    result,worker=run_results(tmp_path,[dict(status='resuming',gpu_seconds=2800),
        RuntimeError('worker interrupted'),dict(status='resuming',gpu_seconds=2800),
        dict(status='completed',gpu_seconds=2800)])
    assert worker.spawn.call_count==4 and result['status']=='completed'
    assert len(result['failures'])==1

def test_worker_time_limit_resumes_instead_of_stopping(tmp_path):
    result,worker=run_results(tmp_path,[dict(status='compute_time_limit',gpu_seconds=3500),
        dict(status='resuming',gpu_seconds=3500),dict(status='resuming',gpu_seconds=3500),
        dict(status='evaluating',gpu_seconds=3500),dict(status='completed',gpu_seconds=1500)])
    assert result['status']=='completed' and worker.spawn.call_count==5
    assert result['gpu_seconds_used']>14400 and result['gpu_seconds_limit'] is None

def test_persistent_failure_is_explicit(tmp_path):
    with pytest.raises(RuntimeError,match='persistent failure'):
        run_results(tmp_path,[RuntimeError('persistent failure')]*3)
    result=__import__('json').loads((tmp_path/'full/coordinator.json').read_text())
    assert result['status']=='recovery_failed'

def test_latest_bundle_ignores_uncommitted_directory(tmp_path):
    import json
    for step in (178,228):
        p=tmp_path/f'checkpoint_{step:06d}';p.mkdir()
        (p/'trainer.pt').write_bytes(b'checkpoint');(p/'replay.sqlite').write_bytes(b'replay')
        (p/'complete.json').write_text(json.dumps({'actor_updates':step}))
    (tmp_path/'checkpoint_000278').mkdir()
    assert launcher.latest_bundle(tmp_path)[0]==228
