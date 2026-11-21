"""Regression coverage for the premature-stop bug in long BC orchestration."""
from unittest.mock import Mock,patch

import pytest

from scripts import modal_vial_bc_dual as launcher


def run_with_results(results):
    worker=Mock()
    calls=[]
    for result in results:
        call=Mock()
        if isinstance(result,Exception):
            call.get.side_effect=result
        else:
            call.get.return_value=result
        calls.append(call)
    worker.spawn.side_effect=calls
    request={'deadline_epoch':10000}
    with patch.object(launcher,'train',worker),patch.object(launcher.time,'time',return_value=0):
        result=launcher.coordinate.local(request)
    return result,worker


def test_slow_allocation_does_not_end_job():
    result,worker=run_with_results([
        {'status':'slow_allocation','steps':118360},
        {'status':'completed','steps':500000},
    ])
    assert worker.spawn.call_count==2
    assert result=={'status':'completed','steps':500000}


def test_segment_boundary_and_slow_allocation_both_continue():
    result,worker=run_with_results([
        {'status':'resuming','steps':118260},
        {'status':'slow_allocation','steps':118360},
        {'status':'resuming','steps':226000},
        {'status':'completed','steps':500000},
    ])
    assert worker.spawn.call_count==4
    assert result['steps']==500000


def test_transient_worker_failure_is_retried():
    result,worker=run_with_results([
        RuntimeError('worker disconnected'),
        {'status':'resuming','steps':200000},
        {'status':'completed','steps':500000},
    ])
    assert worker.spawn.call_count==3
    assert result['status']=='completed'


def test_repeated_worker_errors_are_reported_as_errors():
    with pytest.raises(RuntimeError,match='worker failure'):
        run_with_results([RuntimeError('worker failure') for _ in range(3)])


def test_budget_limit_is_not_reported_as_training_completion():
    result,worker=run_with_results([
        {'status':'compute_time_limit','step':400000},
    ])
    assert worker.spawn.call_count==1
    assert result['status']=='compute_time_limit'


def test_expired_budget_does_not_start_another_worker():
    worker=Mock()
    with patch.object(launcher,'train',worker),patch.object(launcher.time,'time',return_value=10000):
        result=launcher.coordinate.local({'deadline_epoch':10000})
    worker.spawn.assert_not_called()
    assert result['status']=='compute_time_limit'
