"""Bounded, checkpoint-resumable continuation of vial BC to 500k updates.

Each Modal execution trains for at most 3h45. On a time-budget stop, a durable
Modal retry restores the last committed complete checkpoint, including Adam/RNG.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal
from scripts.modal_train import (ROOT, REMOTE_ROOT, image, data_volume, run_volume,
                                cache_volume, verify_workspace)

app = modal.App('chemlab-vial-bc-long')
SOURCE_RUN = 'smolvla-vial98-a100-bc10000-001'
SOURCE_STEP = 10000


def validate(request):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', request['run_id']):
        raise ValueError('Invalid run ID')
    if request['target_step'] != 500000 or request['batch_size'] != 8:
        raise ValueError('This experiment is fixed to 500k total updates at batch 8')
    if request['source_run'] != SOURCE_RUN or request['source_step'] != SOURCE_STEP:
        raise ValueError('Expected the verified 10k BC source checkpoint')


def latest_complete(out, source):
    candidates = []
    for marker in out.glob('checkpoint_*/cloud_complete.json'):
        directory = marker.parent
        proof = json.loads(marker.read_text())
        step = int(directory.name.split('_')[1])
        if proof['step'] != step:
            raise ValueError('Checkpoint marker step mismatch')
        if not all((directory / name).is_file() for name in
                   ('training_state.pt','model.safetensors','config.json','data_manifest.json','heldout_evaluation.json')):
            raise ValueError('Committed checkpoint files missing')
        candidates.append((step, directory))
    match=re.fullmatch(r'checkpoint_([0-9]{6})',source.name)
    fallback_step=int(match.group(1)) if match else SOURCE_STEP
    return max([*candidates,(fallback_step,source)],key=lambda value:value[0])


def command_for(request, out, resume):
    validate(request)
    command = [sys.executable, '-u', '-m', 'scripts.train_vial_vla_bc',
            '--dataset', '/datasets/weigh_vial_hybrid98.hdf5', '--output', str(out),
            '--resume', str(resume), '--steps', str(request['target_step']),
            '--batch-size', '8', '--max-seconds', '13500', '--save-every', '25000',
            '--seed', '0', '--chunk-size', '50', '--n-action-steps', '2',
            '--heldout-samples', '128', '--bf16', '--split-json', str(out/'split.json')]
    if resume.parent == out:
        command.append('--reuse-resume-checkpoint')
    return command


@app.function(image=image, gpu='A100-80GB', cpu=(8,8), memory=(32768,32768),
              timeout=14400, startup_timeout=600, max_containers=1, scaledown_window=2,
              retries=modal.Retries(max_retries=10, initial_delay=10, max_delay=60),
              volumes={'/datasets':data_volume.with_mount_options(read_only=True),
                       '/runs':run_volume,'/model-cache':cache_volume})
def train(request):
    import torch
    from chemlab_rl.vial_vla_data import file_sha256
    validate(request)
    run_volume.reload()
    out = Path('/runs')/request['run_id']
    source = Path('/runs')/SOURCE_RUN/f'checkpoint_{SOURCE_STEP:06d}'
    manifest = json.loads((source/'data_manifest.json').read_text())
    if file_sha256('/datasets/weigh_vial_hybrid98.hdf5') != manifest['dataset_sha256']:
        raise ValueError('Dataset differs from original BC')
    if request['dataset_sha256'] != manifest['dataset_sha256']:
        raise ValueError('Requested data hash differs')
    split = {key:manifest[key] for key in ('train_episodes','heldout_episodes','excluded_episodes')}
    out.mkdir(exist_ok=True)
    request_file=out/'request.json'
    if request_file.exists() and json.loads(request_file.read_text()) != request:
        raise ValueError('Run ID already belongs to another request')
    request_file.write_text(json.dumps(request,indent=2))
    (out/'split.json').write_text(json.dumps(split,indent=2))
    start_step,resume=latest_complete(out,source)
    if start_step >= request['target_step']:
        return dict(status='completed',steps=start_step,checkpoint=str(resume))
    command=command_for(request,out,resume)
    launched=time.time()
    launch=dict(start_step=start_step,target_step=request['target_step'],resume=str(resume),
                command=command,started_epoch=launched,gpu=torch.cuda.get_device_name(0),
                train_script_sha256=hashlib.sha256((Path(REMOTE_ROOT)/'scripts/train_vial_vla_bc.py').read_bytes()).hexdigest())
    with (out/'segments.jsonl').open('a') as stream:
        stream.write(json.dumps(launch)+'\n')
    (out/'progress.json').write_text(json.dumps(dict(status='loading',step=start_step,**launch),indent=2))
    run_volume.commit()
    print(json.dumps(dict(event='continuation_started',**launch)),flush=True)
    proc=subprocess.Popen(command,cwd=REMOTE_ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1,env={**os.environ,'CHEMLAB_DEADLINE_UTC':str(launched+14100)})
    try:
        with (out/'console.log').open('a',buffering=1) as log:
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
                try:
                    row=json.loads(line)
                except ValueError:
                    continue
                if row.get('event')=='checkpoint':
                    directory=Path(row['path'])
                    # Printed only after model, optimizer/RNG and heldout files
                    # have been written. A partial checkpoint is never resumed.
                    (directory/'cloud_complete.json').write_text(json.dumps(dict(step=row['step'],committed_epoch=time.time())))
                    (out/'progress.json').write_text(json.dumps(dict(status='training',step=row['step'],target_step=request['target_step'],checkpoint=str(directory)),indent=2))
                    run_volume.commit()
                elif 'step' in row and 'loss' in row and row.get('event')!='heldout_loss':
                    (out/'progress.json').write_text(json.dumps(dict(status='training',target_step=request['target_step'],**row),indent=2))
        code=proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        run_volume.commit()
        cache_volume.commit()
    if code:
        raise RuntimeError(f'Trainer exited {code}; retry will restore last committed checkpoint')
    result=json.loads((out/'training_result.json').read_text())
    (out/f'segment_{start_step:06d}_{result["steps"]:06d}.json').write_text(json.dumps(result,indent=2))
    complete=result['steps']>=request['target_step']
    status=dict(status='completed' if complete else 'resuming',step=result['steps'],
                target_step=request['target_step'],checkpoint=result['checkpoint'],
                final_heldout=result['final_heldout'],segment_elapsed_seconds=result['elapsed_seconds'])
    (out/'progress.json').write_text(json.dumps(status,indent=2))
    run_volume.commit()
    if not complete:
        raise RuntimeError(f'Planned resumable boundary at update {result["steps"]}; target remains 500000')
    return status


@app.local_entrypoint()
def main(run_id: str='smolvla-vial98-a100-bc500000-001'):
    verify_workspace()
    source=ROOT/'docs/pilot'/SOURCE_RUN/'wrapper_result.json'
    prior=json.loads(source.read_text())
    request=dict(run_id=run_id,target_step=500000,batch_size=8,source_run=SOURCE_RUN,
                 source_step=SOURCE_STEP,dataset_sha256=prior['dataset_sha256'])
    validate(request)
    out=ROOT/'docs/pilot'/run_id
    out.mkdir(exist_ok=False)
    (out/'wrapper_launch.json').write_text(json.dumps(request,indent=2))
    # spawn().get() remains durable across local disconnects in detached mode.
    call=train.spawn(request)
    (out/'function_call.json').write_text(json.dumps(dict(function_call_id=call.object_id),indent=2))
    print(json.dumps(dict(event='submitted',function_call_id=call.object_id,**request)),flush=True)
    result=call.get()
    (out/'wrapper_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
