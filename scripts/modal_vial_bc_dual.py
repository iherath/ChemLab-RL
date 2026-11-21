"""Resumable two-H100 BC continuation, with an overall compute-time limit."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal
from scripts.modal_train import image,data_volume,run_volume,cache_volume,verify_workspace,ROOT
from scripts.modal_vial_bc_long import latest_complete

app=modal.App('chemlab-vial-bc-dual-h100')
volumes={'/datasets':data_volume.with_mount_options(read_only=True),'/runs':run_volume,'/model-cache':cache_volume}


def execute(request,verification=False):
    from chemlab_rl.vial_vla_data import file_sha256
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',request['run_id']):raise ValueError('Invalid run ID')
    if request['target_step']!=500000 or request['global_batch_size']!=8:raise ValueError('Fixed 500k/global8 experiment')
    run_volume.reload()
    source_run=request.get('source_run','smolvla-vial98-a100-bc500000-001')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',source_run):raise ValueError('Invalid source run')
    source_root=Path('/runs')/source_run
    fallback=Path('/runs/smolvla-vial98-a100-bc10000-001/checkpoint_010000')
    _,source=latest_complete(source_root,fallback)
    out=Path('/runs')/request['run_id'];out.mkdir(exist_ok=True)
    step,resume=latest_complete(out,source)
    # latest_complete's fallback step is 10k; the actual source may be newer.
    if resume==source:step=int(source.name.split('_')[1])
    manifest=json.loads((resume/'data_manifest.json').read_text())
    if file_sha256('/datasets/weigh_vial_hybrid98.hdf5')!=manifest['dataset_sha256']:raise ValueError('Dataset changed')
    request_path=out/'request.json'
    if request_path.exists() and json.loads(request_path.read_text())!=request:raise ValueError('Conflicting request')
    request_path.write_text(json.dumps(request,indent=2))
    remaining=request['deadline_epoch']-time.time()
    if remaining<600:
        status=dict(status='compute_time_limit',step=step,target_step=500000,checkpoint=str(resume))
        (out/'progress.json').write_text(json.dumps(status,indent=2));run_volume.commit();return status
    target=step+60 if verification else 500000
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=2',
             '-m','scripts.train_vial_bc_distributed','--resume',str(resume),'--output',str(out),
             '--steps',str(target),'--max-seconds',str(min(13500,remaining-300)),
             '--save-every','25000']
    if not verification:
        command+=['--max-update-seconds','0.16']
    segment=dict(start_step=step,target_step=target,resume=str(resume),command=command,started_epoch=time.time(),
                 gpus=2,gpu_type='H100',global_batch_size=8,per_device_batch_size=4)
    with (out/'segments.jsonl').open('a') as stream:stream.write(json.dumps(segment)+'\n')
    print(json.dumps(dict(event='distributed_continuation_started',**segment)),flush=True)
    run_volume.commit()
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        with (out/'console.log').open('a',buffering=1) as stream:
            for line in proc.stdout:
                print(line,end='',flush=True);stream.write(line)
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('event')=='checkpoint':
                    directory=Path(row['path'])
                    (directory/'cloud_complete.json').write_text(json.dumps(dict(step=row['step'],committed_epoch=time.time())))
                    run_volume.commit()
                if 'step' in row:
                    (out/'progress.json').write_text(json.dumps(dict(status='training',target_step=target,**row),indent=2))
            code=proc.wait()
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        run_volume.commit();cache_volume.commit()
    if code:raise RuntimeError(f'Distributed trainer exited {code}')
    result=json.loads((out/'training_result.json').read_text())
    (out/f'segment_{step:06d}_{result["steps"]:06d}.json').write_text(json.dumps(result,indent=2))
    if result['status']=='slow_allocation':
        (out/'progress.json').write_text(json.dumps(result,indent=2));run_volume.commit();return result
    if verification:
        import torch
        checkpoint=Path(result['checkpoint'])
        saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
        if saved['step']!=target or len(saved.get('distributed_cuda_rngs',[]))!=2:raise ValueError('Distributed checkpoint incomplete')
        optimizer_steps=[int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
        if not optimizer_steps or min(optimizer_steps)<target:raise ValueError('Adam continuation counters differ')
        result['optimizer_checkpoint_verified']=True
    else:
        result['status']='completed' if result['steps']>=500000 else 'resuming'
    (out/'progress.json').write_text(json.dumps(result,indent=2));run_volume.commit()
    return result


@app.function(image=image,gpu='H100:2',cpu=(16,16),memory=(65536,65536),timeout=900,retries=0,
              max_containers=1,scaledown_window=2,volumes=volumes)
def verify(request):return execute(request,verification=True)


@app.function(image=image,gpu='H100:2',cpu=(16,16),memory=(65536,65536),timeout=14400,
              retries=1,single_use_containers=True,
              max_containers=1,scaledown_window=2,volumes=volumes)
def train(request):return execute(request)


@app.function(image=image,cpu=.125,memory=512,timeout=86400,retries=0,max_containers=1,
              scaledown_window=2)
def coordinate(request):
    """Keep the run alive across segment ends and rejected GPU allocations."""
    errors=0
    while time.time()<request['deadline_epoch']-600:
        try:
            result=train.spawn(request).get()
            errors=0
        except Exception as error:
            errors+=1
            print(json.dumps(dict(event='segment_error',consecutive_errors=errors,error=str(error))),flush=True)
            if errors>=3:raise
            continue
        print(json.dumps(dict(event='segment_finished',status=result['status'],steps=result.get('steps',result.get('step')))),flush=True)
        if result['status'] in ('completed','compute_time_limit'):
            return result
        # A slow allocation and a normal time-budget stop both have a complete
        # checkpoint. The next single-use input requests a fresh GPU container.
    return dict(status='compute_time_limit',target_step=500000)


@app.local_entrypoint()
def main(run_id: str='smolvla-vial98-2h100-bc500000-001',verification: bool=False,
         source_run: str='smolvla-vial98-a100-bc500000-001',max_hours: float=21):
    verify_workspace()
    if not 0<max_hours<=21:raise ValueError('Compute-time budget must be at most 21h')
    out=ROOT/'docs/pilot'/run_id;out.mkdir(exist_ok=False)
    # 21h of requested resources is about $193 at current list prices, leaving
    # room within the user's $267.23 balance even if throughput disappoints.
    request=dict(run_id=run_id,target_step=500000,global_batch_size=8,
                 source_run=source_run,deadline_epoch=time.time()+(900 if verification else max_hours*3600))
    (out/'wrapper_launch.json').write_text(json.dumps(request,indent=2))
    call=(verify if verification else coordinate).spawn(request)
    (out/'function_call.json').write_text(json.dumps(dict(function_call_id=call.object_id),indent=2))
    print(json.dumps(dict(event='submitted',function_call_id=call.object_id,**request)),flush=True)
    result=call.get()
    (out/'wrapper_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
