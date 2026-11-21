"""Two-GPU BC with global batch eight and valid-element weighted gradients."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--max-seconds',type=float,default=13500)
    p.add_argument('--save-every',type=int,default=25000)
    p.add_argument('--verification-steps',type=int,default=0)
    p.add_argument('--max-update-seconds',type=float,default=0)
    a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader,RandomSampler
    from chemlab_rl.vial_vla_data import VialHDFDataset,load_smolvla
    from chemlab_rl.vla_bc import _save_checkpoint
    from scripts.train_vial_vla_bc import heldout_loss
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('This experiment requires two GPUs, local batch 4, global batch 8')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    device=f'cuda:{rank}'
    torch.manual_seed(rank);np.random.seed(rank);random.seed(rank)
    manifest=json.loads((a.resume/'data_manifest.json').read_text())
    policy,pre,post=load_smolvla(manifest,device,str(a.resume),chunk_size=50,n_action_steps=2)
    policy.train()
    parameters=[v for v in policy.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    state=torch.load(a.resume/'training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    step=int(state['step']);initial_step=step
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    saved_rng=state.get('distributed_cuda_rngs')
    if saved_rng and len(saved_rng)==world:
        torch.cuda.set_rng_state(saved_rng[rank],rank)
    elif rank==0 and state.get('cuda_rng'):
        torch.cuda.set_rng_state(state['cuda_rng'][0],rank)
    else:
        torch.cuda.manual_seed(step+rank)
    epoch=int(state.get('data_epoch',0));batch_in_epoch=int(state.get('batch_in_epoch',0))
    del state
    model=DDP(policy,device_ids=[rank],find_unused_parameters=False)
    training=VialHDFDataset('/datasets/weigh_vial_hybrid98.hdf5',manifest,manifest['train_episodes'],50)
    validation=VialHDFDataset('/datasets/weigh_vial_hybrid98.hdf5',manifest,manifest['heldout_episodes'],50)
    # Both ranks use the same shuffled global batch, then take disjoint halves.
    # This makes valid-element weighting available on CPU without a per-step
    # GPU collective. Each rank still forwards/backpropagates only four samples.
    loader_generator=torch.Generator().manual_seed(epoch)
    sampler=RandomSampler(training,generator=loader_generator)
    loader=DataLoader(training,batch_size=8,sampler=sampler,generator=torch.Generator().manual_seed(0),
                      drop_last=True,num_workers=2,pin_memory=True,persistent_workers=True)
    config=dict(global_batch_size=8,per_device_batch_size=4,world_size=world,lr=1e-4,
                precision='bf16',resume=str(a.resume),steps=a.steps,chunk_size=50,n_action_steps=2,
                state_dim=15,trainable_parameters=sum(v.numel() for v in parameters),
                distributed_loss_weighting='global_valid_action_elements',optimizer_restored=True)
    a.output.mkdir(exist_ok=True,parents=True)
    if rank==0:
        for name,payload in [('data_manifest.json',manifest),('run_config.json',config)]:
            (a.output/name).write_text(json.dumps(payload,indent=2))
        print(json.dumps(dict(event='distributed_model_ready',start_step=step,**config)),flush=True)
    started=time.monotonic();evaluations={};last_checkpoint=a.resume;last_saved_step=step
    def checkpoint():
        nonlocal last_checkpoint,last_saved_step
        rngs=[None]*world if rank==0 else None
        dist.gather_object(torch.cuda.get_rng_state(rank).cpu(),rngs,dst=0)
        if rank==0:
            evaluation=heldout_loss(policy,pre,validation,batch_size=8,samples=128,device=device,bf16=True)
            evaluations[step]=evaluation
            target=_save_checkpoint(policy,pre,post,optimizer,manifest,a.output,step,time.monotonic()-started,config)
            training_state=torch.load(target/'training_state.pt',map_location='cpu',weights_only=False)
            training_state.update(distributed_cuda_rngs=rngs,data_epoch=epoch,batch_in_epoch=batch_in_epoch)
            torch.save(training_state,target/'training_state.pt')
            del training_state
            (target/'heldout_evaluation.json').write_text(json.dumps(evaluation,indent=2))
            with (a.output/'heldout_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,**evaluation))+'\n')
            last_checkpoint=target
            print(json.dumps(dict(event='checkpoint',step=step,path=str(target),heldout_loss=evaluation['loss'])),flush=True)
        dist.barrier()
        last_saved_step=step
    reason='steps';losses=[];throughput_started=None
    control=torch.zeros(1,dtype=torch.int32,device=device)
    try:
        iterator=iter(loader)
        for _ in range(batch_in_epoch):next(iterator)
        while step<a.steps:
            if (step-initial_step)%20==0:
                if rank==0:
                    control[0]=int(time.monotonic()-started>=a.max_seconds or (a.output/'stop_requested').exists())
                dist.broadcast(control,src=0)
                if int(control):reason='time_or_stop_limit';break
            try:batch=next(iterator)
            except StopIteration:
                epoch+=1;batch_in_epoch=0;loader_generator.manual_seed(epoch);iterator=iter(loader);batch=next(iterator)
            batch_in_epoch+=1
            tick=time.monotonic()
            all_valid=int((~batch['action_is_pad']).sum())*14
            batch={key:value[rank::world] if isinstance(value,(torch.Tensor,list)) else value for key,value in batch.items()}
            local_valid=int((~batch['action_is_pad']).sum())*14
            processed=pre(batch)
            preprocess_seconds=time.monotonic()-tick
            events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                local_loss,_=model(processed)
                loss=local_loss*(world*local_valid/all_valid)
            events[1].record()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distributed BC loss')
            loss.backward()
            events[2].record()
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            optimizer.step()
            events[3].record()
            mean=(local_loss.detach()*local_valid/all_valid);dist.all_reduce(mean)
            events[4].record()
            step+=1;torch.cuda.synchronize()
            if step-initial_step==40:throughput_started=time.monotonic()
            row=dict(step=step,loss=float(mean),grad_norm=float(norm),elapsed_seconds=time.monotonic()-started,
                     step_seconds=time.monotonic()-tick,preprocess_seconds=preprocess_seconds,
                     forward_ms=events[0].elapsed_time(events[1]),backward_ms=events[1].elapsed_time(events[2]),
                     optimizer_ms=events[2].elapsed_time(events[3]),loss_reduce_ms=events[3].elapsed_time(events[4]),
                     global_batch_size=8,world_size=world)
            if rank==0:
                losses.append(row['loss'])
                with (a.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if step%20==0 or step==initial_step+1:print(json.dumps(row),flush=True)
            if a.max_update_seconds and step-initial_step==100:
                if rank==0:
                    measured=(time.monotonic()-throughput_started)/60
                    control[0]=int(measured>a.max_update_seconds)
                    print(json.dumps(dict(event='throughput_check',seconds_per_update=measured,
                                          accepted=not bool(control[0]),limit=a.max_update_seconds)),flush=True)
                dist.broadcast(control,src=0)
                if int(control):reason='slow_allocation';break
            if a.verification_steps and step-initial_step>=a.verification_steps:
                reason='verification_limit';break
            if step%a.save_every==0:checkpoint()
        if step!=last_saved_step:checkpoint()
        if rank==0:
            result=dict(status='completed' if step>=a.steps else reason,steps=step,initial_step=initial_step,
                        target_step=a.steps,checkpoint=str(last_checkpoint),final_heldout=evaluations.get(step),
                        elapsed_seconds=time.monotonic()-started,world_size=world,global_batch_size=8,
                        optimizer_restored=True,last_loss=losses[-1] if losses else None)
            (a.output/'training_result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(event='result',**result)),flush=True)
    finally:
        training.close();validation.close();dist.destroy_process_group()


if __name__=='__main__':main()
