"""Durable pilot-to-full coordinator with versioned replay/checkpoint recovery."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import modal
from scripts.modal_train import image,run_volume,data_volume,cache_volume,verify_workspace

app=modal.App('chemlab-vial-vadar25-full')
SOURCE='/runs/smolvla-vial98-2h100-bc500000-004/checkpoint_400000'
SEEDS=list(range(1001,1011))
TARGET=2000
WORKER_SECONDS=3500  # Per-worker boundary only; never a total training cutoff.

def pilot_gate(result):
    """A modest task-score dip is not a veto; reject invalid or collapsed runs."""
    if result.get('status')!='completed':return False,'Pilot did not finish its checks'
    if not all(result.get(key,{}).get('passed') for key in ('density_check','post_density_check')):
        return False,'Sampling densities failed consistency checks'
    if result.get('training',{}).get('actor_updates',0)<1:return False,'No real actor updates completed'
    evaluations=[result.get(key,{}) for key in ('finetuned_stochastic','finetuned_native_ode')]
    if any(e.get('trials')!=10 for e in evaluations):return False,'Pilot evaluation is incomplete'
    if not any(e.get('successes',0)>0 for e in evaluations):return False,'Both samplers lost all task success'
    return True,'Pilot updates are valid and task completion is retained; no baseline-matching threshold'

def latest_bundle(root):
    candidates=[]
    for marker in root.glob('checkpoint_*/complete.json'):
        row=json.loads(marker.read_text());directory=marker.parent
        if not all((directory/f).is_file() for f in ('trainer.pt','replay.sqlite')):
            raise ValueError('Committed checkpoint bundle is missing files')
        candidates.append((row['actor_updates'],directory,row))
    return max(candidates,key=lambda x:x[0]) if candidates else None

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=49152,timeout=3600,retries=0,
              single_use_containers=True,scaledown_window=2,
              volumes={'/runs':run_volume,'/datasets':data_volume.with_mount_options(read_only=True),'/model-cache':cache_volume})
def segment(run_id,pilot_run,remaining_gpu_seconds):
    import hashlib
    import shutil
    import sqlite3
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from chemlab_rl.vial_vla_data import load_smolvla,make_inference_batch,file_sha256
    from chemlab_rl.vial_vla_env import VialVLAEnv,source_identity
    from chemlab_rl.vial_ogpo_training import VialOGPOConfig,VialTrainer,VialReplay,run_vial_online,evaluate_vial_trainer
    from chemlab_rl.ogpo_training import FrozenSmolVLAFeatures,VectorQEnsemble,PhysicalActionAdapter
    from chemlab_rl.smolvla_ogpo import SmolVLAVelocity,ChainConfig
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    run_volume.reload();root=Path('/runs')/run_id;root.mkdir(exist_ok=True)
    prior=Path('/runs')/pilot_run;pilot_result=json.loads((prior/'result.json').read_text())
    accepted,reason=pilot_gate(pilot_result)
    if not accepted:raise ValueError(reason)
    proof=json.loads((prior/'launch.json').read_text());manifest=json.loads((Path(SOURCE)/'data_manifest.json').read_text())
    if proof['simulation_source']!=source_identity() or proof['weights_sha256']!=file_sha256(Path(SOURCE)/'model.safetensors'):
        raise ValueError('Simulator or source checkpoint differs from pilot')
    cfg=dict(proof['config']);cfg['chain']=ChainConfig(**cfg['chain']);config=VialOGPOConfig(**cfg)
    provenance=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
    initial=pilot_result['training']['actor_updates'];end_target=initial+TARGET
    bundle=latest_bundle(root);source=bundle[1] if bundle else prior
    source_step=bundle[0] if bundle else initial
    remaining_updates=max(0,end_target-source_step)
    phase=root/f'segment_{source_step:06d}_{int(time.time())}';phase.mkdir()
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2));run_volume.commit()
    def log(row):
        print(json.dumps(row),flush=True)
        with (root/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (root/'progress.json').write_text(json.dumps(row))
        if row.get('event')=='checkpoint':commit_bundle()
    policy,pre,post=load_smolvla(manifest,'cuda',SOURCE,chunk_size=50,n_action_steps=25)
    features=FrozenSmolVLAFeatures(policy,15)
    critic=VectorQEnsemble(features.feature_dim,action_steps=25,subgoals=4,members=2,hidden=256)
    trainer=VialTrainer(SmolVLAVelocity(policy,cache_prefix=config.cache_prefix),critic,preprocessor=pre,feature_fn=features,
        action_adapter=PhysicalActionAdapter(post,[-1]*14,[1]*14),config=config,device='cuda',seed=0,provenance=provenance)
    # SQLite backup includes any WAL contents and preserves the checkpoint revision.
    with sqlite3.connect(f'file:{source}/replay.sqlite?mode=ro',uri=True) as src,sqlite3.connect(phase/'replay.sqlite') as dest:
        src.backup(dest)
    replay=VialReplay(phase/'replay.sqlite',provenance=provenance,capacity=30000,action_steps=25,seed=0)
    trainer.load_checkpoint(source/'trainer.pt',replay)
    if trainer.actor_updates!=source_step:raise ValueError('Checkpoint step differs from complete marker')
    last_committed=None
    def commit_bundle():
        nonlocal last_committed
        step=trainer.actor_updates
        if step==last_committed:return
        directory=root/f'checkpoint_{step:06d}';directory.mkdir(exist_ok=True)
        shutil.copyfile(phase/'trainer.pt',directory/'trainer.pt')
        with sqlite3.connect(directory/'replay.sqlite') as dest:replay.db.backup(dest)
        marker=dict(actor_updates=step,critic_updates=trainer.critic_updates,environment_steps=trainer.environment_steps,
                    initial_actor_updates=initial,additional_actor_updates=step-initial,committed_epoch=time.time())
        (directory/'complete.json').write_text(json.dumps(marker,indent=2));run_volume.commit()
        last_committed=step
        # Keep two recent recovery points plus milestone evaluations.
        bundles=sorted(root.glob('checkpoint_*/complete.json'),key=lambda p:int(p.parent.name.split('_')[1]))
        for old in bundles[:-2]:
            count=int(old.parent.name.split('_')[1])-initial
            if count%500:shutil.rmtree(old.parent)
    env=VialVLAEnv(max_steps=1600)
    def evaluate(name,seeds,native=False):
        directory=phase/name;directory.mkdir();results=[]
        rng=trainer.generator.get_state();global_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:
            for seed in seeds:
                seed_dir=directory/f'seed_{seed}';seed_dir.mkdir();rows=[];native_step=env.step
                with imageio.get_writer(seed_dir/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=1) as writer:
                    def recorded_step(action):
                        result=native_step(action);obs,_,_,_,info=result
                        writer.append_data(np.concatenate([obs[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
                        rows.append(dict(control=len(rows)+1,doors_m=obs['object'][8:11].tolist(),
                            scale_g=float(obs['object'][7]),subgoals=info['subgoal_dones'].tolist(),success=bool(info['success'])))
                        return result
                    env.step=recorded_step
                    try:
                        if native:
                            obs,info=env.reset(seed=seed);p=trainer.target_actor.policy;p.reset();torch.manual_seed(seed)
                            with torch.inference_mode():
                                for _ in range(1600):
                                    a=post(p.select_action(pre(make_inference_batch(obs))))[0].cpu().numpy()
                                    obs,_,terminal,truncated,info=env.step(np.clip(a,-1,1))
                                    if terminal or truncated:break
                            episode=dict(seed=seed,success=bool(info['success']),controls=len(rows),
                                subgoal_dones=info['subgoal_dones'].tolist(),balance_g=float(obs['object'][7]))
                        else:episode=evaluate_vial_trainer(env,trainer,seeds=(seed,),max_steps=1600)['episodes'][0]
                    finally:env.step=native_step
                episode['max_door_opening_m']=max(max(r['doors_m']) for r in rows)
                episode['video']=str(seed_dir/'rollout.mp4');results.append(episode)
                (seed_dir/'evaluation.json').write_text(json.dumps(episode,indent=2));(seed_dir/'trajectory.json').write_text(json.dumps(rows))
                log(dict(event='evaluation_seed',phase=name,actor_updates=trainer.actor_updates,**episode))
                run_volume.commit()
            result=dict(episodes=results,trials=len(results),successes=sum(r['success'] for r in results),
                actor_updates=trainer.actor_updates,checkpoint=str(root/f'checkpoint_{trainer.actor_updates:06d}'/'trainer.pt'),
                sampler='native_ODE' if native else 'OGPO_stochastic_chain')
            (directory/'summary.json').write_text(json.dumps(result,indent=2));run_volume.commit();return result
        finally:trainer.generator.set_state(rng);torch.set_rng_state(global_rng);torch.cuda.set_rng_state_all(cuda_rng)
    try:
        available=remaining_gpu_seconds-(time.monotonic()-started)
        if available<660:return dict(status='compute_time_limit',actor_updates=source_step,gpu_seconds=time.monotonic()-started)
        amount=min(500,remaining_updates)
        if amount:
            training=run_vial_online(env,trainer,replay,max_environment_steps=amount*25,max_updates=amount,
                max_seconds=min(2700,available-600),checkpoint_path=phase/'trainer.pt',seed=trainer.actor_updates+10000,
                checkpoint_every=50,on_update=log)
        else:
            training=dict(updates=0,actor_updates=trainer.actor_updates,phase='final_evaluation')
        (phase/'training_result.json').write_text(json.dumps(training,indent=2));run_volume.commit()
        achieved=trainer.actor_updates-initial
        evaluation=None;native=None;fresh=None
        # Evaluate progress without cancelling on a noisy score dip.
        if amount and achieved<TARGET and time.monotonic()-started<remaining_gpu_seconds-300:
            evaluation=evaluate('stochastic',SEEDS)
        # Final evaluation gets its own fresh worker, so reaching the training
        # target near a worker deadline cannot skip the requested evaluations.
        if not amount and achieved>=TARGET:
            evaluation=evaluate('stochastic',SEEDS)
            native=evaluate('native_ode',SEEDS,native=True)
            fresh=evaluate('fresh_stochastic',list(range(2001,2011)))
        complete=achieved>=TARGET and native is not None and fresh is not None
        result=dict(status='completed' if complete else 'evaluating' if achieved>=TARGET else 'resuming',initial_actor_updates=initial,
            actor_updates=trainer.actor_updates,additional_actor_updates=achieved,target_additional_actor_updates=TARGET,
            training=training,evaluation=evaluation,native_ode=native,fresh_stochastic=fresh,
            checkpoint=str(root/f'checkpoint_{trainer.actor_updates:06d}'/'trainer.pt'),gpu_seconds=time.monotonic()-started)
        (phase/'result.json').write_text(json.dumps(result,indent=2));save('latest_result.json',result)
        return result
    finally:replay.close();env.close();run_volume.commit();cache_volume.commit()

@app.function(image=image,cpu=.125,memory=512,timeout=86400,retries=0,volumes={'/runs':run_volume})
def coordinate(pilot_call_id,pilot_run,run_id):
    root=Path('/runs')/run_id;root.mkdir(exist_ok=False)
    state=dict(status='waiting_for_pilot',pilot_call_id=pilot_call_id,pilot_run=pilot_run,
        run_id=run_id,target_additional_actor_updates=TARGET,gpu_seconds_limit=None,
        gpu_seconds_used=0,segments=[],failures=[])
    def save():
        (root/'coordinator.json').write_text(json.dumps(state,indent=2));run_volume.commit()
        print(json.dumps(state),flush=True)
    save()
    try:result=modal.FunctionCall.from_id(pilot_call_id).get()
    except Exception as exc:
        state.update(status='pilot_failed',error=repr(exc));save();return state
    accepted,reason=pilot_gate(result);state['pilot_gate']=dict(accepted=accepted,reason=reason)
    if not accepted:state['status']='pilot_needs_review';save();return state
    state['status']='training';save();consecutive=0
    while True:
        tick=time.time()
        try:
            outcome=segment.spawn(run_id,pilot_run,WORKER_SECONDS).get()
            state['gpu_seconds_used']+=outcome['gpu_seconds'];state['segments'].append(outcome)
            consecutive=0
            if outcome['status']=='completed':state['status']='completed';save();return state
        except Exception as exc:
            state['gpu_seconds_used']+=time.time()-tick
            state['failures'].append(dict(error=repr(exc),time=time.time()));consecutive+=1
            # A fresh worker restores the latest complete model+optimizer+replay bundle.
            if consecutive>=3:state.update(status='recovery_failed',error=repr(exc));save();raise
        save()

@app.local_entrypoint()
def main(pilot_call_id:str,pilot_run:str='vial98-bc400k-vadar25-pilot-001',run_id:str='vial98-bc400k-vadar25-full-001'):
    verify_workspace()
    out=Path('docs/pilot')/run_id;out.mkdir(exist_ok=False)
    call=coordinate.spawn(pilot_call_id,pilot_run,run_id)
    (out/'function_call.json').write_text(json.dumps(dict(function_call_id=call.object_id),indent=2))
    result=call.get();(out/'coordinator.json').write_text(json.dumps(result,indent=2))
