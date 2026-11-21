"""Bounded VaDar25 pilot with matched sampler baselines and seed evaluations."""
from dataclasses import asdict
import json
from pathlib import Path
import modal
from scripts.modal_train import image,run_volume,data_volume,cache_volume,verify_workspace

app=modal.App('chemlab-vial-vadar25-pilot')
SOURCE='/runs/smolvla-vial98-2h100-bc500000-004/checkpoint_400000'
SEEDS=list(range(1001,1011))

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=49152,timeout=2700,retries=0,
              scaledown_window=2,volumes={'/runs':run_volume,'/datasets':data_volume.with_mount_options(read_only=True),
                                         '/model-cache':cache_volume})
def pilot(run_id):
    import hashlib
    import time
    import numpy as np
    import torch
    import imageio.v2 as imageio
    import h5py
    from chemlab_rl.vial_vla_data import load_smolvla,make_inference_batch,file_sha256
    from chemlab_rl.vial_vla_env import VialVLAEnv,source_identity
    from chemlab_rl.vial_ogpo_training import (VialOGPOConfig,VialTrainer,VialReplay,canonical_hdf_observation,
        check_sampling_density,import_vial_demonstrations,run_vial_online,evaluate_vial_trainer)
    from chemlab_rl.ogpo_training import FrozenSmolVLAFeatures,VectorQEnsemble,PhysicalActionAdapter
    from chemlab_rl.smolvla_ogpo import SmolVLAVelocity,ChainConfig
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    started=time.monotonic();out=Path('/runs')/run_id;out.mkdir(exist_ok=False)
    def log(row):
        print(json.dumps(row),flush=True)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2));run_volume.commit()
    manifest=json.loads((Path(SOURCE)/'data_manifest.json').read_text())
    dataset='/datasets/weigh_vial_hybrid98.hdf5'
    if file_sha256(dataset)!=manifest['dataset_sha256']:raise ValueError('Dataset mismatch')
    config=VialOGPOConfig(method='vadar',action_steps=25,batch_size=2,group_size=2,actor_microbatch_size=2,
        actor_lr=1e-9,critic_lr=3e-4,critic_warmup_updates=100,cache_prefix=True,
        chain=ChainConfig(steps=10,diffusion_scale=.05))
    proof=dict(checkpoint=SOURCE,weights_sha256=file_sha256(Path(SOURCE)/'model.safetensors'),
        dataset_sha256=manifest['dataset_sha256'],simulation_source=source_identity(),config=asdict(config),
        evaluation_seeds=SEEDS,online_controls=3200,max_updates=228,training_seconds_cap=900,
        auxiliary_step_cost='-1 per executed chunk, reference convention',discount='0.999 per control, exponent actual executed controls')
    provenance=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
    save('launch.json',proof)
    policy,pre,post=load_smolvla(manifest,'cuda',SOURCE,chunk_size=50,n_action_steps=25)
    features=FrozenSmolVLAFeatures(policy,15)
    critic=VectorQEnsemble(features.feature_dim,action_steps=25,subgoals=4,members=2,hidden=256)
    trainer=VialTrainer(SmolVLAVelocity(policy,cache_prefix=True),critic,preprocessor=pre,feature_fn=features,
        action_adapter=PhysicalActionAdapter(post,[-1]*14,[1]*14),config=config,device='cuda',seed=0,provenance=provenance)
    with h5py.File(dataset) as f:
        g=f['data'][manifest['train_episodes'][0]]['obs']
        inputs=[canonical_hdf_observation(g,i,manifest['instruction']) for i in range(2)]
    density=check_sampling_density(trainer,inputs);save('density_check.json',density)
    if not density['passed']:raise RuntimeError('Density consistency failed; no training started')
    env=VialVLAEnv(max_steps=1600)
    replay=VialReplay(out/'replay.sqlite',provenance=provenance,capacity=30000,action_steps=25,seed=0)
    def evaluate(name,native=False):
        results=[];root=out/name;root.mkdir()
        log(dict(event='phase',name=name))
        rng=trainer.generator.get_state();global_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:
            for seed in SEEDS:
                directory=root/f'seed_{seed}';directory.mkdir()
                native_step=env.step;rows=[]
                with imageio.get_writer(directory/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=1) as writer:
                    def recorded_step(action):
                        result=native_step(action);obs,_,_,_,info=result
                        writer.append_data(np.concatenate([obs[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
                        rows.append(dict(control=len(rows)+1,door_openings_m=obs['object'][8:11].tolist(),
                            scale_g=float(obs['object'][7]),subgoals=info['subgoal_dones'].tolist(),success=bool(info['success'])))
                        return result
                    env.step=recorded_step
                    try:
                        if native:
                            obs,info=env.reset(seed=seed);p=trainer.target_actor.policy;p.reset();torch.manual_seed(seed)
                            with torch.inference_mode():
                                for _ in range(1600):
                                    action=post(p.select_action(pre(make_inference_batch(obs))))[0].cpu().numpy()
                                    obs,_,terminated,truncated,info=env.step(np.clip(action,-1,1))
                                    if terminated or truncated:break
                            episode=dict(seed=seed,success=bool(info['success']),controls=len(rows),
                                subgoal_dones=info['subgoal_dones'].tolist(),balance_g=float(obs['object'][7]))
                        else:
                            episode=evaluate_vial_trainer(env,trainer,seeds=(seed,),max_steps=1600)['episodes'][0]
                    finally:env.step=native_step
                episode['max_door_opening_m']=max(max(r['door_openings_m']) for r in rows)
                episode['video']=f'{name}/seed_{seed}/rollout.mp4'
                (directory/'evaluation.json').write_text(json.dumps(episode,indent=2))
                (directory/'trajectory.json').write_text(json.dumps(rows))
                results.append(episode);log(dict(event='evaluation_seed',phase=name,**episode))
                save(name+'.json',dict(episodes=results,successes=sum(r['success'] for r in results),
                    trials=len(results),sampler='native_ODE' if native else 'OGPO_stochastic_chain',action_steps=25))
        finally:
            trainer.generator.set_state(rng);torch.set_rng_state(global_rng);torch.cuda.set_rng_state_all(cuda_rng)
        return dict(successes=sum(r['success'] for r in results),trials=len(results),episodes=results)
    try:
        baseline=evaluate('baseline_stochastic')
        if baseline['successes']==0:
            result=dict(status='sampler_gate_failed',baseline=baseline,optimizer_updates=0,
                reason='Stochastic sampler lost all baseline success; calibrate sampling before training')
            save('result.json',result);return result
        log(dict(event='phase',name='demonstration_import'))
        imported=import_vial_demonstrations(replay,dataset,manifest=manifest,pan_position=env.pan_position,
            instruction=env.instruction,on_progress=log);save('demonstration_import.json',imported)
        log(dict(event='phase',name='pilot_training'))
        training=run_vial_online(env,trainer,replay,max_environment_steps=3200,max_updates=228,max_seconds=900,
            checkpoint_path=out/'trainer.pt',seed=0,checkpoint_every=50,on_update=log)
        save('training_result.json',training)
        post_density=check_sampling_density(trainer,inputs);save('post_density_check.json',post_density)
        stochastic=evaluate('finetuned_stochastic')
        native=evaluate('finetuned_native_ode',native=True)
        export=out/'policy_ema';trainer.target_actor.policy.save_pretrained(export)
        (export/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
        result=dict(status='completed',baseline_stochastic=baseline,finetuned_stochastic=stochastic,
            finetuned_native_ode=native,existing_bc_native_ode_successes=7,existing_bc_native_ode_trials=10,
            training=training,density_check=density,post_density_check=post_density,
            elapsed_seconds=time.monotonic()-started,checkpoint=str(out/'trainer.pt'),ema_policy=str(export))
        save('result.json',result);return result
    finally:replay.close();env.close();run_volume.commit();cache_volume.commit()

@app.local_entrypoint()
def main(run_id:str='vial98-bc400k-vadar25-pilot-001'):
    verify_workspace()
    out=Path('docs/pilot')/run_id;out.mkdir(exist_ok=False)
    call=pilot.spawn(run_id)
    (out/'function_call.json').write_text(json.dumps(dict(function_call_id=call.object_id)))
    result=call.get();(out/'result.json').write_text(json.dumps(result,indent=2))
