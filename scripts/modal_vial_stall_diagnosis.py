"""Controlled chunk-execution and demonstration-prefix tests; no training."""
import json
from pathlib import Path
import modal
from scripts.modal_train import image,run_volume,data_volume,cache_volume,verify_workspace

app=modal.App('chemlab-vial-stall-diagnosis')

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1200,retries=0,
              scaledown_window=2,volumes={'/runs':run_volume,'/model-cache':cache_volume})
def evaluate_5k(run_id):
    import torch
    from scripts.evaluate_vial_vla import evaluate
    torch.set_num_threads(1)
    out=Path('/runs')/run_id
    if out.exists():raise FileExistsError(out)
    try:
        return evaluate('/runs/smolvla-vial98-a100-bc5000-001/checkpoint_005000',out,
                        seed=1000,max_steps=1600,n_action_steps=25)
    finally:
        run_volume.commit();cache_volume.commit()

@app.local_entrypoint()
def bc5k(run_id:str='vial98-bc5000-execute25-eval-001'):
    verify_workspace()
    out=Path('docs/pilot')/run_id;out.mkdir(exist_ok=False)
    result=evaluate_5k.remote(run_id)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1200,retries=0,
              scaledown_window=2,volumes={'/runs':run_volume,
              '/datasets':data_volume.with_mount_options(read_only=True),'/model-cache':cache_volume})
def diagnose(run_id):
    import h5py
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from chemlab_rl.vial_vla_data import VialVLABCPolicy,make_inference_batch
    from chemlab_rl.vial_vla_env import VialVLAEnv
    torch.set_num_threads(1)
    out=Path('/runs')/run_id;out.mkdir(exist_ok=False)
    actor=VialVLABCPolicy('/runs/smolvla-vial98-2h100-bc500000-004/checkpoint_400000',device='cuda',bf16=False)
    env=VialVLAEnv()
    report={'checkpoint_step':400000,'seed':1000,'trials':[]}
    with h5py.File('/datasets/weigh_vial_hybrid98.hdf5') as f:
        g=f['data/demo_0'];actions=g['actions'][:]
        obs,_=env.reset(seed=1000)
        def chunk(observation):
            actor.reset();torch.manual_seed(1000)
            with torch.inference_mode():
                return actor.postprocessor(actor.policy.predict_action_chunk(
                    actor.preprocessor(make_inference_batch(observation)))).cpu().numpy()[0]
        initial_chunk=chunk(obs)
        report['initial_chunk']={'predicted':initial_chunk.tolist(),'demonstrated':actions[:50].tolist(),
            'description':'50 predicted absolute normalized targets from live initial observation, fixed noise seed 1000'}
        # Reconstruct later states by physically replaying demonstration controls.
        # No qpos teleport: contacts, velocities and actuator state evolve normally.
        for execute,prefix,controls in [(2,0,1600),(10,0,1600),(25,0,1600),
                                        (2,50,400),(2,100,400),(2,200,400)]:
            label=f'execute{execute}_prefix{prefix}'
            obs,_=env.reset(seed=1000)
            for a in actions[:prefix]:obs,_,_,_,_=env.step(np.clip(a,-1,1))
            initial=obs['joint_pos'].copy()
            target=g['obs/joint_pos'][prefix]
            row={'label':label,'execute_actions':execute,'demonstration_prefix_controls':prefix,
                 'prefix_state_max_error_rad':float(np.abs(initial-target).max()),
                 'policy_controls_requested':controls,'trajectory':[]}
            if prefix:row['takeover_chunk']=chunk(obs).tolist()
            actor.policy.config.n_action_steps=execute
            actor.reset();torch.manual_seed(1000)
            writer=imageio.get_writer(out/f'{label}.mp4',fps=20,codec='libx264',quality=7,macro_block_size=1)
            try:
                for t in range(controls):
                    writer.append_data(np.concatenate([obs[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
                    action=actor.act(obs)
                    obs,_,terminated,truncated,info=env.step(np.clip(action,-1,1))
                    row['trajectory'].append({'control':t+1,'action':action.tolist(),
                        'joint_pos':obs['joint_pos'].tolist(),'doors_m':obs['object'][8:11].tolist(),
                        'subgoals':info['subgoal_dones'].tolist(),'success':bool(info['success']),
                        'max_joint_from_takeover_rad':float(np.abs(obs['joint_pos']-initial).max())})
                    if (t+1)%200==0:print(json.dumps({'event':'progress','trial':label,'control':t+1}),flush=True)
                    if terminated or truncated:break
                writer.append_data(np.concatenate([obs[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
            finally:writer.close()
            row['executed_controls']=len(row['trajectory'])
            row['max_joint_from_takeover_rad']=max(x['max_joint_from_takeover_rad'] for x in row['trajectory'])
            row['max_shield_opening_m']=max(max(x['doors_m']) for x in row['trajectory'])
            row['final_subgoals']=info['subgoal_dones'].tolist()
            row['success']=bool(info['success']);row['video']=label+'.mp4'
            report['trials'].append(row)
            (out/'diagnosis.json').write_text(json.dumps(report,indent=2));run_volume.commit()
            print(json.dumps({'event':'trial_result',**{k:v for k,v in row.items() if k not in ('trajectory','takeover_chunk')}}),flush=True)
    env.close();cache_volume.commit()
    return {k:v for k,v in report.items() if k!='initial_chunk'} | {'trials':[
        {k:v for k,v in r.items() if k not in ('trajectory','takeover_chunk')} for r in report['trials']]}

@app.local_entrypoint()
def main(run_id:str='vial98-bc400000-stall-diagnosis-001'):
    verify_workspace()
    out=Path('docs/pilot')/run_id;out.mkdir(exist_ok=False)
    result=diagnose.remote(run_id)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
