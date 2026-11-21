"""Predeclared sampling-seed evaluation of BC400k with 25-action execution."""
import json
from pathlib import Path
import modal
from scripts.modal_train import image,run_volume,cache_volume,verify_workspace

app=modal.App('chemlab-vial-bc-seed-evaluation')
SEEDS=list(range(1001,1011))
CHECKPOINT='/runs/smolvla-vial98-2h100-bc500000-004/checkpoint_400000'

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1800,retries=0,
              scaledown_window=2,volumes={'/runs':run_volume,'/model-cache':cache_volume})
def evaluate_seeds(run_id,action_steps=25):
    import time
    import torch
    import numpy as np
    import imageio.v2 as imageio
    from chemlab_rl.vial_vla_data import VialVLABCPolicy,file_sha256
    from chemlab_rl.vial_vla_env import VialVLAEnv,source_identity
    if action_steps not in (10,25):raise ValueError('This comparison uses 10 or 25 executed actions')
    torch.set_num_threads(1)
    out=Path('/runs')/run_id;out.mkdir(exist_ok=False)
    started=time.monotonic()
    request=dict(checkpoint=CHECKPOINT,seeds=SEEDS,n_action_steps=action_steps,chunk_size=50,
                 max_controls=1600,fixed_scene=True,precision='float32',
                 scope='Additional sampling seeds; initial scene is fixed. No training or scene randomization.',
                 checkpoint_manifest_sha256=file_sha256(Path(CHECKPOINT)/'data_manifest.json'),
                 simulation_source=source_identity())
    (out/'request.json').write_text(json.dumps(request,indent=2));run_volume.commit()
    torch.manual_seed(1000)
    actor=VialVLABCPolicy(CHECKPOINT,device='cuda',bf16=False)
    actor.policy.config.n_action_steps=action_steps
    env=VialVLAEnv(max_steps=1600)
    results=[]
    try:
        for seed in SEEDS:
            if time.monotonic()-started>1650:raise TimeoutError('Seed evaluation total time cap')
            directory=out/f'seed_{seed}';directory.mkdir()
            observation,info=env.reset(seed=seed)
            initial=observation['joint_pos'].copy()
            actor.reset();torch.manual_seed(seed);np.random.seed(seed)
            rows=[];maximum=0.;clipped=0;tick=time.monotonic()
            with imageio.get_writer(directory/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=1) as writer:
                for control in range(1,1601):
                    writer.append_data(np.concatenate([observation[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
                    action=np.asarray(actor.act(observation),dtype=np.float32)
                    if action.shape!=(14,) or not np.isfinite(action).all():raise FloatingPointError('Invalid action')
                    clipped+=int((np.abs(action)>1).sum())
                    observation,_,terminated,truncated,info=env.step(np.clip(action,-1,1))
                    doors=observation['object'][8:11]
                    maximum=max(maximum,float(doors.max()))
                    rows.append(dict(control=control,success=bool(info['success']),
                        joint_pos=observation['joint_pos'].tolist(),door_openings_m=doors.tolist(),
                        scale_reading_g=float(observation['object'][7]),subgoals=info['subgoal_dones'].tolist()))
                    if control%400==0:print(json.dumps(dict(event='progress',seed=seed,control=control,max_door_opening_m=maximum)),flush=True)
                    if terminated or truncated:break
                writer.append_data(np.concatenate([observation[k] for k in ('top1','left_wrist','right_wrist')],axis=1))
            result=dict(seed=seed,checkpoint=CHECKPOINT,n_action_steps=action_steps,chunk_size=50,
                completed_evaluation=bool(terminated or truncated),success=bool(info['success']),
                executed_controls=len(rows),termination_reason='native_success' if info['success'] else 'native_horizon',
                max_door_opening_m=maximum,scale_reading_g=float(observation['object'][7]),
                subgoal_dones=info['subgoal_dones'].tolist(),elapsed_seconds=time.monotonic()-tick,
                initial_joint_pos=initial.tolist(),clipped_action_fraction=clipped/(14*len(rows)),
                contact_certification=info['contact_certification'],
                max_postcontrol_penetration_m=info['max_postcontrol_penetration_m'],video=f'seed_{seed}/rollout.mp4')
            (directory/'evaluation.json').write_text(json.dumps(result,indent=2))
            (directory/'trajectory.json').write_text(json.dumps(rows))
            results.append(result)
            summary=dict(**request,status='completed' if len(results)==len(SEEDS) else 'running',
                completed_trials=len(results),successes=sum(r['success'] for r in results),
                success_rate=sum(r['success'] for r in results)/len(results),
                elapsed_seconds=time.monotonic()-started,results=results)
            (out/'summary.json').write_text(json.dumps(summary,indent=2));run_volume.commit()
            print(json.dumps(dict(event='seed_result',**result)),flush=True)
        return summary
    finally:
        env.close();run_volume.commit();cache_volume.commit()

@app.local_entrypoint()
def main(run_id:str='vial98-bc400000-execute25-seeds-001',action_steps:int=25):
    verify_workspace()
    out=Path('docs/pilot')/run_id;out.mkdir(exist_ok=False)
    (out/'request.json').write_text(json.dumps(dict(checkpoint=CHECKPOINT,seeds=SEEDS,n_action_steps=action_steps),indent=2))
    call=evaluate_seeds.spawn(run_id,action_steps)
    (out/'function_call.json').write_text(json.dumps(dict(function_call_id=call.object_id)))
    result=call.get()
    (out/'summary.json').write_text(json.dumps(result,indent=2))
