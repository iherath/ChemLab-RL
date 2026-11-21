"""Optimizer/rollout integration tests using small actors, not VLA performance claims."""
from dataclasses import replace
import copy

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from chemlab_rl.ogpo_training import PhysicalActionAdapter, VectorQEnsemble
from chemlab_rl.smolvla_ogpo import ChainConfig
from chemlab_rl.vla_bc import CAMERAS, STATE_KEY, file_sha256
import chemlab_rl.vial_ogpo_training as vial


def observation(value=0):
    return dict(proprio=np.full(14, value, np.float32), scale_reading_g=np.array([0], np.float32),
                task="Place the vial on the balance.",
                **{key: np.full((4, 4, 3), value, np.uint8) for key in CAMERAS})


def row(value=0, **kwargs):
    return replace(vial.VialTransition(observation(value), observation(value+1),
                   np.full((2, 14), .1, np.float32), np.zeros(4, np.float32),
                   np.zeros(4, np.float32), np.zeros(2, np.float32), False, False, 2), **kwargs)


def preprocess(batch):
    n = len(batch[STATE_KEY])
    return {**batch, "observation.language.tokens": torch.ones(n, 3, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(n, 3, dtype=torch.bool)}


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(32))

    def forward(self, batch, latent, tau):
        return self.bias[None, None] - .2 * latent + .01 * batch[STATE_KEY][:, :1, None]


def trainer(method="ogpo", **kwargs):
    torch.manual_seed(10)
    cfg = vial.VialOGPOConfig(method=method, best_of_n=3, batch_size=2, group_size=3,
                              critic_warmup_updates=0, actor_microbatch_size=1,
                              actor_lr=1e-5, chain=ChainConfig(steps=2, diffusion_scale=.2))
    cfg = replace(cfg, **kwargs)
    critic = VectorQEnsemble(2, action_steps=cfg.action_steps, subgoals=cfg.subgoals, members=2, hidden=12)
    critic.heads[1].load_state_dict(critic.heads[0].state_dict())
    return vial.VialTrainer(Velocity(), critic, preprocessor=preprocess,
                            feature_fn=lambda batch: batch[STATE_KEY][:, :2],
                            action_adapter=PhysicalActionAdapter(lambda x: x, [-1]*14, [1]*14),
                            config=cfg, seed=123, provenance="shared-bc")


def replay(path):
    return vial.VialReplay(path, provenance="shared-bc", capacity=30, seed=2)


@pytest.mark.parametrize("method", ["bon", "ogpo", "vadar"])
def test_real_optimizer_updates_and_frozen_bon_actor(tmp_path, method):
    torch.set_num_threads(1)
    agent = trainer(method)
    data = replay(tmp_path / "replay.sqlite")
    for i in range(4):
        data.add(row(i))
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.critic.state_dict())
    metrics = agent.update(data)
    assert metrics["critic_updates"] == 1 and np.isfinite(metrics["critic_loss"])
    assert any(not torch.equal(critic_before[k], v) for k, v in agent.critic.state_dict().items())
    if method == "bon":
        assert metrics["actor_updates"] == 0
        assert all(torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
        assert all(not p.requires_grad for p in agent.actor.parameters())
    else:
        assert metrics["actor_updates"] == 1 and metrics["nonzero_advantage_fraction"] > 0
        assert any(not torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
        assert metrics["max_abs_log_ratio"] < 1e-7
    data.close()


@pytest.mark.parametrize("method", ["ogpo", "vadar"])
def test_scalar_and_auxiliary_targets_treat_success_and_time_limit_correctly(tmp_path, monkeypatch, method):
    agent = trainer(method, critic_warmup_updates=10, discount=.9)
    for model in (agent.critic, agent.target_critic):
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            for head in model.heads:
                head[-1].bias.fill_(2)
    data = replay(tmp_path / "replay.sqlite")
    data.add(row(1, terminated=True, next_dones=np.ones(4), rewards=np.array([0, 1])))
    data.add(row(2, truncated=True, executed_steps=1))
    targets = []
    original = vial.vector_critic_loss
    def capture(predicted, target):
        targets.extend(target.cpu().tolist())
        return original(predicted, target)
    monkeypatch.setattr(vial, "vector_critic_loss", capture)
    result = agent.update(data)
    assert result["truncated_fraction"] == .5 and result["actor_updates"] == 0
    actual = sorted(targets)
    expected = [[.9], [1.8]] if method == "ogpo" else [[.8]*4, [15.]*4]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    data.close()


def test_bon_uses_greedy_candidate_for_rollout_and_critic_target(tmp_path, monkeypatch):
    agent = trainer("bon", critic_warmup_updates=5)
    calls = []
    def selection(batch, features, count):
        calls.append(count)
        return torch.full((len(features), 2, 14), .4), torch.full((len(features), 2, 1), 3.)
    monkeypatch.setattr(agent, "_candidate_selection", selection)
    assert torch.all(agent.sample_actions([observation()]) == .4)
    data = replay(tmp_path / "replay.sqlite")
    data.add(row(0)); data.add(row(1))
    result = agent.update(data)
    assert calls == [3, 3]
    assert result["td_target_mean"] == pytest.approx(3 * .999**2)
    data.close()


class NativeContractEnvironment:
    instruction = "Place the vial on the balance."

    def reset(self, seed=None):
        self.t = 0
        return observation(), self.info()

    def info(self):
        success = self.t == 3
        return dict(task_id=vial.TASK_ID, subgoal_dones=np.array([self.t >= 1, self.t >= 2,
                     success, success], np.float32), success=success, balance_g=20. if success else 0.)

    def step(self, action):
        assert action.shape == (14,) and np.all(np.abs(action) <= 1)
        assert self.t < 3, "No controls after terminal"
        self.t += 1
        return observation(self.t), float(self.t == 3), self.t == 3, False, self.info()


def test_collector_keeps_actual_terminal_prefix_and_never_resets(tmp_path):
    agent, data, env = trainer(), replay(tmp_path / "replay.sqlite"), NativeContractEnvironment()
    obs, info = env.reset()
    obs, info, first = vial.collect_vial_chunk(env, obs, info, agent, data)
    obs, info, final = vial.collect_vial_chunk(env, obs, info, agent, data)
    assert first["executed_steps"] == 2 and final["executed_steps"] == 1
    assert final["terminated"] and env.t == 3 and agent.environment_steps == 3
    terminal = next(t for t in data.sample(2) if t.terminated)
    assert terminal.next_observation["proprio"][0] == 3
    np.testing.assert_array_equal(terminal.rewards, [1, 0])
    assert np.all(terminal.actions[1] == 0)
    data.close()


def test_runner_optimizer_rollout_evaluation_and_checkpoint_round_trip(tmp_path):
    agent, data = trainer(critic_warmup_updates=1), replay(tmp_path / "replay.sqlite")
    for i in range(4):
        data.add(row(i))
    env = NativeContractEnvironment()
    result = vial.run_vial_online(env, agent, data, max_environment_steps=3, max_updates=3,
                                  max_seconds=60, checkpoint_path=tmp_path / "trainer.pt")
    assert result["environment_steps"] == 3 and result["completed_episodes"] == 1
    assert result["online_successes"] == 1 and agent.actor_updates == 2
    rng = agent.generator.get_state().clone()
    evaluated = vial.evaluate_vial_trainer(env, agent, seeds=(10, 11), max_steps=10)
    assert evaluated["success_rate"] == 1 and torch.equal(agent.generator.get_state(), rng)
    restored = trainer(critic_warmup_updates=1)
    restored.load_checkpoint(tmp_path / "trainer.pt", data)
    assert restored.actor_updates == agent.actor_updates
    for key, value in restored.actor.state_dict().items():
        torch.testing.assert_close(value, agent.actor.state_dict()[key])
    data.close()


def test_runner_collection_cadence_keeps_all_controls_and_final_update(tmp_path):
    agent, data = trainer(critic_warmup_updates=1), replay(tmp_path / "cadence.sqlite")
    for i in range(4):
        data.add(row(i))
    result = vial.run_vial_online(NativeContractEnvironment(), agent, data,
                                  max_environment_steps=8, max_updates=10, max_seconds=60,
                                  update_every_chunks=2, checkpoint_path=tmp_path / "cadence.pt")
    assert result["environment_steps"] == 8 and agent.environment_steps == 8
    assert result["actor_updates"] == 3 and result["critic_updates"] == 4
    assert result["update_every_chunks"] == 2
    data.close()


def test_importer_excludes_heldout_and_discards_post_success_controls(tmp_path, monkeypatch):
    path = tmp_path / "demos.hdf5"
    pan = np.array([.1, .09, .864])
    joints = np.zeros((6, 14), np.float32)
    joints[2:, 6] = .5
    objects = np.zeros((6, 11), np.float32)
    objects[:, :3] = [-.12, .06, .775]
    objects[:, 3] = 1
    objects[1:4, 10] = .1
    objects[2, 2] = .9
    objects[3:, :3] = pan
    objects[3:, 7] = 20
    with h5py.File(path, "w") as f:
        for name in ("train", "heldout"):
            g = f.create_group("data/"+name)
            g.create_dataset("actions", data=np.zeros((5, 14), np.float32))
            g.create_dataset("dones", data=[0, 0, 0, 1, 1])
            for prefix, sl in (("obs", slice(None, -1)), ("next_obs", slice(1, None))):
                g.create_dataset(prefix+"/joint_pos", data=joints[sl])
                g.create_dataset(prefix+"/object", data=objects[sl])
                for key in ("top1_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"):
                    g.create_dataset(prefix+"/"+key, data=np.zeros((5, 4, 4, 3), np.uint8))
    manifest = dict(task_id=vial.TASK_ID, dataset_sha256=file_sha256(path),
                    train_episodes=["train"], heldout_episodes=["heldout"], episode_lengths={"train": 4})
    data = replay(tmp_path / "replay.sqlite")
    result = vial.import_vial_demonstrations(data, path, manifest=manifest,
                                            pan_position=pan, instruction="Place the vial.")
    assert result["effective_controls"] == 4 and result["transitions"] == 2
    assert len(data) == 2 and sum(t.terminated for t in data.sample(2)) == 1
    stored = [data._decode(payload) for (payload,) in data.db.execute("SELECT payload FROM transitions ORDER BY id")]
    with h5py.File(path) as f:
        for i, transition in enumerate(stored):
            before = vial.canonical_hdf_observation(f["data/train/obs"], 2*i, "Place the vial.")
            after = vial.canonical_hdf_observation(f["data/train/next_obs"], 2*i+1, "Place the vial.")
            for key in before:
                np.testing.assert_array_equal(transition.observation[key], before[key])
                np.testing.assert_array_equal(transition.next_observation[key], after[key])
    data.close()

    limited = vial.VialReplay(tmp_path / "limited.sqlite", provenance="test", capacity=1)
    vial.import_vial_demonstrations(limited, path, manifest=manifest, pan_position=pan, instruction="Place the vial.")
    assert len(limited) == 1 and limited.sample(1)[0].terminated
    limited.close()

    failed = replay(tmp_path / "failed.sqlite")
    original = failed._encode_transition
    calls = 0
    def fail_second(transition):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected serialization failure")
        return original(transition)
    monkeypatch.setattr(failed, "_encode_transition", fail_second)
    with pytest.raises(ValueError, match="injected serialization"):
        vial.import_vial_demonstrations(failed, path, manifest=manifest, pan_position=pan, instruction="Place the vial.")
    assert len(failed) == 0 and failed.checkpoint_state()["revision"] == 0
    failed.close()


def test_replay_rejects_success_without_termination_and_invalid_action_scale(tmp_path):
    data = replay(tmp_path / "replay.sqlite")
    for invalid in (row(rewards=np.ones(2)), row(next_dones=np.ones(4)), row(actions=np.full((2,14), 255.))):
        with pytest.raises(ValueError):
            data.add(invalid)
    assert len(data) == 0
    data.close()


def test_density_gate_recomputes_complete_matching_chain():
    result = vial.check_sampling_density(trainer(), [observation(), observation(1)])
    assert result["passed"] and result["ratio_min"] == pytest.approx(1.)
    assert result["chain_shape"] == [2, 3, 50, 32]


def test_25_control_collection_stops_at_success_and_masks_suffix(tmp_path):
    agent=trainer('vadar',action_steps=25)
    data=vial.VialReplay(tmp_path/'long.sqlite',provenance='shared-bc',action_steps=25)
    env=NativeContractEnvironment();obs,info=env.reset()
    _,_,result=vial.collect_vial_chunk(env,obs,info,agent,data)
    assert result['terminated'] and result['executed_steps']==3 and env.t==3
    stored=data.sample(1)[0]
    assert stored.actions.shape==(25,14) and stored.rewards.shape==(25,)
    assert stored.rewards[2]==1 and not stored.rewards[3:].any()
    assert not stored.actions[3:].any()
    evaluated=vial.evaluate_vial_trainer(env,agent,seeds=(1001,),max_steps=1600)
    assert evaluated['action_steps']==25 and evaluated['episodes'][0]['controls']==3
    assert evaluated['success_rate']==1
    data.close()


def test_25_control_td_discount_uses_actual_horizon(tmp_path,monkeypatch):
    agent=trainer('vadar',action_steps=25,critic_warmup_updates=10,discount=.9)
    for model in (agent.critic,agent.target_critic):
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            for head in model.heads:head[-1].bias.fill_(2)
    data=vial.VialReplay(tmp_path/'long.sqlite',provenance='shared-bc',action_steps=25)
    for h,truncated in [(25,False),(7,True)]:
        data.add(row(actions=np.zeros((25,14),np.float32),rewards=np.zeros(25,np.float32),
                     executed_steps=h,truncated=truncated))
    targets=[];original=vial.vector_critic_loss
    def capture(predicted,target):
        targets.extend(target.cpu().tolist());return original(predicted,target)
    monkeypatch.setattr(vial,'vector_critic_loss',capture)
    result=agent.update(data)
    assert result['actual_horizon_mean']==16
    np.testing.assert_allclose(sorted(targets),sorted([[-1+2*.9**25]*4,[-1+2*.9**7]*4]),atol=1e-6)
    agent.save_checkpoint(tmp_path/'trainer.pt',data)
    restored=trainer('vadar',action_steps=25,critic_warmup_updates=10,discount=.9)
    restored.load_checkpoint(tmp_path/'trainer.pt',data)
    assert restored.critic_updates==1
    data.close()


def test_demonstration_import_25_control_prefix_and_terminal_suffix(tmp_path):
    path=tmp_path/'demo.hdf5';length=27;pan=np.array([.1,.09,.864])
    joints=np.zeros((length,14),np.float32);joints[10:,6]=.5
    obj=np.zeros((length,11),np.float32);obj[:,3]=1;obj[:,2]=.8
    obj[:26,10]=.1;obj[10:26,2]=.9;obj[26,:3]=pan;obj[26,7]=20
    with h5py.File(path,'w') as f:
        g=f.create_group('data/train');g['actions']=np.zeros((length,14),np.float32)
        g['dones']=np.arange(length)==26
        for prefix in ('obs','next_obs'):
            g[prefix+'/joint_pos']=joints;g[prefix+'/object']=obj
            for key in ('top1_image','robot0_eye_in_hand_image','robot1_eye_in_hand_image'):
                g[prefix+'/'+key]=np.zeros((length,4,4,3),np.uint8)
    manifest=dict(task_id=vial.TASK_ID,dataset_sha256=file_sha256(path),train_episodes=['train'],episode_lengths={'train':27})
    data=vial.VialReplay(tmp_path/'long.sqlite',provenance='test',action_steps=25)
    result=vial.import_vial_demonstrations(data,path,manifest=manifest,pan_position=pan,instruction='Place vial')
    rows=[data._decode(p) for (p,) in data.db.execute('SELECT payload FROM transitions ORDER BY id')]
    assert result['transitions']==2 and result['effective_controls']==27
    assert [r.executed_steps for r in rows]==[25,2]
    assert rows[-1].terminated and rows[-1].rewards[1]==1
    assert not rows[-1].rewards[2:].any() and not rows[-1].actions[2:].any()
    data.close()
