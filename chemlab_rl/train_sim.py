"""BC on Leo's weigh HDF5, then a short VaDar + one-step OGPO loop.

Paper OGPO (flow policy, CUDA, OGPO_public) is not used here: this machine has
no NVIDIA GPU, and that stack is Linux/CUDA. This file is the K=1 Gaussian
stand-in, with rollout GIFs and loss plots so we can see the pipeline move.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
_CLONE = ROOT / "chem_rl_sim"
sys.path.insert(0, str(_CLONE / "third_party" / "urgantry_sim"))
sys.path.insert(0, str(_CLONE))

import numpy as np
import torch

from chemlab_rl.ogpo import SquashedGaussianPolicy
from chemlab_rl.train import TrainConfig, _act, _behavior_clone, _store, _update_actor, _update_critic
from chemlab_rl.vadar import CriticEnsemble, ReplayBuffer, sparse_reward
from chemlab_rl.vial_env import SUBGOALS, VialWeighingAdapter, load_hdf5_demos

DEFAULT_HDF5 = ROOT / "data" / "teleop" / "weigh_vial_grok_v2.hdf5"


def run(hdf5: Path, cfg: TrainConfig, out_dir: Path, gif_every: int) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "device: cpu/mps Gaussian stand-in. "
        f"torch.cuda.is_available()={torch.cuda.is_available()} "
        "(paper OGPO needs Linux + CUDA + OGPO_public)."
    )

    data = load_hdf5_demos(hdf5)
    n_demos = int(data["demo_success"].shape[0])
    print(
        f"loaded {hdf5}\n"
        f"  demos={n_demos}  success={int(data['demo_success'].sum())}/{n_demos}  "
        f"transitions={data['obs'].shape[0]}"
    )
    if data["demo_success"].sum() < 1:
        raise SystemExit("HDF5 has no successful demo")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = VialWeighingAdapter(seed=cfg.seed, horizon=cfg.horizon)
    replay = ReplayBuffer(cfg.replay_size, env.obs_dim, env.action_dim, env.n_subgoals)
    for i in range(data["obs"].shape[0]):
        replay.add(
            data["obs"][i],
            data["actions"][i],
            data["next_obs"][i],
            data["rewards"][i],
            data["dones"][i],
            float(data["finals"][i]),
        )

    policy = SquashedGaussianPolicy(env.obs_dim, env.action_dim, cfg.hidden)
    bc_loss, bc_hist = _behavior_clone(policy, data["obs"], data["actions"], cfg, rng)
    print(f"BC loss={bc_loss:.3f}  (epochs={cfg.bc_epochs})")
    _eval_bc(policy, data)

    target_policy = copy.deepcopy(policy)
    critic = CriticEnsemble(env.obs_dim, env.action_dim, env.n_subgoals, cfg.n_critics, cfg.hidden)
    target_critic = copy.deepcopy(critic)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    log = {
        "bc_epoch_loss": bc_hist,
        "critic_loss": [],
        "actor_loss": [],
        "env_step": [],
        "episode": [],
        "success": [],
        "n_subgoals": [],
        "ep_reward": [],
        "dones": {name: [] for name in SUBGOALS},
        "gifs": [],
    }

    def snapshot(tag: str) -> None:
        gif = _rollout_gif(target_policy, env, tag, out_dir, cfg.seed + 7)
        log["gifs"].append(str(gif))
        print(f"  saved {gif}")

    snapshot("00_after_bc")

    print(
        f"VaDar {cfg.weighting} + one-step OGPO  "
        f"online_episodes={cfg.online_episodes} horizon={cfg.horizon}"
    )
    successes = 0
    env_steps = 0
    updates = 0
    last = np.zeros(env.n_subgoals, dtype=np.float32)
    for episode in range(cfg.online_episodes):
        obs = env.reset(seed=cfg.seed + 1000 + episode)
        info = {"success": False, "done": last, "terminated": False, "truncated": False}
        ep_reward = 0.0
        for _ in range(env.horizon):
            done_s = env.done_vector()
            action = _act(target_policy, obs)
            next_obs, info = env.step(action)
            _store(replay, obs, action, next_obs, done_s, info)
            ep_reward += float(sparse_reward(info["done"]).mean())
            obs = next_obs
            env_steps += 1
            if len(replay) >= cfg.batch_size:
                c_loss = _update_critic(
                    replay, rng, policy, critic, target_critic, critic_opt, cfg
                )
                a_loss = _update_actor(
                    replay, rng, policy, target_policy, target_critic, actor_opt, cfg
                )
                log["critic_loss"].append(c_loss)
                log["actor_loss"].append(a_loss)
                log["env_step"].append(env_steps)
                updates += 1
            if info["terminated"] or info["truncated"]:
                break
        last = np.asarray(info["done"], dtype=np.float32)
        successes += int(info["success"])
        log["episode"].append(episode)
        log["success"].append(int(info["success"]))
        log["n_subgoals"].append(int(last.sum()))
        log["ep_reward"].append(ep_reward)
        for name, flag in zip(SUBGOALS, last):
            log["dones"][name].append(int(flag))
        bits = " ".join(f"{name}={int(flag)}" for name, flag in zip(SUBGOALS, last))
        print(
            f"episode {episode:03d}  success={int(info['success'])}  "
            f"t={info['t']}  reward={ep_reward:.1f}  {bits}"
        )
        if gif_every > 0 and (episode + 1) % gif_every == 0:
            snapshot(f"{episode + 1:02d}_ep{episode:03d}")

    if gif_every <= 0 or cfg.online_episodes % gif_every != 0:
        snapshot(f"{cfg.online_episodes:02d}_final")

    env.close()
    plot_path = _write_plots(log, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))
    print(
        f"online success {successes}/{cfg.online_episodes}  "
        f"env_steps={env_steps}  updates={updates}"
    )
    print(f"plots {plot_path}")
    print(
        "This Gaussian is not paper OGPO. Do not treat success rate as a paper result."
    )
    return {
        "bc_loss": bc_loss,
        "successes": successes,
        "env_steps": env_steps,
        "updates": updates,
        "n_demos": n_demos,
        "out_dir": str(out_dir),
    }


def _eval_bc(policy: SquashedGaussianPolicy, data: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        n = min(1024, data["obs"].shape[0])
        idx = np.random.default_rng(0).choice(data["obs"].shape[0], size=n, replace=False)
        obs = torch.from_numpy(data["obs"][idx])
        act = torch.from_numpy(data["actions"][idx])
        mu, _std = policy._mu_std(obs)
        pred = torch.tanh(mu)
        mae = (pred - act).abs().mean().item()
    print(f"BC action MAE (random {n} steps)={mae:.3f}")


def _rollout_gif(policy, env: VialWeighingAdapter, tag: str, out_dir: Path, seed: int) -> Path:
    import imageio
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    obs = env.reset(seed=seed)
    states = [env.mj_state()]
    info = {"success": False, "done": np.zeros(env.n_subgoals, dtype=np.float32), "t": 0}
    for _ in range(env.horizon):
        obs, info = env.step(_act(policy, obs))
        states.append(env.mj_state())
        if info["terminated"] or info["truncated"]:
            break
    bits = "".join(str(int(v)) for v in info["done"])
    title = f"{tag}  t={info['t']}  done={bits}  success={int(info['success'])}"

    m, d = env._env.model, env._env.data
    nq, nv = m.nq, m.nv
    renderer = mujoco.Renderer(m, height=480, width=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.10, 0.04, 0.88)
    cam.distance, cam.azimuth, cam.elevation = 1.15, 125.0, -35.0
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    arr = np.stack(states)
    for i in range(0, arr.shape[0], 8):
        st = arr[i]
        d.time = float(st[0])
        d.qpos[:] = st[1 : 1 + nq]
        d.qvel[:] = st[1 + nq : 1 + nq + nv]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 34), fill=(20, 20, 20))
        draw.text((10, 6), title, fill=(240, 240, 240), font=font)
        frames.append(np.asarray(img))
    renderer.close()

    mp4 = out_dir / f"rollout_{tag}.mp4"
    gif = out_dir / f"rollout_{tag}.gif"
    writer = imageio.get_writer(mp4, fps=10, macro_block_size=1)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        "fps=10,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    )
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)
    return gif


def _write_plots(log: dict, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(log["bc_epoch_loss"]) + 1), log["bc_epoch_loss"], color="#1f4e79")
    ax.set_title("BC loss (NLL)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean batch NLL")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if log["env_step"]:
        ax.plot(log["env_step"], log["critic_loss"], color="#b85c38", lw=0.6, alpha=0.5, label="critic")
        ax.plot(log["env_step"], log["actor_loss"], color="#2a6f4e", lw=0.6, alpha=0.5, label="actor")
        ax.set_title("Online losses")
        ax.set_xlabel("env step")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(log["episode"], log["ep_reward"], color="#1f4e79", marker="o", ms=3)
    ax.set_title("Episode reward (mean sparse r)")
    ax.set_xlabel("episode")
    ax.set_ylabel("sum_t mean_g r_g")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for name, ys in log["dones"].items():
        ax.plot(log["episode"], ys, marker="o", ms=3, label=name)
    ax.plot(log["episode"], log["success"], color="black", ls="--", label="task success")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Subgoals completed (latched)")
    ax.set_xlabel("episode")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("VaDar + one-step OGPO on Grok weigh demos (Gaussian, not paper OGPO)")
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "train_run")
    p.add_argument("--bc-epochs", type=int, default=12)
    p.add_argument("--online-episodes", type=int, default=12)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--gif-every", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.hdf5.exists():
        raise SystemExit(f"missing {args.hdf5}")
    cfg = TrainConfig(
        seed=args.seed,
        bc_epochs=args.bc_epochs,
        online_episodes=args.online_episodes,
        horizon=args.horizon,
        batch_size=args.batch_size,
        replay_size=200_000,
        quiet=True,
    )
    run(args.hdf5, cfg, args.out_dir, args.gif_every)


if __name__ == "__main__":
    main()
