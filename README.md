# Cyclic voltammetry starter: VaDar + OGPO

A CPU toy for trying [VaDar](https://arxiv.org/abs/2610.26629) (value disaggregation) with the policy update from [OGPO](https://arxiv.org/abs/2605.03065) (off-policy generative policy optimization) on a long-horizon lab task. Two arms weigh a solid, dispense electrolyte, seat cell and potentiostat leads, then let a voltammetry sweep finish.

This is an RL sandbox. It is not a Franka or UR driver, and it is not a procedure for a real electrochemical cell.

The official codebases ([OGPO_public](https://github.com/simchowitzlabpublic/OGPO_public), [vadar-public](https://github.com/simchowitzlabpublic/vadar-public)) are CUDA / Robomimic stacks. They are not vendored here.

## Run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m chemlab_rl.train
python -m unittest chemlab_rl.test_core
```

`python -m chemlab_rl.train` behavior-clones a scripted demonstrator, then fine-tunes online and prints per-subgoal success. The default run is a pipeline check, not a benchmark: a few epochs of cloning plus a cold critic will not solve the lab. `python -m unittest chemlab_rl.test_core` checks that the scripted demonstrator itself finishes every subgoal. Networks are 128–128 with an ensemble of 2; Appendix H.6 of VaDar uses widths 512 and an ensemble of 10.

## How the code maps onto the papers

VaDar is a wrapper around a value-based backbone. On the long-horizon RoboCasa tasks that backbone is OGPO. The split here is the same:

| File | Role |
| --- | --- |
| `chemlab_rl/env.py` | Dual-arm kinematic lab. `info["done"]` is the length-4 subgoal vector. The observation has poses and progress, and no current-goal index. |
| `chemlab_rl/vadar.py` | Per-subgoal sparse reward (Eq. 3.2), Table 1 weights, clipped TD targets (Eq. 5.2), shared-trunk vector critic, entrywise ensemble mean, EMA targets, replay of `(r, d)`. |
| `chemlab_rl/ogpo.py` | Squashed Gaussian, group-mean advantage (Eq. C.16), clipped PPO (Eq. C.17), behavior-cloning NLL. |
| `chemlab_rl/train.py` | Offline demos, BC warm start, then Algorithm 2: act with `pi_bar(s)`, update every critic head, aggregate with next-future weights, one OGPO actor step. |

The learned policy is `pi(a | s)`. Subgoal ids are a training-time critic input only, so deployment does not need a subgoal selector.

Default aggregation is **next-future**: every subgoal after the current unfinished one. `TrainConfig.weighting` can be `current_future`, `current_myopic`, or `next_myopic`.

## What is intentionally simplified

OGPO's importance ratio is a product over `K` denoising steps of a flow or diffusion policy. `SquashedGaussianPolicy.log_prob` is the `K = 1` case. A flow policy can replace that class if `sample` returns the executed action and `log_prob` returns the sum of per-step log densities. The trainer resamples actions from the target policy at buffer states; it does not run PPO on the stored environment actions.

Also left out:

- Flow matching, diffusion, and the noise injection OGPO uses so a flow step has a density
- Action chunks of length 8, best-of-N test-time selection, and the 20k-step critic warmup
- OGPO+ stabilizers: success-buffer regularization, two-sided conservative advantages, Q-variance reduction
- A goal-conditioned hierarchical baseline. Conditioning `pi` on a subgoal id would add that without changing the lab.
- Pixels, MuJoCo, and real robot control. Closing a gripper inside `REACH` snaps the nearest object; that is a contact proxy.

Subgoals, in the order next-future assumes:

1. `weigh_solid` — left arm, noisy mass in band
2. `dispense_electrolyte` — right arm, noisy volume in band
3. `seat_cables` — both arms, three leads (working, counter, reference)
4. `finish_sweep` — advances only after 1–3 are true
