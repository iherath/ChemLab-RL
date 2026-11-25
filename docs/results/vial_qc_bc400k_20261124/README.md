# Vial-placement QC: final results

**Training completed on November 24, 2026, at 3:28 p.m. EST.** It stopped at **175,000 critic and actor updates** under the configured plateau rule. Both the selected 25K checkpoint and the last 175K policy succeeded in **20/20 final rollouts**.

Run: `vial-qc-bc400k-h100-main-20261122-003`. The task is native MuJoCo vial placement, starting from Michael's BC400K SmolVLA checkpoint. QC uses Best-of-N action-chunk selection with N = 8 and continued replay behavior cloning.

## Measured outcomes

| Policy | Selection seeds 1001–1010 | Final seeds 2001–2020 |
| --- | ---: | ---: |
| Original BC400K | 9/10 | Not evaluated in this final set |
| Selected QC, 25K updates | 10/10 | **20/20** |
| Last QC, 175K updates | 10/10 | **20/20** |

The selection curve is 10/10 at 25K, 50K and 75K; 9/10 at 100K; 10/10 at 125K; 9/10 at 150K; and 10/10 at 175K. Ties retain the first best checkpoint, so 25K remained selected. Final results did not affect selection or stopping.

These are stochastic rollout seeds in the **same fixed initial scene**, with **one training seed**. The two final policy scores use the same 20 seeds and should not be pooled as 40 independent trials. The episode-level 95% Wilson interval for 20/20 is **83.9%–100%**. The experiment does not establish held-out scene or powder generalization, or a statistically demonstrated improvement over BC.

## Success curves

| Horizontal axis | Image | PDF | Data |
| --- | --- | --- | --- |
| Critic optimizer updates | [PNG](plots/success_vs_updates.png) | [PDF](plots/success_vs_updates.pdf) | [CSV](plots/success_vs_updates.csv) |
| Training environment controls | [PNG](plots/success_vs_env_controls.png) | [PDF](plots/success_vs_env_controls.pdf) | [CSV](plots/success_vs_env_controls.csv) |
| Recorded training time | [PNG](plots/success_vs_walltime.png) | [PDF](plots/success_vs_walltime.pdf) | [CSV](plots/success_vs_walltime.csv) |

[Evaluation table](plots/evaluations.csv). Curves contain selection evaluations only; final tests are reported separately. Shading represents episode-level Wilson intervals, not variation across training seeds. Recorded training time is not a billing meter or the full calendar duration including interruptions.

## Rollout trajectories and actions

Trajectory figures show measured vial/tool paths, scale readings and door positions. Matching PDF figures and trajectory CSVs are included in the [plots folder](plots/).

| Policy and rollout | Outcome | Trajectory | Joints and actions |
| --- | --- | --- | --- |
| BC, seed 1001 | Success | [Plot](plots/baseline_u000000_seed1001_trajectory.png) | [Plot](plots/baseline_u000000_seed1001_joint_actions.png) |
| BC, seed 1002 | Success | [Plot](plots/baseline_u000000_seed1002_trajectory.png) | [Plot](plots/baseline_u000000_seed1002_joint_actions.png) |
| BC, seed 1004 | Failure | [Plot](plots/baseline_u000000_seed1004_trajectory.png) | [Plot](plots/baseline_u000000_seed1004_joint_actions.png) |
| Selected QC at 25K, final seed 2001 | Success | [Plot](plots/best_u025000_seed2001_trajectory.png) | [Plot](plots/best_u025000_seed2001_joint_actions.png) |
| Selected QC at 25K, final seed 2002 | Success | [Plot](plots/best_u025000_seed2002_trajectory.png) | [Plot](plots/best_u025000_seed2002_joint_actions.png) |
| Last QC at 175K, final seed 2001 | Success | [Plot](plots/final_u175000_seed2001_trajectory.png) | [Plot](plots/final_u175000_seed2001_joint_actions.png) |
| Last QC at 175K, final seed 2002 | Success | [Plot](plots/final_u175000_seed2002_trajectory.png) | [Plot](plots/final_u175000_seed2002_joint_actions.png) |

Earlier selection plots are separated into [historical 25K selection rollouts](historical/25k_selection/) and [historical 100K rollouts](historical/100k/). The 100K copies use `latest_u100000_*` names to distinguish that interim snapshot from the completed run. The primary plots above show the actual final tests on seeds 2001/2002.

## Videos

| Policy | Seed 1 | Seed 2 |
| --- | --- | --- |
| Original BC | [1001](videos/bc_seed1001.mp4) | [1002](videos/bc_seed1002.mp4) |
| Selected QC at 25K | [2001](videos/selected25k_seed2001.mp4) | [2002](videos/selected25k_seed2002.mp4) |
| Last QC at 175K | [2001](videos/qc175k_seed2001.mp4) | [2002](videos/qc175k_seed2002.mp4) |

There are no 200K or 400K milestone videos because early stopping ended the run at 175K. Final rollout videos are supplied above.

## Stopping and saved policies

Patience counted four completed evaluations without a strictly higher success count, beginning at 100K: 100K, 125K, 150K and 175K. A 10/10 score selected the 25K checkpoint but did not immediately stop training. The 500K limit was a maximum, not a required run length.

Checkpoints remain on the Modal volume `chemlab-rl-runs`, under `vial-qc-bc400k-h100-main-20261122-003/checkpoints/`:

| Artifact | Generation |
| --- | --- |
| Selected 25K policy, evaluated on the final set | `generation_00000010` |
| 175K policy snapshot bound to the final QC evaluation | `generation_00000067` |
| Completed training state and replay bundle | `generation_00000072` |

These checkpoints store policy deltas, critics and training state. Load them through the project runner with the original BC400K base checkpoint and saved processors at `/runs/shared-bc400k-20261121/checkpoint_400000`.

A coordinator preemption and a later GPU fault interrupted execution. Recovery retained the original model, simulation and training protocol sources, the existing allocation ledger and the absolute deadline. Final export completed successfully and contains all ten evaluation records: 80 selection/baseline episodes and 40 final-policy episodes.
