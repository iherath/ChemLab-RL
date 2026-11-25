# Vial-placement QC final results: November 24, 2026

QC training completed at **3:28 p.m. EST on November 24**, early-stopping at **175,000 critic and actor updates**. The selected 25K checkpoint and the final 175K policy each achieved **20/20 successes** on the final stochastic rollout seeds. No QC training job remains active for this completed run.

[Open the plots and video index](results/vial_qc_bc400k_20261124/README.md).

## Results

| Evaluation | Updates | Successes |
| --- | ---: | ---: |
| Original BC baseline | 0 QC | 9/10 |
| QC selection | 25,000 | 10/10 |
| QC selection | 50,000 | 10/10 |
| QC selection | 75,000 | 10/10 |
| QC selection | 100,000 | 9/10 |
| QC selection | 125,000 | 10/10 |
| QC selection | 150,000 | 9/10 |
| QC selection | 175,000 | 10/10 |
| Selected 25K policy, final set | 25,000 | **20/20** |
| Last QC policy, final set | 175,000 | **20/20** |

Selection used seeds 1001–1010; final testing used 2001–2020 and never drove checkpoint selection or stopping. Ties retained the earlier checkpoint. Both final policies were tested on the same 20 seeds, so their results should not be pooled as 40 independent observations.

The initial scene was fixed, and there was one training seed. These results measure stochastic rollout reliability in that scene. They do not establish held-out scene or powder generalization, or a statistically demonstrated improvement over BC. The 95% Wilson interval for either 20/20 score is 83.9%–100%; the BC baseline used a different ten-seed evaluation set.

## Method and stopping

The run starts from Michael's original BC400K SmolVLA checkpoint. The method is QC with native-ODE Best-of-N chunk selection, N = 8, plus continued flow-matching behavior cloning on demonstration-seeded and accumulated online replay. It predicts 50 controls and executes 25 before replanning, with the VLM frozen and BF16 inference/training. The mathematical protocol and simulation sources remained unchanged throughout recovery.

A 10/10 result at 25K selected the best checkpoint. The configured stopping rule began counting patience at 100K and required four completed evaluations without a strictly higher success count. Those checks were 100K, 125K, 150K and 175K, producing `stop_reason=selection_plateau`. This conservative minimum explains why training continued after reaching 10/10. The user-requested 500K limit was an upper bound.

The run collected 173,012 training environment controls and completed 120 evaluation episodes: 80 baseline/selection episodes and 40 final-policy episodes. A coordinator preemption and a GPU fault interrupted execution; both were recovered using complete checkpoints with the original budget and deadline preserved. Interrupted, uncheckpointed work was replayed. No final result is inferred from an interrupted episode.

## Delivered artifacts

The [final results bundle](results/vial_qc_bc400k_20261124/) includes success curves against optimizer updates, environment controls and recorded training time; vial/tool trajectories, scale and door measurements, joint/action plots; PNG/PDF/CSV exports; and baseline, selected-policy and final-policy videos.

The 200K/400K milestones were not reached, so no videos at those steps exist. Final selected-25K and last-175K rollout videos replace them as the completed-run demonstrations. Earlier 25K selection and 100K progress plots are separated under `historical/` in the bundle. The 100K copies are named `latest_u100000_*` to distinguish them from the final 175K test plots.

## Checkpoint identity

Run ID: `vial-qc-bc400k-h100-main-20261122-003`. Volume: `chemlab-rl-runs`. Checkpoint paths are under this run's `checkpoints/` directory.

| Role | Bundle | Evaluated trainer SHA256 |
| --- | --- | --- |
| Selected 25K policy | `generation_00000010` | `57bbcf99f5bf5ecdb0a69fbb04a57ee914d9ee0ba6da5b958828c567d72ce683` |
| Last 175K policy, final-test snapshot | `generation_00000067` | `03466bd2419b05c98ba57b0f53d07ff0d7f765f5d42b6a1b3eb102e38ff3794f` |
| Completed training and replay state | `generation_00000072` | See its committed bundle marker |

The original BC400K base weights and processors are required alongside the saved policy deltas. Their remote path is `/runs/shared-bc400k-20261121/checkpoint_400000`. Exact evaluation summaries and the committed coordinator/export indexes are the source of these results; the final export reports `status=rendered`, `partial=false`, ten evaluation records and 120 trials.
