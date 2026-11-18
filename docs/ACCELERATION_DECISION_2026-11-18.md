# November 18 collection and simulation acceleration

The user requested more Modal compute to obtain BC and OGPO–VaDar results by
midnight, with a simpler powder representation as a fallback. The physical
requirements remain: actuator-driven manipulation, valid contact/support,
conserved material, density-consistent mass, and an independently measured
balance load.

At approximately16:15 EST the user explicitly chose: **Keep full physics;
report BC and whatever RL completes.** The reduced-model and shorter-task
alternatives below are documented possibilities, not authorized changes to the
current experiment. Prioritize accepted physical data and measured BC results.

## Decision for the current physical model

Keep the running complete 200-grain reference test and its frozen physical
sources. After acceptance, submit all 29 remaining demonstrations concurrently.
Merge the first five as soon as their four new episodes and the reused reference
are accepted. This permits BC5 to start while the remaining episodes finish.
Merge the full 30 only when all 30 pass; a failure outside the pilot must not
discard a valid pilot. BC5 and BC30 may overlap as two separately bounded A100
jobs. Preserve unsuccessful episodes and do not retry automatically.

The previous five-demo gate before starting the remaining 25 was our conservative
scheduling choice. The user has now prioritized deadline throughput, and the
cost of the additional simultaneous jobs is modest. This change affects
scheduling, not the trajectories, targets, material split, physics settings,
acceptance tests, or source identity. The explicit complete-reference gate
remains.

Published Modal pricing on November 18 gives $0.0000131 per physical CPU-core
second and $0.00000222 per GiB-second. With request and limit both at two cores
and 12 GiB, each collector costs at most approximately $0.190224 per compute
hour. Twenty-nine two-hour calls are approximately $11.03. Two one-hour merges
add approximately $0.38; five minutes of startup per 31 calls would add $0.49.
These are resource-cost calculations, not measured billing. Storage, other
existing runs, and GPU jobs are separate. Running the same jobs concurrently
does not multiply their CPU-hours; it can spend more on failed parameter
variants before an early failure is discovered. Published Starter limits allow
100 containers and ten GPUs, but workspace-specific overrides remain unverified.

Sources: [Modal pricing](https://modal.com/pricing),
[resource configuration](https://modal.com/docs/guide/resources).

## Local measurements

A short holding probe preserves the initialized state, controls, timestep,
solver settings, and every-substep validation. It is a throughput experiment,
not evidence that an entire manipulation trajectory is valid.

| Variant | Seconds per 0.1 simulated second |
|---|---:|
| One thread, images off | 2.840 |
| One thread, three 224×224 camera images | 2.907 |
| Same cameras with 200 grain geoms visually hidden | 2.925 |

Camera generation costs about 80 ms per control. Hidden grains save only a few
milliseconds within rendering and provide no measurable overall gain. Hiding
grains would also remove powder appearance from the VLA's observations.

A separate matched thread-count probe measured 2.820, 2.900, and 2.812 seconds
per control with one, two, and four total threads. A single coupled constraint
island contains more than 96% of constraints. The sampled physical states and
sensor arrays were bitwise identical between variants, but no useful speedup
was observed. These are short local measurements, not a Modal CPU benchmark.

Installed MuJoCo 3.16 provides `mju_threadpool(data, workers)`, where workers
are additional to the caller. It uses its own threads; setting an OpenMP
environment variable does not activate this pool. The engine can parallelize
collision work and independent constraint islands. Time steps along one
trajectory remain sequential.

Sources: [MuJoCo 3.16 simulation guide](https://github.com/google-deepmind/mujoco/blob/3.16.0/doc/programming/simulation.rst),
[thread implementation](https://github.com/google-deepmind/mujoco/blob/3.16.0/src/engine/engine_thread.cc),
[solver dispatch](https://github.com/google-deepmind/mujoco/blob/3.16.0/src/engine/engine_forward.c).

## Deadline implications

Concurrent collection can reduce the remaining collection time to roughly one
episode duration plus merging, subject to successful execution and service
capacity. It does not shorten a single evaluation or serial online RL episode.
Independent evaluations can also run concurrently, with explicit per-job caps.
BC loss, optimizer activity, complete task success, and held-out performance
must remain separate reported outcomes. More resources cannot guarantee that
the policy learns a useful behavior within a short training budget.

The implemented RL runner collects serially. A vectorized actor/learner would
require additional implementation and validation; it is not obtained merely by
requesting more GPU memory or CPU cores. The short, explicitly demonstration-
seeded RL pilot can access late task states without waiting for one new online
episode to finish, but this does not establish policy improvement.

## Reduced powder model fallback

Removing individual grain *simulation* could yield a much larger gain than
hiding grain images. Such a fallback must be a separately named, validated
model and dataset. It cannot inherit the coarse-grain fixture's accuracy claims.

A defensible reduced model would retain the actual robot, tool, vials, grasps,
and collision checks. Source powder would be finite occupied volume. Actual
tool–bed overlap and swept motion would determine loading, limited by source
inventory and scoop capacity. Mass would be bulk density times volume; the
current solid-grain density would require a packing fraction before use as bulk
density. Material-dependent contact reaction, retention, and release laws would
remain explicit approximations. Flight and aperture capture would determine
whether released material enters the receiver or spills.

Mass must be conserved across source, tool, flight, receiver, and spill, with
payload weight and inertia coupled consistently to the physical bodies. The
balance must read the pan sensor. Agreement between an imposed payload weight
and that sensor verifies load coupling, not the accuracy of the approximate
scoop/release law. Basic validation would include zero pickup without tool–bed
interaction, no unlimited pickup from a stationary tool, density scaling,
finite capacity, aperture misses, conservation, and independent pan-force tests.

The original scalar transfer environment does not meet these requirements and
will not be re-enabled as a shortcut. A narrower dosing-only experiment using
the current grain physics is another possible scope reduction; it must be
labeled as a subtask rather than full weighing.
