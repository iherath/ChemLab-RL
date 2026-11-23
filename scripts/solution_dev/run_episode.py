"""One expert episode with its skill log, looks and final state.
   python scripts/solution_dev/run_episode.py 3mL 2 [solid_id] [solid_g]"""
import io, contextlib, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
with contextlib.redirect_stderr(io.StringIO()):
    from chemlab_rl.solution_env import SolutionPrepEnv
    from chemlab_rl.solution_skills import run_expert
    env = SolutionPrepEnv(images=False)
syr = sys.argv[1] if len(sys.argv) > 1 else "3mL"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sid = int(sys.argv[3]) if len(sys.argv) > 3 else seed % 5
sg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
t = time.time()
r = run_expert(env, seed=seed, syringe=syr, solid_id=sid, solid_g=sg, record=False)
print(f"{syr} seed {seed} solid {sid}: {r['outcome']}  steps {r['steps']}  wall {time.time() - t:.0f}s")
for e in r["log"]:
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
for l in r["looks"]:
    print("   look", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})
i = r["info"]
print("   dones", i["subgoal_dones"], "vial", round(i["vial_mL"], 4), "undissolved g", round(i["liquid"]["undissolved_g"], 5),
      "pen mm", round(i["max_contact_penetration_m"] * 1000, 3), "grain proj", i["grain_projections"])
