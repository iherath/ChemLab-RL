import sys, time, json, numpy as np
from chemlab_rl.solution_env import SolutionPrepEnv
from chemlab_rl.solution_skills import run_expert
syr = sys.argv[1] if len(sys.argv) > 1 else "3mL"; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
env = SolutionPrepEnv(images=False)
t = time.time()
r = run_expert(env, seed=seed, syringe=syr, solid_id=0, solid_g=0.12, record=False)
print(f"{syr} seed {seed}: {r['outcome']}  steps {r['steps']}  wall {time.time()-t:.0f}s")
for e in r["log"]: print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
i = r["info"]; print("   dones", i["subgoal_dones"], "vial", round(i["vial_mL"], 4), "spill", round(i["spilled_mL"], 4), "cons", i["conservation_error_mL"], "pen mm", round(i["max_contact_penetration_m"]*1000, 3))
