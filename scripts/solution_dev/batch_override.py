"""Parallel expert batch with PARAM_RANGES overrides; world-geom blocks are named (W:<geom>).
   python scripts/solution_dev/batch_override.py "raise_h=0.18,0.20;handover_dx=0.02,0.045" 1mL:0 3mL:4 ..."""
import sys, os, time, io, contextlib, warnings, multiprocessing as mp
OVR = {k: tuple(map(float, v.split(","))) for k, v in (kv.split("=") for kv in sys.argv[1].split(";") if kv)}
def job(a):
    syr, seed = a
    warnings.filterwarnings("ignore")
    with contextlib.redirect_stderr(io.StringIO()):
        from chemlab_rl import solution_skills as S
        S.PARAM_RANGES.update(OVR)
        import chem_rl_sim.teleop.keyframe as K
        _lab = K.Session._label
        def _label(self, g, carried=()):
            out = _lab(self, g, carried)
            if out == "table":
                import mujoco
                out = "W:" + (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")
            return out
        K.Session._label = _label
        from chemlab_rl.solution_env import SolutionPrepEnv
        env = SolutionPrepEnv(images=False); t = time.time()
        r = S.run_expert(env, seed=seed, syringe=syr, solid_id=seed % 5, solid_g=0.05 + 0.05 * (seed % 5), record=False)
    i = r["info"]
    return f"{syr} seed {seed:2d}: {r['outcome'][:170]:170s} steps {r['steps']:5d} vial {i['vial_mL']:.4f} pen {i['max_contact_penetration_m']*1000:.2f}mm wall {time.time()-t:.0f}s"
if __name__ == "__main__":
    jobs = [(s, int(k)) for s, k in (a.split(":") for a in sys.argv[2:])]
    with mp.Pool(min(8, os.cpu_count())) as pool:
        out = []
        for l in pool.imap(job, jobs):
            print(l, flush=True); out.append(l)
    print(f"success {sum(': success' in l for l in out)}/{len(out)}")
