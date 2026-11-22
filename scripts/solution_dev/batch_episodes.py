import sys, os, time, warnings, multiprocessing as mp
def job(args):
    syr, seed = args
    import io, contextlib
    warnings.filterwarnings("ignore")
    with contextlib.redirect_stderr(io.StringIO()):
        from chemlab_rl.solution_env import SolutionPrepEnv
        from chemlab_rl.solution_skills import run_expert
        env = SolutionPrepEnv(images=False)
        t = time.time()
        r = run_expert(env, seed=seed, syringe=syr, solid_id=seed % 5, solid_g=0.05 + 0.05 * (seed % 5), record=False)
    i = r["info"]
    sw = next((e for e in r["log"] if e["skill"] == "swirl"), {})
    dips = [e.get("dips") for e in r["log"] if e["skill"] == "draw"]
    return f"{syr} seed {seed:2d} solid {seed%5}: {r['outcome'][:110]:110s} steps {r['steps']:5d} vial {i['vial_mL']:.4f} spill {i['spilled_mL']:.4f} pen {i['max_contact_penetration_m']*1000:.2f}mm dips {dips} swirl {sw.get('seconds')} wall {time.time()-t:.0f}s"
if __name__ == "__main__":
    jobs = [(s, int(k)) for s, k in (a.split(":") for a in sys.argv[1:])]
    with mp.Pool(min(8, os.cpu_count())) as pool:
        for line in pool.imap(job, jobs):
            print(line, flush=True)
