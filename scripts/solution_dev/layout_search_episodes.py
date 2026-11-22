"""Choose the prep-area location by running real expert episodes per candidate layout."""
import sys, io, contextlib, warnings, itertools, json, multiprocessing as mp, re
warnings.filterwarnings("ignore")
BASE = {"vial_a": (-0.14, 0.02), "reservoir": (-0.27, 0.02), "syringe_rack": (-0.205, 0.08), "cap": (-0.30, -0.10)}
WIDE = {"vial_a": (0.0, 0.0), "reservoir": (-0.02, 0.0), "syringe_rack": (-0.01, 0.03), "cap": (-0.02, -0.02)}

def layout(sx, sy, wide):
    return {k: (x + sx + (WIDE[k][0] if wide else 0), y + sy + (WIDE[k][1] if wide else 0)) for k, (x, y) in BASE.items()}

def job(args):
    lid, lay, seed, syr = args
    with contextlib.redirect_stderr(io.StringIO()):
        import chemlab_rl.solution_scene as S
        S.LAYOUT.update(lay)
        from chemlab_rl.solution_env import SolutionPrepEnv
        from chemlab_rl.solution_skills import run_expert
        env = SolutionPrepEnv(images=False)
        r = run_expert(env, seed=seed, syringe=syr, solid_id=seed % 5, solid_g=0.1, record=False)
    out = r["outcome"]
    cls = "success" if out == "success" else re.sub(r"#\d+|[\d.]+ mm|\(.*", "", out)[:80]
    return lid, cls

if __name__ == "__main__":
    lays = {}
    for sx, sy, wide in itertools.product((0.0, -0.06, -0.12), (0.0, -0.06, -0.12), (False, True)):
        lays[f"sx{sx:+.2f}_sy{sy:+.2f}_{'wide' if wide else 'base'}"] = layout(sx, sy, wide)
    jobs = [(lid, lay, seed, syr) for lid, lay in lays.items() for seed, syr in ((100, "1mL"), (101, "2mL"), (102, "3mL"), (103, "5mL"))]
    res = {}
    with mp.Pool(8) as pool:
        for lid, cls in pool.imap_unordered(job, jobs):
            res.setdefault(lid, []).append(cls)
    rows = sorted(res.items(), key=lambda kv: -sum(c == "success" for c in kv[1]))
    for lid, cls in rows:
        print(f"{sum(c == 'success' for c in cls)}/4  {lid:24s}", "; ".join(sorted(set(c for c in cls if c != "success")))[:160])
    json.dump({k: lays[k] for k, _ in rows[:3]}, open(sys.argv[1], "w"), indent=1)
