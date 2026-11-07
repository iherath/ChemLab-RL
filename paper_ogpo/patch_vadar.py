"""Copy chem_vial_utils.py into vadar-public and register the env. Idempotent."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARK = "chem_vial (chemlab-rl-project)"
HERE = Path(__file__).resolve().parent
SRC = HERE / "chem_vial_utils.py"


def _wrapped(body: str) -> str:
    return f"    # --- {MARK} ---\n{body.rstrip()}\n    # --- end {MARK} ---\n"


def _insert_once(text: str, needle: str, body: str, after: bool = True) -> str:
    if body.strip() in text:
        return text
    if needle not in text:
        raise SystemExit(f"patch needle not found:\n{needle!r}")
    block = _wrapped(body)
    if after:
        return text.replace(needle, needle + "\n" + block, 1)
    return text.replace(needle, block + "\n" + needle, 1)


def patch_subgoals(root: Path) -> None:
    path = root / "envs" / "subgoals.py"
    text = path.read_text()
    spec = """    'chem_vial': dict(
        names=['open_door', 'grasp_vial', 'place_vial', 'close_shield'],
        dag=None,
        discounts=[0.9, 0.99, 0.995, 0.999],
    ),"""
    text = _insert_once(
        text,
        "    'pusht_obstacle': dict(names=['task_done', 'obstacle_cost'], dag=None, discounts=None,\n"
        "                           head_types=['subgoal', 'cost']),\n",
        spec,
    )
    key = """    if env_name.startswith('chem_vial'):
        return 'chem_vial'"""
    text = _insert_once(
        text,
        "    if env_name.startswith('pusht') and 'obstacle' in env_name:\n"
        "        return 'pusht_obstacle'\n"
        "    return None\n",
        key,
        after=False,
    )
    path.write_text(text)
    print(f"patched {path}")


def patch_env_utils(root: Path) -> None:
    path = root / "envs" / "env_utils.py"
    text = path.read_text()
    body = """    elif env_name.startswith("chem_vial"):
        from envs import chem_vial_utils
        env = chem_vial_utils.make_env(env_name, seed=0, subgoal_cfg=subgoal_cfg)
        eval_env = chem_vial_utils.make_env(env_name, seed=42, subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = chem_vial_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None"""
    text = _insert_once(
        text,
        "    else:\n        raise ValueError(f'Unsupported environment: {env_name}')\n",
        body,
        after=False,
    )
    path.write_text(text)
    print(f"patched {path}")


def patch_main_helper(root: Path) -> None:
    path = root / "vadar" / "utils" / "main_helper.py"
    text = path.read_text()
    maker = """    if env_name.startswith('chem_vial'):
        from envs.chem_vial_utils import make_env
        return make_env"""
    text = _insert_once(
        text,
        "    # RoboCasa / LIBERO first: their venvs may not have robomimic installed.\n"
        "    if env_name.startswith('robocasa_'):\n",
        maker,
        after=False,
    )
    vec = """        elif env_name.startswith('chem_vial'):
            env = make_env(env_name, seed=env_seed, subgoal_cfg=subgoal_cfg)"""
    text = _insert_once(
        text,
        "        elif subgoal_cfg is not None:\n"
        "            env = make_env(env_name, seed=env_seed, post_success_steps=post_success_steps, subgoal_cfg=subgoal_cfg)\n",
        vec,
        after=False,
    )
    bounds = """    elif env_name.startswith('chem_vial'):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)"""
    text = _insert_once(
        text,
        "    else:\n"
        '        raise ValueError(f"Unknown environment: {env_name}, please specify min max reward")\n',
        bounds,
        after=False,
    )
    path.write_text(text)
    print(f"patched {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("vadar_root", type=Path)
    args = p.parse_args()
    root = args.vadar_root.resolve()
    if not (root / "vadar" / "main.py").is_file():
        raise SystemExit(f"{root} is not a vadar-public checkout")
    dest = root / "envs" / "chem_vial_utils.py"
    shutil.copy2(SRC, dest)
    print(f"copied {SRC.name} -> {dest}")
    patch_subgoals(root)
    patch_env_utils(root)
    patch_main_helper(root)
    print("chem_vial registered. Set CHEMVIAL_DATASET, CHEM_RL_SIM, CHEMLAB_ROOT, PYTHONPATH.")


if __name__ == "__main__":
    main()
