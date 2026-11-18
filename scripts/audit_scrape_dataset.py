"""Print a compact source-data audit without loading a policy or image movies."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemlab_rl.vla_bc import inspect_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--heldout-powders", nargs="+", type=int, default=[5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = inspect_dataset(args.dataset, tuple(args.heldout_powders))
    text = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)

if __name__ == "__main__":
    main()
