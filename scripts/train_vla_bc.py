"""CLI for pretrained SmolVLA BC on the scrape HDF5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemlab_rl.vla_bc import train_main
if __name__ == "__main__":
    train_main()
