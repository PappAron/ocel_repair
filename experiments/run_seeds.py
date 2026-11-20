"""Run ocel_repair's `compare` once per seed and keep one JSON + one log per seed (B5).

Run from inside ocel_repair, e.g.
  python experiments\\run_seeds.py data\\ocel2-p2p.sqlite --tag p2p --seeds 0 1 2 3 4 --epochs 2000
Extra arguments after `--` go to main.py unchanged, e.g. `-- --drop-fraction 0.5`.
Seeds whose JSON already exists are skipped, so an interrupted run can simply be restarted.
Then:  python experiments\\aggregate_seeds.py experiments\\results\\seeds_p2p
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("source_path")
parser.add_argument("--tag", required=True, help="Name of the output folder: results/seeds_<tag>")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--slim", action="store_true", help="Skip attribute loading (use for BPI 2017)")
argv = sys.argv[1:]
extra = argv[argv.index("--") + 1:] if "--" in argv else []
argv = argv[: argv.index("--")] if "--" in argv else argv
args = parser.parse_args(argv)

out_dir = HERE / "results" / f"seeds_{args.tag}"
out_dir.mkdir(parents=True, exist_ok=True)
entry = [str(HERE / "run_compare_slim.py")] if args.slim else ["main.py"]

for seed in args.seeds:
    json_path = out_dir / f"seed{seed}.json"
    if json_path.exists():
        print(f"seed {seed}: {json_path.name} exists, skipped")
        continue
    cmd = [sys.executable, *entry, "compare", args.source_path, "--loader", "sqlite",
           "--epochs", str(args.epochs), "--seed", str(seed),
           "--results-json", str(json_path), *extra]
    print(f"seed {seed}: running ...", flush=True)
    started = time.time()
    with open(out_dir / f"seed{seed}.log", "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"seed {seed}: exit {code}, {time.time() - started:.0f} s", flush=True)
    if code != 0:
        sys.exit(f"seed {seed} failed; see {out_dir / f'seed{seed}.log'}")
