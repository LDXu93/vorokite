#!/usr/bin/env python3
"""
02_batch_metrics.py
Batch over different seeds -> compute metrics -> stream results to summary.csv.

Usage:
  python examples/02_batch_metrics.py --n 120 --seeds 10 --out out/batch
  # optional: append to existing CSV instead of overwriting:
  # python examples/02_batch_metrics.py --n 120 --seeds 10 --out out/batch --append
"""
import argparse
from pathlib import Path
import csv
import numpy as np
from vorokite import Solver

def run_once(n: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.random((n, 2))
    centers = np.unique(centers, axis=0)
    s = Solver(centers.tolist(), bc="Periodic")
    s.voronoi_recon(save_fig=False)
    return dict(
        var_neighbors=s.compute_stochasticity(type=0),
        var_dist=s.compute_stochasticity(type=1),
        min_dist=s.compute_stochasticity(type=2),
        std_area_all=s.compute_stochasticity(type=3),
        std_area_internal=s.compute_stochasticity(type=4),
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120, help="number of centers per run")
    p.add_argument("--seeds", type=int, default=8, help="how many seeds to run (0..seeds-1)")
    p.add_argument("--out", type=str, default="out/batch", help="output directory")
    p.add_argument("--append", action="store_true", help="append to an existing CSV (default: overwrite)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"

    fieldnames = ["seed", "n", "var_neighbors", "var_dist", "min_dist", "std_area_all", "std_area_internal"]
    mode = "a" if args.append and out_csv.exists() else "w"

    print(f"[vorokite] Writing results to: {out_csv} (mode={mode})")

    with out_csv.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()

        for seed in range(args.seeds):
            try:
                metrics = run_once(args.n, seed)
                row = dict(seed=seed, n=args.n, **metrics)
                w.writerow(row)
                f.flush()  # ensure row hits disk right away
                print(f"[seed {seed:02d}] {metrics}")
            except Exception as e:
                # keep going; you still get a partial CSV
                print(f"[seed {seed:02d}] ERROR: {e!r}")

    print(f"[vorokite] Done. CSV at: {out_csv}")

if __name__ == "__main__":
    main()
