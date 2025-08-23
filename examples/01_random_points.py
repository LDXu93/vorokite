#!/usr/bin/env python3
"""
01_random_points.py
Quick start: random centers -> Voronoi -> metrics -> export.

Usage:
  python examples/01_random_points.py --n 150 --seed 0 --out out --plot

Requires:
  pip install vorokite
  (optional for --plot) pip install "vorokite[plots]"
"""
import argparse
from pathlib import Path

import numpy as np
from vorokite import Solver, points_indomain

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150, help="number of random centers")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--out", type=str, default="out", help="output directory")
    p.add_argument("--plot", action="store_true", help="save a PNG figure")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    centers = rng.random((args.n, 2))  # in [0,1]^2 already
    # Deduplicate (rare, but protects SciPy/Qhull)
    centers = np.unique(centers, axis=0)
    centers = points_indomain(centers, (np.array([0., 0.]), np.array([1., 1.])))

    s = Solver(centers.tolist(), bc="Periodic")
    s.voronoi_recon(save_fig=args.plot, fig_name=str(Path(args.out) / "voro_random.png"))

    # metrics
    print("[vorokite] metrics on random centers:")
    print("  var(#neighbors)           :", s.compute_stochasticity(type=0))
    print("  var(inter-nuclei distance):", s.compute_stochasticity(type=1))
    print("  min inter-nuclei distance :", s.compute_stochasticity(type=2))
    print("  std(area)*N (all cells)   :", s.compute_stochasticity(type=3))
    print("  std(area)*N (internal)    :", s.compute_stochasticity(type=4))

    # export geometry
    s.write_geometry(fold_path=args.out, file_type="csv")
    # also NPY, if you want both:
    # s.write_geometry(fold_path=str(Path(args.out)/"npy"), file_type="npy")

if __name__ == "__main__":
    main()
