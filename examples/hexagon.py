import argparse
import numpy as np
from pathlib import Path

from vorokite import calculate_hexagon_centers, points_indomain, Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="save a PNG figure")
    parser.add_argument("--out", default="out", help="output folder")
    args = parser.parse_args()

    centers = calculate_hexagon_centers(radius=0.18)
    centers = points_indomain(np.array(centers), (np.array([0.,0.]), np.array([1.,1.])))

    s = Solver(centers.tolist(), bc="Periodic")
    s.voronoi_recon(save_fig=args.plot, fig_name=str(Path(args.out) / "voro_hex.png"))
    s.write_geometry(dir_path=args.out if hasattr(s, "export_geometry") is False else args.out, file_type="csv")  # compat
    # if you adopted write_geometry name in solver:
    # s.write_geometry(dir_path=args.out, file_type="csv")

    print("neighbors variance:", s.compute_stochasticity(type=0))

if __name__ == "__main__":
    main()
