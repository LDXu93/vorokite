import argparse
import numpy as np
from pathlib import Path

from vorokite import calculate_rectangles_centers, points_indomain, Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    centers = calculate_rectangles_centers((1.0, 1.0), (0.25, 0.25))
    centers = points_indomain(np.array(centers), (np.array([0.,0.]), np.array([1.,1.])))

    s = Solver(centers, bc="Periodic")
    s.voronoi_recon(save_fig=args.plot, fig_name=str(Path(args.out) / "voro_rect.png"))
    s.write_geometry(dir_path=args.out, file_type="csv")

    print("min inter-nuclei dist:", s.compute_stochasticity(type=2))

if __name__ == "__main__":
    main()
