import numpy as np
from pathlib import Path
from vorokite import calculate_rectangles_centers, Solver

def test_export_csv_and_npy(tmp_path: Path):
    centers = calculate_rectangles_centers((1.0, 1.0), (0.25, 0.25))
    s = Solver(centers, bc="Periodic")
    s.voronoi_recon(save_fig=False)

    # CSV
    out_csv = tmp_path / "csv_out"
    s.write_geometry(fold_path=str(out_csv), file_type="csv")
    nodes = np.loadtxt(out_csv / "nodes.csv", delimiter=",")
    edges = np.loadtxt(out_csv / "edges.csv", delimiter=",", dtype=int)
    assert nodes.ndim == 2 and nodes.shape[1] == 2
    assert edges.ndim == 2 and edges.shape[1] == 2

    # NPY
    out_npy = tmp_path / "npy_out"
    s.write_geometry(fold_path=str(out_npy), file_type="npy")
    nodes2 = np.load(out_npy / "nodes.npy")
    edges2 = np.load(out_npy / "edges.npy")
    assert nodes2.shape == nodes.shape
    assert edges2.shape == edges.shape
