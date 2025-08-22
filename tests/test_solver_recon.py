import numpy as np
from vorokite import calculate_rectangles_centers, points_indomain, Solver

def _make_centers():
    centers = calculate_rectangles_centers((1.0, 1.0), (0.25, 0.25))
    centers = points_indomain(np.array(centers), (np.array([0.,0.]), np.array([1.,1.])))
    return centers.tolist()

def test_voronoi_recon_outputs_shapes_and_bounds():
    s = Solver(_make_centers(), bc="Periodic")
    nodes, edges, top, bot = s.voronoi_recon(save_fig=False)

    assert nodes.shape[1] == 2 and nodes.shape[0] > 0
    assert np.all(nodes >= -1e-9) and np.all(nodes <= 1 + 1e-9)

    assert len(edges) > 0
    n = nodes.shape[0]
    assert all(0 <= a < n and 0 <= b < n and a != b for a,b in edges)

    assert top[0].size > 0 and bot[0].size > 0

def test_neighbors_and_metrics_are_finite():
    s = Solver(_make_centers(), bc="Periodic")
    s.voronoi_recon(save_fig=False)
    s.compute_neighbor()

    assert len(s.center_inside) >= 1
    assert any(len(nbs) > 0 for nbs in s.center_neighbor)

    m0 = s.compute_stochasticity(type=0)
    m1 = s.compute_stochasticity(type=1)
    m2 = s.compute_stochasticity(type=2)
    m3 = s.compute_stochasticity(type=3)
    m4 = s.compute_stochasticity(type=4)

    for m in (m0, m1, m3, m4):
        assert np.isfinite(m)
    assert m2 > 0.0 and np.isfinite(m2)
