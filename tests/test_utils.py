import numpy as np
from vorokite import (
    get_nested_list_length,
    chunk_list,
    points_indomain,
    calculate_rectangles_centers,
    calculate_hexagon_centers,
    calculate_triangle_centers,
)

def test_get_nested_list_length_and_chunk():
    lst = list(range(10))
    chunks = chunk_list(lst, 3)
    assert get_nested_list_length(chunks) == len(lst)
    assert chunks[0] == [0,1,2]
    assert chunks[-1] == [9]

def test_points_indomain_filters():
    pts = np.array([[0.1,0.2],[1.1,0.5],[-0.1,0.3],[0.9,1.0]])
    kept = points_indomain(pts, (np.array([0,0]), np.array([1,1])))
    assert kept.shape == (2,2)
    assert (kept == np.array([[0.1,0.2],[0.9,1.0]])).all()

def test_center_generators_nonempty():
    rect = calculate_rectangles_centers((1.0,1.0),(0.25,0.25))
    hexs = calculate_hexagon_centers(0.2)
    tris = calculate_triangle_centers(0.2)
    assert len(rect) > 0 and len(hexs) > 0 and len(tris) > 0
