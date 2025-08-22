import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def get_nested_list_length(nested_list: Sequence[Sequence]) -> int:
    """
    Sum of lengths of all sublists.

    Examples
    --------
    >>> get_nested_list_length([[1,2,3],[4,5],[6,7,8,9]])
    9
    """
    return sum(len(el) for el in nested_list)


def chunk_list(lst: Sequence, chunk_size: int) -> List[List]:
    """
    Split a sequence into chunks of size `chunk_size`.
    """
    return [list(lst[i : i + chunk_size]) for i in range(0, len(lst), chunk_size)]


def points_indomain(centers: np.ndarray, bound: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Keep points within a rectangular domain.

    Parameters
    ----------
    centers : (N,2) array-like
    bound   : (lower, upper) where each is a length-2 array-like

    Returns
    -------
    (M,2) ndarray of points within [lower, upper].
    """
    lower, upper = bound
    centers = np.asarray(centers, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    keep = np.logical_and(np.all(centers >= lower, axis=1), np.all(centers <= upper, axis=1))
    return centers[keep]


def calculate_rectangles_centers(domain_size: Tuple[float, float], rectangle_size: Tuple[float, float]):
    """
    Regular grid of rectangle centers covering the domain.
    """
    domain_width, domain_height = domain_size
    rectangle_width, rectangle_height = rectangle_size

    if rectangle_width > domain_width or rectangle_height > domain_height:
        raise ValueError("Rectangle dimensions cannot exceed domain dimensions.")

    centers = []
    for y in np.arange(0.0, domain_height, rectangle_height):
        for x in np.arange(0.0, domain_width, rectangle_width):
            centers.append((x + rectangle_width / 2.0, y + rectangle_height / 2.0))
    return centers


def calculate_rectangles90_centers(domain_size: Tuple[float, float], side_length: float):
    """
    Staggered grid (every other row offset by half side_length).
    """
    domain_width, domain_height = domain_size
    centers = []

    num_rows = int(domain_height / (side_length * 0.5)) + 1
    num_cols = int(domain_width / side_length) + 1

    for i in range(num_rows):
        for j in range(num_cols):
            x = j * side_length + (i % 2) * (side_length / 2.0)
            y = i * (side_length * 0.5)
            centers.append((x, y))
    return centers


def calculate_hexagon_centers(radius: float):
    """
    Hexagonal packing of centers in the unit square, given circumscribed radius.
    """
    centers = []
    side = math.sqrt(3.0) * radius
    num_rows = int(1.0 / (radius * 1.5)) + 1
    num_cols = int(1.0 / side) + 1

    for i in range(num_rows):
        for j in range(num_cols):
            x = j * side + (i % 2) * (side / 2.0)
            y = i * (radius * 1.5)
            centers.append((x, y))
    return centers


def calculate_triangle_centers(side_length: float):
    """
    Centers for an equilateral-triangle tiling over the unit square.
    """
    height = math.sqrt(3.0) * side_length / 6.0
    num_x = int(1.0 / side_length)
    num_y = int(1.0 / (height * 0.5))

    centers = []
    for row in range(num_y):
        y = row * height * 0.5
        for col in range(num_x):
            x = col * side_length + (0.5 * side_length * (row % 2))
            centers.append((x, y))
    return centers
