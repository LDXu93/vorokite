# vorokite.utils — geometry helpers (public version)
import numpy as np
import math

def get_nested_list_length(nested_list):
    """
    Calculates the total length of a nested list by summing the lengths of its elements.
    """
    total_length = 0
    for element in nested_list:
        total_length += len(element)
    return total_length

def chunk_list(lst, chunk_size):
    """
    Divides a list into chunks of a specified size.
    """
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def points_indomain(centers, bound):
    """
    Keep only points within the rectangular domain [lower_bound, upper_bound].
    """
    lower_bound, upper_bound = bound
    rows_to_keep = np.logical_and(
        np.all(centers >= lower_bound, axis=1),
        np.all(centers <= upper_bound, axis=1),
    )
    return centers[rows_to_keep]

def calculate_rectangles_centers(domain_size, rectangle_size):
    domain_width, domain_height = domain_size
    rectangle_width, rectangle_height = rectangle_size
    if rectangle_width > domain_width or rectangle_height > domain_height:
        raise ValueError("Rectangle dimensions cannot exceed domain dimensions.")
    centers = []
    for y in np.arange(0, domain_height, rectangle_height):
        for x in np.arange(0, domain_width, rectangle_width):
            centers.append((x + rectangle_width / 2, y + rectangle_height / 2))
    return centers

def calculate_rectangles90_centers(domain_size, side_length):
    domain_width, domain_height = domain_size
    centers = []
    num_rows = int(domain_height / (side_length * 0.5)) + 1
    num_cols = int(domain_width / side_length) + 1
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * side_length + (i % 2) * (side_length / 2)
            y = i * (side_length * 0.5)
            centers.append((x, y))
    return centers

def calculate_hexagon_centers(radius):
    centers = []
    side_length = math.sqrt(3) * radius
    num_rows = int(1 / (radius * 1.5)) + 1
    num_cols = int(1 / side_length) + 1
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * side_length + (i % 2) * (side_length / 2)
            y = i * (radius * 1.5)
            centers.append((x, y))
    return centers

def calculate_triangle_centers(side_length):
    height = math.sqrt(3) * side_length / 6
    num_triangles_x = int(1 / side_length)
    num_triangles_y = int(1 / (height * 0.5))
    centers = []
    for row in range(num_triangles_y):
        y = row * height * 0.5
        for col in range(num_triangles_x):
            x = col * side_length + (0.5 * side_length * (row % 2))
            centers.append((x, y))
    return centers
