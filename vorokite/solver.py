#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vorokite.solver
A lightweight Voronoi utility for 2D center sets (unit-square domain).
- Periodic BC tiling (for robust tessellation near boundaries)
- Finite polygon reconstruction from SciPy Voronoi (cropped to [0,1]^2)
- Geometry & neighbor queries
- Stochasticity metrics
- CSV/NPY export (Abaqus I/O removed in the public version)

Dependencies: numpy, scipy, shapely
Optional: matplotlib (only if save_fig=True in voronoi_recon)
"""
from __future__ import annotations

import os
from math import isclose
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString

from .utils import get_nested_list_length


class Solver:
    """
    Voronoi helper for 2D center sets on the unit square [0,1]×[0,1].

    Workflow:
        s = Solver(centers, bc="Periodic")
        s.voronoi_recon(save_fig=False)
        s.compute_neighbor()
        s.compute_stochasticity(type=0)
        s.export_geometry(dir_path="./out", file_type="csv")

    Attributes set by voronoi_recon():
        unique_vertices : (N,2) float array
        full_ridge_vertices_unique : list[list[int]]
        upper_bound, lower_bound : index arrays of boundary nodes (y==1.0, y==0.0)
        polygon_list : list[list[tuple]]
        polygon_area : list[float]
        polygon_area_inside : list[float]    # cells not touching boundary
        polygon_sidelength : list[float]
        center_all, center_inside : list[int]
        cell_all, cell_inside : list[Polygon]
    """

    def __init__(self, centers: Sequence[Tuple[float, float]], bc: str = "Periodic"):
        # store original centers as list of tuples
        self.centers_ori: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in centers]
        self.centers: List[Tuple[float, float]] = (
            self.periodic_bc(self.centers_ori) if bc == "Periodic" else list(self.centers_ori)
        )

        # Voronoi diagram over possibly tiled centers
        self.vor = Voronoi(self.centers)

        # geometry / graph (populated by voronoi_recon)
        self.unique_vertices: Optional[np.ndarray] = None
        self.full_ridge_vertices_unique: Optional[List[List[int]]] = None
        self.upper_bound: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None

        # cell bookkeeping
        self.polygon_list: List[List[Tuple[float, float]]] = []
        self.polygon_area: List[float] = []
        self.polygon_area_inside: List[float] = []
        self.polygon_sidelength: List[float] = []

        self.center_all: List[int] = []
        self.center_inside: List[int] = []
        self.cell_all: List[Polygon] = []
        self.cell_inside: List[Polygon] = []

        # neighbor graph
        self.center_neighbor: List[List[int]] = []
        self.cell_neighbor: List[List[Polygon]] = []
        self.cell_distance: List[List[float]] = []

    # --------------------
    # Basic utilities
    # --------------------
    @staticmethod
    def periodic_bc(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Tile points with offsets in {-1,0,1}^2 and drop duplicates.
        Returns a list of unique (x,y) tuples (order-preserving).
        """
        offsets = (-1.0, 0.0, 1.0)
        tiled = [(px + dx, py + dy) for (px, py) in points for dx in offsets for dy in offsets]
        # de-duplicate while preserving order
        return list(dict.fromkeys(tiled))

    @staticmethod
    def extract_edges(polygon: Polygon) -> List[LineString]:
        if not polygon.is_valid:
            raise ValueError("Invalid polygon")
        ext = list(polygon.exterior.coords)
        return [LineString([ext[i], ext[i + 1]]) for i in range(len(ext) - 1)]

    @staticmethod
    def check_shared_edges(poly1: Polygon, poly2: Polygon) -> bool:
        """
        True if polygons share at least one boundary segment (edge).
        """
        edges1 = Solver.extract_edges(poly1)
        edges2 = Solver.extract_edges(poly2)
        for e1 in edges1:
            for e2 in edges2:
                # Shapely's equals compares geometries (orientation-agnostic)
                if e1.equals(e2):
                    return True
        return False

    # --------------------
    # Voronoi to finite polygons
    # --------------------
    def voronoi_finite_polygons_2d(self, radius: Optional[float] = None):
        """
        Reconstruct infinite regions of a 2D Voronoi diagram into finite polygons.

        Returns
        -------
        regions : list[list[int]]
            Indices of vertices for each (reconstructed) region.
        vertices : np.ndarray, shape (M,2)
            Coordinates of the (possibly augmented) vertex set.
        ridges_map : list[tuple]  (internal; used by recon)
        """
        if self.vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions: List[List[int]] = []
        new_vertices = self.vor.vertices.tolist()

        center = self.vor.points.mean(axis=0)
        if radius is None:
            radius = float(self.vor.points.ptp().max() * 2.0)

        # map of ridges incident to each input point index
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        ridges_map = []  # keep for debugging/inspection if needed

        for p1, region_idx in enumerate(self.vor.point_region):
            region = self.vor.regions[region_idx]
            if not region or all(v >= 0 for v in region):
                # finite region
                if region:
                    new_regions.append(list(region))
                continue

            ridges = all_ridges.get(p1, [])  # FIX: avoid undefined local variable
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue  # already finite on this side

                # Missing endpoint of an infinite ridge
                t = self.vor.points[p2] - self.vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = self.vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
                ridges_map.append((p1, p2, v1, v2))

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices), ridges_map

    def voronoi_recon(self, save_fig: bool = False, fig_name: str = "voro.png"):
        """
        Build finite, unit-square-clipped polygons; record vertices/edges and
        boundary node sets; optionally save a figure.
        """
        regions, vertices, _ = self.voronoi_finite_polygons_2d()
        # clip to unit square
        min_x = min_y = 0.0
        max_x = max_y = 1.0
        box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

        final_vertices: List[np.ndarray] = []
        ridge_vertices: List[List[int]] = []

        # reset containers
        self.polygon_list.clear()
        self.polygon_area.clear()
        self.polygon_area_inside.clear()
        self.polygon_sidelength.clear()
        self.center_all.clear()
        self.center_inside.clear()
        self.cell_all.clear()
        self.cell_inside.clear()

        # iterate regions
        for region in regions:
            poly_coords = vertices[region]
            poly_orig = Polygon(poly_coords)
            poly = poly_orig.intersection(box)

            if isinstance(poly, Point) or poly.area == 0.0:
                continue

            self.polygon_area.append(float(poly.area))
            self.polygon_sidelength.append(float(poly.exterior.length))

            polygon = [tuple(p) for p in poly.exterior.coords]

            # boundary contact check
            connected_to_boundary = any(
                isclose(p[0], 0.0, abs_tol=1e-5)
                or isclose(p[0], 1.0, abs_tol=1e-5)
                or isclose(p[1], 0.0, abs_tol=1e-5)
                or isclose(p[1], 1.0, abs_tol=1e-5)
                for p in polygon
            )

            # record center-to-cell maps
            for center_index, (x, y) in enumerate(self.centers):
                pt = Point(x, y)
                if poly_orig.contains(pt):
                    if center_index not in self.center_all:
                        self.center_all.append(center_index)
                        self.cell_all.append(poly)
                    if not connected_to_boundary:
                        self.center_inside.append(center_index)
                        self.cell_inside.append(poly)
                        self.polygon_area_inside.append(float(poly.area))

            # accumulate vertex indices for edges
            poly_ridge_vertices: List[int] = []
            current_length = get_nested_list_length(ridge_vertices) if ridge_vertices else 0
            for i, p in enumerate(polygon):
                final_vertices.append(np.array(p, dtype=float))
                poly_ridge_vertices.append(i + current_length)
            ridge_vertices.append(poly_ridge_vertices)
            self.polygon_list.append(polygon)

        # Optional plotting (lazy import)
        if save_fig:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap("rainbow")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(*zip(*self.centers), "ko", ms=2)
            for polygon in self.polygon_list:
                ax.fill(*zip(*polygon), alpha=1.0, color=cmap(np.random.rand()))
            ax.axis("off")
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.margins(0, 0)
            fig.savefig(fig_name, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # Build edges, discarding vertical edges lying strictly on x=0 or x=1
        tor = 1e-5
        full_ridge_vertices: List[List[int]] = []
        for element in ridge_vertices:
            for i in range(len(element) - 1):
                a = element[i]
                b = element[i + 1]
                x1 = final_vertices[a][0]
                x2 = final_vertices[b][0]
                if (isclose(x1, 0.0, abs_tol=tor) and isclose(x2, 0.0, abs_tol=tor)) or (
                    isclose(x1, 1.0, abs_tol=tor) and isclose(x2, 1.0, abs_tol=tor)
                ):
                    continue
                full_ridge_vertices.append([a, b])

        # de-duplicate vertices (rounded for stability) and remap edges
        unique_vertices, inverse_indices = np.unique(
            np.array(final_vertices).round(decimals=4), axis=0, return_inverse=True
        )
        remapped_edges: List[List[int]] = []
        for a, b in full_ridge_vertices:
            ia, ib = int(inverse_indices[a]), int(inverse_indices[b])
            if ia != ib:
                remapped_edges.append(sorted([ia, ib]))
        remapped_edges = np.unique(np.array(remapped_edges), axis=0).tolist()

        self.unique_vertices = unique_vertices
        self.full_ridge_vertices_unique = remapped_edges
        self.lower_bound = np.where(self.unique_vertices[:, 1] == 0.0)[0]
        self.upper_bound = np.where(self.unique_vertices[:, 1] == 1.0)[0]

        return self.unique_vertices, self.full_ridge_vertices_unique, self.upper_bound, self.lower_bound

    # --------------------
    # Neighbors & metrics
    # --------------------
    def compute_neighbor(self) -> None:
        if self.unique_vertices is None:
            self.voronoi_recon(save_fig=False)

        self.center_neighbor = []
        self.cell_neighbor = []
        self.cell_distance = []

        for idx, center in enumerate(self.center_inside):
            current_neighbors: List[int] = []
            neighbor_cells: List[Polygon] = []
            neighbor_dists: List[float] = []

            poly_current = self.cell_inside[idx]
            point1 = self.centers[center]

            for jdx, other_center in enumerate(self.center_all):
                if center == other_center:
                    continue
                poly_neigh = self.cell_all[jdx]
                if self.check_shared_edges(poly_current, poly_neigh):
                    point2 = self.centers[other_center]
                    neighbor_dists.append(float(np.linalg.norm(np.array(point1) - np.array(point2))))
                    neighbor_cells.append(poly_neigh)
                    current_neighbors.append(other_center)

            # Ensure uniqueness/stability
            self.center_neighbor.append(sorted(set(current_neighbors)))
            self.cell_neighbor.append(neighbor_cells)
            self.cell_distance.append(neighbor_dists)

    def compute_stochasticity(self, type: int = 0) -> float:
        """
        type=0: variance of the number of neighbors per (internal) cell
        type=1: variance of inter-nuclei distances among neighbors
        type=2: minimum inter-nuclei distance among neighbors
        type=3: std(area) * N over all cells
        type=4: std(area) * N over internal cells only
        """
        if not self.center_neighbor:
            self.compute_neighbor()

        if type == 0:
            n_neigh = [len(neighbors) for neighbors in self.cell_neighbor]
            return float(np.var(n_neigh))
        elif type == 1:
            flat = [d for distances in self.cell_distance for d in distances]
            return float(np.var(flat)) if flat else 0.0
        elif type == 2:
            flat = [d for distances in self.cell_distance for d in distances]
            return float(np.min(flat)) if flat else float("inf")
        elif type == 3:
            return float(np.std(self.polygon_area) * len(self.polygon_area)) if self.polygon_area else 0.0
        elif type == 4:
            return (
                float(np.std(self.polygon_area_inside) * len(self.polygon_area_inside))
                if self.polygon_area_inside
                else 0.0
            )
        else:
            raise ValueError("Unknown stochasticity type. Use 0,1,2,3, or 4.")

    # --------------------
    # Export (public version)
    # --------------------
    def export_geometry(self, dir_path: str = "./out", file_type: str = "csv") -> None:
        """
        Export the current geometry (nodes + edges) to CSV or NPY.
        - CSV:   nodes.csv (float64), edges.csv (int)
        - NPY:   nodes.npy, edges.npy
        """
        if self.unique_vertices is None or self.full_ridge_vertices_unique is None:
            self.voronoi_recon(save_fig=False)

        os.makedirs(dir_path, exist_ok=True)
        if file_type.lower() == "csv":
            np.savetxt(os.path.join(dir_path, "nodes.csv"), self.unique_vertices, delimiter=",")
            np.savetxt(
                os.path.join(dir_path, "edges.csv"),
                np.asarray(self.full_ridge_vertices_unique, dtype=int),
                delimiter=",",
                fmt="%d",
            )
        elif file_type.lower() == "npy":
            np.save(os.path.join(dir_path, "nodes.npy"), self.unique_vertices)
            np.save(os.path.join(dir_path, "edges.npy"), np.asarray(self.full_ridge_vertices_unique, dtype=int))
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'npy'.")
