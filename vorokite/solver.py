#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vorokite.solver (public version)

Utilities for 2D Voronoi reconstruction on the unit square [0,1] x [0,1].
- Periodic tiling of input centers (optional)
- Finite polygon reconstruction from SciPy Voronoi
- Shapely-based clipping to the unit square
- Neighbor graph & simple stochasticity metrics
- CSV/NPY export for nodes/edges (no Abaqus writers)

Dependencies: numpy, scipy, shapely
Optional: matplotlib (only when calling plotting methods)
"""
import os
from math import isclose
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString

from vorokite.utils import get_nested_list_length


class Solver:
    """
    Voronoi helper for 2D center sets on [0,1]×[0,1].

    Typical usage:
        s = Solver(centers, bc="Periodic")
        s.voronoi_recon(save_fig=False)
        s.compute_neighbor()
        v = s.compute_stochasticity(type=0)
        s.write_geometry("./out", file_type="csv")

    Attributes populated by `voronoi_recon()`:
        unique_vertices : (N,2) ndarray[float]
        full_ridge_vertices_unique : list[list[int]]
        upper_bound, lower_bound   : index tuples (np.where results) for y==1.0 and y==0.0
        polygon_list               : list of polygon coordinate lists
        polygon_area               : list[float]
        polygon_area_inside        : list[float] (cells not touching boundary)
        polygon_sidelength         : list[float]
        center_all, center_inside  : list[int]
        cell_all, cell_inside      : list[Polygon]
    """

    def __init__(self, centers: List[Tuple[float, float]], bc: str = "Periodic"):
        # Store original centers as tuples
        self.centers_ori: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in centers]
        self.centers: List[Tuple[float, float]] = (
            self.periodic_bc(self.centers_ori) if bc == "Periodic" else list(self.centers_ori)
        )

        # Build Voronoi diagram over possibly tiled centers
        self.vor = Voronoi(self.centers)

        # Geometry/graph placeholders (populated later)
        self.unique_vertices: Optional[np.ndarray] = None
        self.full_ridge_vertices_unique: Optional[List[List[int]]] = None
        self.upper_bound = None
        self.lower_bound = None

        # Cell bookkeeping
        self.polygon_list: List[List[Tuple[float, float]]] = []
        self.polygon_area: List[float] = []
        self.polygon_area_inside: List[float] = []
        self.polygon_sidelength: List[float] = []

        self.center_all: List[int] = []
        self.center_inside: List[int] = []
        self.cell_all: List[Polygon] = []
        self.cell_inside: List[Polygon] = []

        # Neighbor graph
        self.center_neighbor: List[List[int]] = []
        self.cell_neighbor: List[List[Polygon]] = []
        self.cell_distance: List[List[float]] = []

    # --------------------
    # Basic utilities
    # --------------------
    @staticmethod
    def periodic_bc(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Tile points with offsets in {-1,0,1}² and drop duplicates (order-preserving).
        """
        offsets = (-1.0, 0.0, 1.0)
        tiled = [(px + dx, py + dy) for (px, py) in points for dx in offsets for dy in offsets]
        return list(dict.fromkeys(tiled))

    @staticmethod
    def extract_edges(polygon: Polygon) -> List[LineString]:
        if not polygon.is_valid:
            raise ValueError("Invalid polygon")
        coords = list(polygon.exterior.coords)
        return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]

    @staticmethod
    def check_shared_edges(poly1: Polygon, poly2: Polygon) -> bool:
        """
        True if polygons share at least one boundary segment (edge).
        """
        e1 = Solver.extract_edges(poly1)
        e2 = Solver.extract_edges(poly2)
        for a in e1:
            for b in e2:
                if a.equals(b):
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
        vertices: np.ndarray of shape (M,2)
        ridges  : last ridges list (kept to match original behavior)
        """
        if self.vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = self.vor.vertices.tolist()

        center = self.vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(self.vor.points, axis=0).max() * 2.0 # ndarray.ptp() removed in numpy 2.....


        # Map of ridges incident to each input point index
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        ridges = []  # keep for return to preserve your original API

        # Reconstruct per point
        for p1, region in enumerate(self.vor.point_region):
            vertices = self.vor.regions[region]
            if not vertices:
                continue

            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            # keep your exact line (results unchanged). ridges stays [] if p1 not present.
            if p1 in all_ridges:
                ridges = all_ridges[p1]

            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                # Compute the missing endpoint of an infinite ridge
                t = self.vor.points[p2] - self.vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = self.vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region CCW
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)].tolist()

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices), ridges

    def voronoi_recon(self, save_fig: bool = False, fig_name: str = "voro.png"):
        """
        Build finite, unit-square-clipped polygons; record vertices/edges and
        boundary node sets; optionally save a figure.
        """
        regions, vertices, _ = self.voronoi_finite_polygons_2d()

        # unit box
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

        for region in regions:
            poly_coords = vertices[region]
            poly_orig = Polygon(poly_coords)
            poly = poly_orig.intersection(box)
            if isinstance(poly, Point) or poly.area == 0.0:
                continue

            self.polygon_area.append(float(poly.area))
            self.polygon_sidelength.append(float(poly.exterior.length))

            polygon = [tuple(p) for p in poly.exterior.coords]

            # Check boundary contact
            connected_to_boundary = any(
                isclose(p[0], 0.0, abs_tol=1e-5)
                or isclose(p[0], 1.0, abs_tol=1e-5)
                or isclose(p[1], 0.0, abs_tol=1e-5)
                or isclose(p[1], 1.0, abs_tol=1e-5)
                for p in polygon
            )

            # Map centers to cells (simple but O(N^2); OK for small problems)
            for center_index, (x, y) in enumerate(self.centers):
                if poly_orig.contains(Point(x, y)):
                    if center_index not in self.center_all:
                        self.center_all.append(center_index)
                        self.cell_all.append(poly)
                    if not connected_to_boundary:
                        self.center_inside.append(center_index)
                        self.cell_inside.append(poly)
                        self.polygon_area_inside.append(float(poly.area))

            # collect vertex indices per polygon (for edge building)
            current_len = get_nested_list_length(ridge_vertices) if ridge_vertices else 0
            poly_ridge_vertices: List[int] = []
            for i, p in enumerate(polygon):
                final_vertices.append(np.array(p, dtype=float))
                poly_ridge_vertices.append(i + current_len)
            ridge_vertices.append(poly_ridge_vertices)
            self.polygon_list.append(polygon)

        if save_fig:
            import matplotlib.pyplot as plt  # lazy import
            cmap = plt.cm.get_cmap("rainbow")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(*zip(*self.centers), "ko")
            for polygon in self.polygon_list:
                ax.fill(*zip(*polygon), alpha=1.0, color=cmap(np.random.rand()))
            ax.axis("off")
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.margins(0, 0)
            fig.savefig(fig_name, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # Remove vertical boundary edges (x=0 or x=1)
        tor = 1e-5
        full_ridge_vertices: List[List[int]] = []
        for element in ridge_vertices:
            for i, sub in enumerate(element[:-1]):
                x1 = final_vertices[sub][0]
                x2 = final_vertices[element[i + 1]][0]
                if (isclose(x1, 0.0, abs_tol=tor) and isclose(x2, 0.0, abs_tol=tor)) or (
                    isclose(x1, 1.0, abs_tol=tor) and isclose(x2, 1.0, abs_tol=tor)
                ):
                    continue
                full_ridge_vertices.append([sub, element[i + 1]])

        # De-duplicate vertices (rounded) and remap edges
        self.unique_vertices, inverse = np.unique(
            np.array(final_vertices).round(decimals=4), axis=0, return_inverse=True
        )

        remapped = []
        for a, b in full_ridge_vertices:
            ia, ib = int(inverse[a]), int(inverse[b])
            if ia != ib:
                remapped.append(sorted([ia, ib]))
        self.full_ridge_vertices_unique = np.unique(np.array(remapped), axis=0).tolist()

        # Boundary node sets
        self.lower_bound = np.where(self.unique_vertices[:, 1] == 0.0)
        self.upper_bound = np.where(self.unique_vertices[:, 1] == 1.0)

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

        for idx, center_idx in enumerate(self.center_inside):
            current_neighbors: List[int] = []
            neighbor_cells: List[Polygon] = []
            neighbor_dists: List[float] = []

            poly_current = self.cell_inside[idx]
            p1 = self.centers[center_idx]

            for jdx, other_center in enumerate(self.center_all):
                if center_idx == other_center:
                    continue
                poly_neigh = self.cell_all[jdx]
                if self.check_shared_edges(poly_current, poly_neigh):
                    p2 = self.centers[other_center]
                    neighbor_dists.append(float(np.linalg.norm(np.array(p1) - np.array(p2))))
                    neighbor_cells.append(poly_neigh)
                    current_neighbors.append(other_center)

            # unique + stable order
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
            return float(np.var(n_neigh)) if n_neigh else 0.0
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
    def write_geometry(self, fold_path: str = "./out", file_type: str = "csv") -> None:
        """
        Export geometry (nodes + edges) to CSV or NPY.
        - CSV: nodes.csv (float64), edges.csv (int)
        - NPY: nodes.npy, edges.npy
        """
        if self.unique_vertices is None or self.full_ridge_vertices_unique is None:
            self.voronoi_recon(save_fig=False)

        os.makedirs(fold_path, exist_ok=True)
        if file_type.lower() == "csv":
            np.savetxt(os.path.join(fold_path, "nodes.csv"), self.unique_vertices, delimiter=",")
            np.savetxt(
                os.path.join(fold_path, "edges.csv"),
                np.asarray(self.full_ridge_vertices_unique, dtype=int),
                delimiter=",",
                fmt="%d",
            )
        elif file_type.lower() == "npy":
            np.save(os.path.join(fold_path, "nodes.npy"), self.unique_vertices)
            np.save(os.path.join(fold_path, "edges.npy"), np.asarray(self.full_ridge_vertices_unique, dtype=int))
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'npy'.")

    # --------------------
    # Plotting helpers (lazy matplotlib)
    # --------------------
    def plot_voronoi(self):
        import matplotlib.pyplot as plt
        from scipy.spatial import voronoi_plot_2d

        fig, ax = plt.subplots(figsize=(5, 5))
        voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors="orange", line_width=2)
        ax.plot(*zip(*self.centers), "bo", label="Centers")
        ax.set_title("Voronoi Diagram and Centers")
        ax.legend()
        plt.show()

    def plot_centers_with_pbc(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(*zip(*self.centers_ori), "ro", label="Original Centers")
        ax.plot(*zip(*self.centers), "bo", label="Periodic Centers")
        ax.set_title("Original Centers and Centers with Periodic Boundary Conditions")
        ax.legend()
        plt.show()

    def plot_finite_voronoi(self):
        import matplotlib.pyplot as plt

        regions, vertices, _ = self.voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.get_cmap("rainbow")
        for region in regions:
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.5, color=cmap(np.random.rand()))
        ax.plot(*zip(*self.centers), "ko", label="Centers")
        ax.set_title("Finite Voronoi Regions")
        ax.legend()
        ax.axis("off")
        plt.show()

    def plot_clipped_and_boundary_cells(self):
        import matplotlib.pyplot as plt

        regions, vertices, _ = self.voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.get_cmap("rainbow")
        box = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        for region in regions:
            polygon = vertices[region]
            poly = Polygon(polygon).intersection(box)
            connected_to_boundary = any(
                isclose(p[0], 0.0, abs_tol=1e-5)
                or isclose(p[0], 1.0, abs_tol=1e-5)
                or isclose(p[1], 0.0, abs_tol=1e-5)
                or isclose(p[1], 1.0, abs_tol=1e-5)
                for p in poly.exterior.coords
            )
            color = "r" if connected_to_boundary else "g"
            ax.fill(*zip(*poly.exterior.coords), alpha=0.5, color=color)
        ax.plot(*zip(*self.centers), "ko", label="Centers")
        ax.set_title("Clipped and Boundary Cells")
        ax.legend()
        ax.axis("off")
        plt.show()

    def plot_neighbors(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(*zip(*self.centers), "ko", label="Centers")
        for i, neighbors in enumerate(self.center_neighbor):
            src_idx = self.center_inside[i]  # FIX: map to actual center index
            x1, y1 = self.centers[src_idx]
            for neighbor in neighbors:
                x2, y2 = self.centers[neighbor]
                ax.plot([x1, x2], [y1, y2], "b-", alpha=0.3)
        ax.set_title("Voronoi Cell Neighbor Connectivity")
        ax.legend()
        plt.show()

    def plot_neighbor_distribution(self):
        import matplotlib.pyplot as plt

        n_neighbors = [len(neighbors) for neighbors in self.cell_neighbor]
        plt.hist(n_neighbors, bins=20, color="blue", alpha=0.7)
        plt.title("Distribution of Neighbors per Voronoi Cell")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Frequency")
        plt.show()

    def plot_area_distribution(self):
        import matplotlib.pyplot as plt

        area_values = self.polygon_area_inside  # internal cells only
        plt.hist(area_values, bins=20, color="green", alpha=0.7)
        plt.title("Cell Area Distribution (Internal Cells)")
        plt.xlabel("Cell Area")
        plt.ylabel("Frequency")
        plt.show()

    def plot_lattice_structure(self, filename: Optional[str] = None):
        """
        Plot the Voronoi lattice structure (edges + nodes), no axes.
        If `filename` is provided, save a PNG to that path.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))

        # Plot edges thicker for visibility
        for a, b in self.full_ridge_vertices_unique or []:
            x1, y1 = self.unique_vertices[a]
            x2, y2 = self.unique_vertices[b]
            ax.plot([x1, x2], [y1, y2], "k-", alpha=1.0, linewidth=4.0)

        # Plot nodes
        if self.unique_vertices is not None and len(self.unique_vertices) > 0:
            ax.plot(*zip(*self.unique_vertices), "ro", label="Nodes")

        ax.axis("off")
        if self.unique_vertices is not None and len(self.unique_vertices) > 0:
            xs = self.unique_vertices[:, 0]
            ys = self.unique_vertices[:, 1]
            ax.set_xlim(xs.min() - 0.02, xs.max() + 0.02)
            ax.set_ylim(ys.min() - 0.02, ys.max() + 0.02)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if filename:
            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def plot_initial_centers(self, save_fig: bool = False, fig_name: str = "initial_centers.png"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(*zip(*self.centers), "ko", label="Centers")
        ax.set_title("Initial Centers (after PBC if enabled)")
        ax.axis("equal")
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()

    def plot_voronoi_before_pbc(self, save_fig: bool = False, fig_name: str = "voronoi_before_pbc.png"):
        """
        Plot a Voronoi diagram based on original centers (pre-tiling).
        """
        import matplotlib.pyplot as plt
        from scipy.spatial import voronoi_plot_2d

        vor_pre = Voronoi(self.centers_ori)
        fig, ax = plt.subplots(figsize=(5, 5))
        voronoi_plot_2d(vor_pre, ax=ax, show_vertices=False, line_colors="blue", line_width=1)
        ax.plot(*zip(*self.centers_ori), "ko", label="Original Centers")
        ax.set_title("Voronoi Diagram (Before PBC)")
        ax.axis("equal")
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()

    def plot_with_pbc(self, save_fig: bool = False, fig_name: str = "with_pbc.png"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(*zip(*self.centers_ori), "ko", label="Original Centers")
        ax.plot(*zip(*self.centers), "r+", label="Centers with PBC")
        ax.set_title("Centers with Periodic Boundary Conditions")
        ax.axis("equal")
        ax.legend()
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()

    def plot_finite_voronoi_cells(self, save_fig: bool = False, fig_name: str = "finite_voronoi_cells.png"):
        import matplotlib.pyplot as plt

        regions, vertices, _ = self.voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.get_cmap("rainbow")
        for region in regions:
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color=cmap(np.random.rand()))
        ax.set_title("Finite Voronoi Cells")
        ax.axis("equal")
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()

    def plot_cells_connected_to_boundary(self, save_fig: bool = False, fig_name: str = "boundary_cells.png"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.get_cmap("rainbow")
        for polygon in self.polygon_list:
            arr = np.array(polygon)
            touches = any(
                isclose(p[0], 0.0, abs_tol=1e-5)
                or isclose(p[0], 1.0, abs_tol=1e-5)
                or isclose(p[1], 0.0, abs_tol=1e-5)
                or isclose(p[1], 1.0, abs_tol=1e-5)
                for p in arr
            )
            if touches:
                ax.fill(*zip(*arr), alpha=0.6, color="red", label="Boundary Connected Cells")
            else:
                ax.fill(*zip(*arr), alpha=0.6, color=cmap(np.random.rand()))
        ax.set_title("Cells Connected to Boundary")
        ax.axis("equal")
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()

    def plot_edge_cells_removed(self, save_fig: bool = False, fig_name: str = "edge_cells_removed.png"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.get_cmap("rainbow")
        for polygon in self.polygon_list:
            arr = np.array(polygon)
            touches = any(
                isclose(p[0], 0.0, abs_tol=1e-5)
                or isclose(p[0], 1.0, abs_tol=1e-5)
                or isclose(p[1], 0.0, abs_tol=1e-5)
                or isclose(p[1], 1.0, abs_tol=1e-5)
                for p in arr
            )
            if touches:
                continue
            ax.fill(*zip(*arr), alpha=0.6, color=cmap(np.random.rand()))
        ax.set_title("Edge Cells Removed for Stochasticity Calculation")
        ax.axis("equal")
        if save_fig:
            plt.savefig(fig_name, bbox_inches="tight")
        plt.show()
