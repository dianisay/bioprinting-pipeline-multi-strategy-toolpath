"""Honeycomb scaffold layer planner.

Generates a hex-grid infill pattern inside the wound boundary,
optimizes nozzle travel via TSP, and projects onto the wound surface.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def plan_honeycomb(
    boundary_mm: NDArray[np.float64],
    depth_mm: float,
    hex_size: float = 3.0,
    nozzle_dia: float = 0.4,
) -> dict:
    """Generate honeycomb toolpath for a structural layer.

    Args:
        boundary_mm: (N, 2) wound boundary points in mm
        depth_mm: z-height of this layer (mm below surface)
        hex_size: hexagon side length in mm
        nozzle_dia: nozzle diameter for line width

    Returns:
        dict with waypoints, normals, is_deposition, path_length_mm, num_cells
    """
    x_min, y_min = boundary_mm.min(axis=0)
    x_max, y_max = boundary_mm.max(axis=0)
    void_width = x_max - x_min
    void_length = y_max - y_min

    nx, ny, hex_side = _compute_grid_params(void_width, void_length, hex_size)

    if nx < 1 or ny < 1:
        logger.warning(f"Wound too small for honeycomb: {void_width:.1f}x{void_length:.1f} mm")
        return _empty_result()

    # Generate hex grid centers
    centers = _create_hex_grid(nx, ny, hex_side)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    centers[:, 0] += cx - centers[:, 0].mean()
    centers[:, 1] += cy - centers[:, 1].mean()

    # Filter: keep only cells inside boundary
    from matplotlib.path import Path as MplPath
    boundary_path = MplPath(boundary_mm)
    inside_mask = boundary_path.contains_points(centers)
    centers = centers[inside_mask]
    num_cells = len(centers)

    if num_cells == 0:
        return _empty_result()

    # Generate perimeter path for each cell
    waypoints = []
    is_dep = []

    # Simple nearest-neighbor ordering for cell visit order
    order = _greedy_tsp(centers)

    for i, cell_idx in enumerate(order):
        center = centers[cell_idx]
        perim = _hexagon_perimeter(center, hex_side * 0.85, n_per_edge=8)

        if i > 0:
            # Travel move to next cell
            waypoints.append(perim[0:1])
            is_dep.append(np.array([False]))

        # Deposition around hexagon
        waypoints.append(perim)
        is_dep.append(np.ones(len(perim), dtype=bool))

    all_waypoints = np.vstack(waypoints)
    # Add z coordinate (depth below surface)
    all_waypoints_3d = np.column_stack([
        all_waypoints,
        np.full(len(all_waypoints), -depth_mm),
    ])

    all_is_dep = np.concatenate(is_dep)

    # Normals point up (toward wound opening)
    normals = np.zeros_like(all_waypoints_3d)
    normals[:, 2] = 1.0

    # Path length
    diffs = np.diff(all_waypoints_3d, axis=0)
    path_length = float(np.linalg.norm(diffs, axis=1).sum())

    logger.info(f"Honeycomb layer: {num_cells} cells, {path_length:.0f} mm path, z={-depth_mm:.1f}")

    return {
        "waypoints": all_waypoints_3d,
        "normals": normals,
        "is_deposition": all_is_dep,
        "path_length_mm": path_length,
        "num_cells": num_cells,
    }


def _compute_grid_params(void_width: float, void_length: float, target_size: float):
    hex_side = target_size
    col_spacing = hex_side * 1.5
    row_spacing = hex_side * np.sqrt(3)
    nx = max(1, int(void_width / col_spacing))
    ny = max(1, int(void_length / row_spacing))
    return nx, ny, hex_side


def _create_hex_grid(nx: int, ny: int, hex_side: float) -> NDArray:
    col_spacing = hex_side * 1.5
    row_spacing = hex_side * np.sqrt(3)
    centers = []
    for iy in range(ny):
        for ix in range(nx):
            x = ix * col_spacing
            y = iy * row_spacing
            if ix % 2 == 1:
                y += row_spacing / 2
            centers.append([x, y])
    return np.array(centers, dtype=np.float64)


def _hexagon_perimeter(center: NDArray, side: float, n_per_edge: int = 8) -> NDArray:
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    vertices = np.column_stack([
        center[0] + side * np.cos(angles),
        center[1] + side * np.sin(angles),
    ])
    # Interpolate along edges
    points = []
    for i in range(6):
        start = vertices[i]
        end = vertices[(i + 1) % 6]
        edge_pts = np.linspace(start, end, n_per_edge, endpoint=False)
        points.append(edge_pts)
    return np.vstack(points)


def _greedy_tsp(points: NDArray) -> list[int]:
    """Greedy nearest-neighbor TSP for cell visit order."""
    n = len(points)
    if n <= 1:
        return list(range(n))
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        current = order[-1]
        dists = np.linalg.norm(points - points[current], axis=1)
        dists[visited] = np.inf
        nearest = int(np.argmin(dists))
        order.append(nearest)
        visited[nearest] = True
    return order


def _empty_result() -> dict:
    return {
        "waypoints": np.zeros((0, 3)),
        "normals": np.zeros((0, 3)),
        "is_deposition": np.zeros(0, dtype=bool),
        "path_length_mm": 0.0,
        "num_cells": 0,
    }
