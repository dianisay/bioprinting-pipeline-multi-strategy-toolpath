"""Hybrid layer planner: honeycomb core + geodesic perimeter.

For transition layers (between deep structural and surface conformal),
uses honeycomb in the center and a geodesic boundary ring. This mimics
the basement membrane architecture in skin tissue.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from multistrategy.toolpath.honeycomb_planner import plan_honeycomb
from multistrategy.toolpath.geodesic_planner import plan_geodesic

logger = logging.getLogger(__name__)


def plan_hybrid(
    boundary_mm: NDArray[np.float64],
    depth_mm: float,
    hex_size: float = 3.0,
    geodesic_spacing: float = 1.5,
    perimeter_fraction: float = 0.25,
    surface_mesh=None,
) -> dict:
    """Generate hybrid toolpath: honeycomb core + geodesic perimeter ring.

    The outer `perimeter_fraction` of the wound width gets geodesic paths
    (for conformal edge coverage). The inner region gets honeycomb
    (for structural support).

    Args:
        boundary_mm: (N, 2) wound boundary in mm
        depth_mm: z-height of this layer
        hex_size: hexagon cell size for core
        geodesic_spacing: path spacing for perimeter
        perimeter_fraction: what fraction of radius is geodesic ring
        surface_mesh: optional mesh for true geodesics

    Returns:
        dict with waypoints, normals, is_deposition, path_length_mm, num_cells
    """
    centroid = boundary_mm.mean(axis=0)
    radii = np.linalg.norm(boundary_mm - centroid, axis=1)

    # Shrink boundary for honeycomb core
    shrink = 1.0 - perimeter_fraction
    inner_boundary = centroid + (boundary_mm - centroid) * shrink

    # Plan honeycomb on inner region
    honeycomb_result = plan_honeycomb(inner_boundary, depth_mm, hex_size)

    # Plan geodesic ring on outer perimeter (between inner and outer boundary)
    # Use only 2-3 contours in the perimeter zone
    perimeter_width = radii.mean() * perimeter_fraction
    n_perimeter_passes = max(1, int(perimeter_width / geodesic_spacing))

    geodesic_result = plan_geodesic(
        boundary_mm, depth_mm,
        spacing_mm=perimeter_width / max(n_perimeter_passes, 1),
        surface_mesh=surface_mesh,
    )

    # Filter geodesic waypoints to only keep those in the perimeter zone
    if len(geodesic_result["waypoints"]) > 0:
        geo_pts = geodesic_result["waypoints"][:, :2]
        dist_from_center = np.linalg.norm(geo_pts - centroid, axis=1)
        inner_radius = radii.mean() * shrink
        perimeter_mask = dist_from_center >= inner_radius * 0.9
        geo_waypoints = geodesic_result["waypoints"][perimeter_mask]
        geo_normals = geodesic_result["normals"][perimeter_mask]
        geo_is_dep = geodesic_result["is_deposition"][perimeter_mask]
    else:
        geo_waypoints = np.zeros((0, 3))
        geo_normals = np.zeros((0, 3))
        geo_is_dep = np.zeros(0, dtype=bool)

    # Combine: honeycomb first (deeper support), then geodesic perimeter
    if len(honeycomb_result["waypoints"]) > 0 and len(geo_waypoints) > 0:
        # Add travel move between honeycomb and geodesic
        travel = geo_waypoints[0:1]
        waypoints = np.vstack([
            honeycomb_result["waypoints"],
            travel,
            geo_waypoints,
        ])
        normals = np.vstack([
            honeycomb_result["normals"],
            geo_normals[0:1],
            geo_normals,
        ])
        is_dep = np.concatenate([
            honeycomb_result["is_deposition"],
            np.array([False]),
            geo_is_dep,
        ])
    elif len(honeycomb_result["waypoints"]) > 0:
        waypoints = honeycomb_result["waypoints"]
        normals = honeycomb_result["normals"]
        is_dep = honeycomb_result["is_deposition"]
    else:
        waypoints = geo_waypoints
        normals = geo_normals
        is_dep = geo_is_dep

    diffs = np.diff(waypoints, axis=0) if len(waypoints) > 1 else np.zeros((0, 3))
    path_length = float(np.linalg.norm(diffs, axis=1).sum())

    logger.info(
        f"Hybrid layer: {honeycomb_result['num_cells']} hex cells + "
        f"{len(geo_waypoints)} geodesic pts, {path_length:.0f} mm"
    )

    return {
        "waypoints": waypoints,
        "normals": normals,
        "is_deposition": is_dep,
        "path_length_mm": path_length,
        "num_cells": honeycomb_result["num_cells"],
    }
