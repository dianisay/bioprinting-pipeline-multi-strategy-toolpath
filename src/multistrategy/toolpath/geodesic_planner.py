"""Geodesic conformal layer planner.

Generates boundary-parallel deposition paths that follow wound surface curvature.
Uses the heat method for geodesic distance computation, then extracts iso-contours
as toolpath passes.

When a surface mesh is available, uses potpourri3d (geometry-central).
Otherwise, falls back to a 2D boundary-offset approximation.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def plan_geodesic(
    boundary_mm: NDArray[np.float64],
    depth_mm: float,
    spacing_mm: float = 1.5,
    adaptive_curvature: bool = True,
    curvature_factor: float = 0.5,
    surface_mesh=None,
) -> dict:
    """Generate geodesic toolpath for a conformal surface layer.

    Args:
        boundary_mm: (N, 2) wound boundary points in mm
        depth_mm: z-height of this layer
        spacing_mm: distance between adjacent geodesic paths
        adaptive_curvature: tighter spacing on high-curvature regions
        curvature_factor: 0=uniform, 1=fully adaptive
        surface_mesh: optional trimesh.Trimesh for true geodesic computation

    Returns:
        dict with waypoints, normals, is_deposition, path_length_mm
    """
    if surface_mesh is not None:
        return _plan_mesh_geodesic(
            boundary_mm, depth_mm, spacing_mm,
            adaptive_curvature, curvature_factor, surface_mesh
        )
    else:
        return _plan_2d_offset(boundary_mm, depth_mm, spacing_mm)


def _plan_mesh_geodesic(
    boundary_mm, depth_mm, spacing_mm,
    adaptive_curvature, curvature_factor, mesh
) -> dict:
    """Geodesic toolpath on a triangulated mesh using heat method."""
    try:
        import potpourri3d as pp3d
        import trimesh
    except ImportError:
        logger.warning("potpourri3d not available, falling back to 2D offset")
        return _plan_2d_offset(boundary_mm, depth_mm, spacing_mm)

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Find boundary vertex (closest to centroid of boundary_mm)
    centroid_2d = boundary_mm.mean(axis=0)
    dists_to_centroid = np.linalg.norm(vertices[:, :2] - centroid_2d, axis=1)
    source_vertex = int(np.argmin(dists_to_centroid))

    # Compute geodesic distances
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    distances = solver.compute_distance(source_vertex)

    # Extract iso-contours
    d_max = distances.max()
    contour_levels = np.arange(spacing_mm, d_max, spacing_mm)

    all_waypoints = []
    all_is_dep = []

    for level in contour_levels:
        contour_pts = _extract_mesh_contour(vertices, faces, distances, level)
        if len(contour_pts) < 3:
            continue

        if all_waypoints:
            # Travel to start of next contour
            all_waypoints.append(contour_pts[0:1])
            all_is_dep.append(np.array([False]))

        all_waypoints.append(contour_pts)
        all_is_dep.append(np.ones(len(contour_pts), dtype=bool))

    if not all_waypoints:
        return _plan_2d_offset(boundary_mm, depth_mm, spacing_mm)

    waypoints = np.vstack(all_waypoints)
    is_dep = np.concatenate(all_is_dep)

    # Compute vertex normals at waypoints
    normals = _interpolate_normals(waypoints, mesh)

    diffs = np.diff(waypoints, axis=0)
    path_length = float(np.linalg.norm(diffs, axis=1).sum())

    num_paths = len(contour_levels)
    logger.info(f"Geodesic layer (mesh): {num_paths} paths, {path_length:.0f} mm, z={-depth_mm:.1f}")

    return {
        "waypoints": waypoints,
        "normals": normals,
        "is_deposition": is_dep,
        "path_length_mm": path_length,
    }


def _plan_2d_offset(
    boundary_mm: NDArray[np.float64],
    depth_mm: float,
    spacing_mm: float,
) -> dict:
    """Fallback: 2D boundary-offset approximation (no mesh needed).

    Shrinks boundary inward at regular intervals to produce
    approximately boundary-parallel contours.
    """
    centroid = boundary_mm.mean(axis=0)
    radii = np.linalg.norm(boundary_mm - centroid, axis=1)
    max_radius = radii.max()
    angles = np.arctan2(boundary_mm[:, 1] - centroid[1], boundary_mm[:, 0] - centroid[0])

    # Sort by angle for smooth contour
    sort_idx = np.argsort(angles)
    radii_sorted = radii[sort_idx]
    angles_sorted = angles[sort_idx]

    num_contours = max(1, int(max_radius / spacing_mm))
    all_waypoints = []
    all_is_dep = []

    for i in range(num_contours):
        offset = (i + 1) * spacing_mm
        shrink_factor = max(0.0, 1.0 - offset / max_radius)
        if shrink_factor < 0.05:
            break

        # Generate contour at this offset
        n_pts = max(32, int(64 * shrink_factor))
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        # Interpolate radii at these angles
        r_interp = np.interp(theta, angles_sorted + np.pi, radii_sorted, period=2 * np.pi)
        r_offset = r_interp * shrink_factor

        contour = np.column_stack([
            centroid[0] + r_offset * np.cos(theta),
            centroid[1] + r_offset * np.sin(theta),
            np.full(n_pts, -depth_mm),
        ])

        if all_waypoints:
            all_waypoints.append(contour[0:1])
            all_is_dep.append(np.array([False]))

        all_waypoints.append(contour)
        all_is_dep.append(np.ones(n_pts, dtype=bool))

    if not all_waypoints:
        return {
            "waypoints": np.zeros((0, 3)),
            "normals": np.zeros((0, 3)),
            "is_deposition": np.zeros(0, dtype=bool),
            "path_length_mm": 0.0,
        }

    waypoints = np.vstack(all_waypoints)
    is_dep = np.concatenate(all_is_dep)

    normals = np.zeros_like(waypoints)
    normals[:, 2] = 1.0  # Upward for flat approximation

    diffs = np.diff(waypoints, axis=0)
    path_length = float(np.linalg.norm(diffs, axis=1).sum())

    logger.info(f"Geodesic layer (2D offset): {num_contours} contours, {path_length:.0f} mm")

    return {
        "waypoints": waypoints,
        "normals": normals,
        "is_deposition": is_dep,
        "path_length_mm": path_length,
    }


def _extract_mesh_contour(vertices, faces, distances, level, n_points=100):
    """Extract iso-contour at given geodesic distance level via marching."""
    contour_pts = []
    for face in faces:
        d = distances[face]
        for i in range(3):
            j = (i + 1) % 3
            if (d[i] - level) * (d[j] - level) < 0:
                t = (level - d[i]) / (d[j] - d[i] + 1e-10)
                pt = vertices[face[i]] * (1 - t) + vertices[face[j]] * t
                contour_pts.append(pt)

    if len(contour_pts) < 2:
        return np.zeros((0, 3))

    pts = np.array(contour_pts)
    # Order by angle around centroid
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    return pts[order]


def _interpolate_normals(waypoints, mesh):
    """Compute surface normals at waypoint locations."""
    try:
        closest, _, face_ids = mesh.nearest.on_surface(waypoints)
        normals = mesh.face_normals[face_ids]
    except Exception:
        normals = np.zeros_like(waypoints)
        normals[:, 2] = 1.0
    return normals
