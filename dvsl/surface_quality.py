"""Topology-aware quality control for triangular surface meshes.

The module deliberately distinguishes the *surface* area Jacobian

    J_A = area(deformed triangle) / area(source triangle)

from an ambient 3-D Jacobian determinant.  Vertex positions on a triangular
surface do not identify a unique 3 x 3 derivative of an ambient flow.

Self-intersections are found with an AABB sweep broad phase followed by a
tolerance-aware triangle/triangle separating-axis test.  Pairs sharing any
vertex index are excluded because neighboring triangles meet by construction.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyvista as pv

try:
    from vtkmodules.vtkCommonDataModel import vtkTriangle
except ImportError:  # The SAT implementation below is the portable fallback.
    vtkTriangle = None

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:  # CSV and JSON reports remain available.
    Workbook = None


QC_DEFINITION_VERSION = "mesh-qc-v1"
REPLACEMENT_RULE = (
    "Do not replace or exclude automatically. If a selected target fails, retain "
    "the original intended-subject manifest and screen unused targets with the "
    "same frozen criteria. For each failed subject, choose the unused QC-pass "
    "target with the smallest absolute difference in sorted-cohort rank; break "
    "ties by the numerically smaller subject ID. Record the proposed substitution "
    "and obtain approval before any training is started or restarted."
)


@dataclass(frozen=True)
class MeshQCConfig:
    """Frozen, scale-normalized thresholds used by mesh QC.

    ``near_degenerate_area_ratio`` is triangle area divided by the squared
    bounding-box diagonal.  ``intersection_tolerance_ratio`` is a length
    tolerance divided by the bounding-box diagonal.  ``collapse_threshold``
    and ``expansion_threshold`` are dimensionless surface area ratios.
    """

    near_degenerate_area_ratio: float = 1.0e-12
    intersection_tolerance_ratio: float = 1.0e-12
    collapse_threshold: float = 0.2
    expansion_threshold: float = 5.0
    triangle_quality_review_threshold: float = 1.0e-4

    def validate(self) -> None:
        if self.near_degenerate_area_ratio < 0:
            raise ValueError("near_degenerate_area_ratio must be non-negative")
        if self.intersection_tolerance_ratio < 0:
            raise ValueError("intersection_tolerance_ratio must be non-negative")
        if not 0 < self.collapse_threshold < 1:
            raise ValueError("collapse_threshold must lie in (0, 1)")
        if self.expansion_threshold <= 1:
            raise ValueError("expansion_threshold must be greater than 1")
        if not 0 <= self.triangle_quality_review_threshold <= 1:
            raise ValueError("triangle_quality_review_threshold must lie in [0, 1]")


DEFAULT_QC_CONFIG = MeshQCConfig()


def load_triangle_mesh(path: Path | str) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """Load a surface and return triangulated PolyData, points, and faces."""

    mesh = pv.read(Path(path))
    if isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface()
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(f"Unsupported mesh type in {path}: {type(mesh)}")
    mesh = mesh.triangulate()
    points = np.array(mesh.points, dtype=np.float64, copy=True)
    face_stream = np.asarray(mesh.faces, dtype=np.int64)
    if face_stream.size == 0:
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        cells = face_stream.reshape(-1, 4)
        if np.any(cells[:, 0] != 3):
            raise ValueError(f"Triangulation produced non-triangular cells in {path}")
        faces = np.array(cells[:, 1:4], dtype=np.int64, copy=True)
    return mesh, points, faces


def _distribution(
    values: np.ndarray,
    prefix: str,
    *,
    percentiles: Iterable[int] = (1, 5, 50, 95, 99),
) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    result: dict[str, float | int] = {f"{prefix}_valid_count": int(finite.size)}
    percentile_names = tuple(
        "median" if percentile == 50 else f"p{percentile}"
        for percentile in percentiles
    )
    names = ("min", *percentile_names, "max", "mean", "std")
    if finite.size == 0:
        result.update({f"{prefix}_{name}": math.nan for name in names})
        return result
    result[f"{prefix}_min"] = float(np.min(finite))
    for percentile in percentiles:
        label = "median" if percentile == 50 else f"p{percentile}"
        result[f"{prefix}_{label}"] = float(np.percentile(finite, percentile))
    result[f"{prefix}_max"] = float(np.max(finite))
    result[f"{prefix}_mean"] = float(np.mean(finite))
    result[f"{prefix}_std"] = float(np.std(finite))
    return result


def _canonical_face_duplicates(faces: np.ndarray) -> tuple[int, int]:
    if len(faces) == 0:
        return 0, 0
    canonical = np.sort(faces, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    duplicate_groups = counts[counts > 1]
    return int(np.sum(duplicate_groups - 1)), int(duplicate_groups.size)


def _exact_vertex_duplicates(points: np.ndarray) -> tuple[int, int]:
    finite_rows = points[np.isfinite(points).all(axis=1)]
    if len(finite_rows) == 0:
        return 0, 0
    _, counts = np.unique(finite_rows, axis=0, return_counts=True)
    duplicate_groups = counts[counts > 1]
    return int(np.sum(duplicate_groups - 1)), int(duplicate_groups.size)


def _face_geometry(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = points[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.divide(
        cross,
        double_area[:, None],
        out=np.full_like(cross, np.nan),
        where=double_area[:, None] > 0,
    )
    return triangles, 0.5 * double_area, normals


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(
    point: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance: float
) -> bool:
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
    orientation_tolerance: float,
) -> bool:
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if (
        ((o1 > orientation_tolerance and o2 < -orientation_tolerance)
         or (o1 < -orientation_tolerance and o2 > orientation_tolerance))
        and ((o3 > orientation_tolerance and o4 < -orientation_tolerance)
             or (o3 < -orientation_tolerance and o4 > orientation_tolerance))
    ):
        return True
    return bool(
        (abs(o1) <= orientation_tolerance and _point_on_segment_2d(c, a, b, tolerance))
        or (abs(o2) <= orientation_tolerance and _point_on_segment_2d(d, a, b, tolerance))
        or (abs(o3) <= orientation_tolerance and _point_on_segment_2d(a, c, d, tolerance))
        or (abs(o4) <= orientation_tolerance and _point_on_segment_2d(b, c, d, tolerance))
    )


def _point_in_triangle_2d(
    point: np.ndarray, triangle: np.ndarray, orientation_tolerance: float
) -> bool:
    signs = np.asarray(
        [
            _orient2d(triangle[0], triangle[1], point),
            _orient2d(triangle[1], triangle[2], point),
            _orient2d(triangle[2], triangle[0], point),
        ]
    )
    return bool(
        np.all(signs >= -orientation_tolerance)
        or np.all(signs <= orientation_tolerance)
    )


def _coplanar_triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    local_scale = max(
        float(np.ptp(np.vstack((first_2d, second_2d)), axis=0).max()),
        tolerance,
    )
    orientation_tolerance = tolerance * local_scale
    for i in range(3):
        for j in range(3):
            if _segments_intersect_2d(
                first_2d[i],
                first_2d[(i + 1) % 3],
                second_2d[j],
                second_2d[(j + 1) % 3],
                tolerance,
                orientation_tolerance,
            ):
                return True
    return _point_in_triangle_2d(
        first_2d[0], second_2d, orientation_tolerance
    ) or _point_in_triangle_2d(second_2d[0], first_2d, orientation_tolerance)


def _vtk_triangles_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    if vtkTriangle is None:
        return False
    return bool(
        vtkTriangle.TrianglesIntersect(
            first[0],
            first[1],
            first[2],
            second[0],
            second[1],
            second[2],
        )
    )


def triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
    *,
    use_vtk: bool = True,
) -> bool:
    """Narrow-phase intersection test for two non-degenerate 3-D triangles."""

    if use_vtk and _vtk_triangles_intersect(first, second):
        return True

    edges_first = np.roll(first, -1, axis=0) - first
    edges_second = np.roll(second, -1, axis=0) - second
    normal_first = np.cross(edges_first[0], edges_first[1])
    normal_second = np.cross(edges_second[0], edges_second[1])
    norm_first = float(np.linalg.norm(normal_first))
    norm_second = float(np.linalg.norm(normal_second))
    if norm_first == 0 or norm_second == 0:
        return False
    unit_first = normal_first / norm_first
    unit_second = normal_second / norm_second

    distance_second_to_first = (second - first[0]) @ unit_first
    distance_first_to_second = (first - second[0]) @ unit_second
    if (
        np.min(distance_second_to_first) > tolerance
        or np.max(distance_second_to_first) < -tolerance
        or np.min(distance_first_to_second) > tolerance
        or np.max(distance_first_to_second) < -tolerance
    ):
        return False

    coplanar = bool(
        np.max(np.abs(distance_second_to_first)) <= tolerance
        and np.max(np.abs(distance_first_to_second)) <= tolerance
        and abs(float(np.dot(unit_first, unit_second))) >= 1.0 - 1.0e-10
    )
    if coplanar:
        return _coplanar_triangles_intersect(
            first, second, unit_first, tolerance
        )

    axes: list[np.ndarray] = [unit_first, unit_second]
    for edge_first in edges_first:
        length_first = float(np.linalg.norm(edge_first))
        for edge_second in edges_second:
            length_second = float(np.linalg.norm(edge_second))
            axis = np.cross(edge_first, edge_second)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12 * length_first * length_second:
                continue
            axes.append(axis / axis_norm)
    for axis in axes:
        projection_first = first @ axis
        projection_second = second @ axis
        if (
            np.max(projection_first) < np.min(projection_second) - tolerance
            or np.max(projection_second) < np.min(projection_first) - tolerance
        ):
            return False
    return True


def self_intersection_metrics(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Count intersecting non-adjacent triangle pairs.

    The broad phase sweeps triangle AABBs along x.  The narrow phase uses
    :func:`triangles_intersect`.  A pair sharing one or more vertex indices is
    excluded before the narrow phase, covering shared-vertex and shared-edge
    neighbors.
    """

    triangles, areas, _ = _face_geometry(points, faces)
    valid_triangle = np.isfinite(triangles).all(axis=(1, 2)) & (areas > 0)
    minimum = np.min(triangles, axis=1) - tolerance
    maximum = np.max(triangles, axis=1) + tolerance
    order = np.argsort(minimum[:, 0], kind="mergesort")
    active: list[int] = []
    intersecting_pairs: list[tuple[int, int]] = []
    broad_phase_pairs = 0
    adjacent_pairs_excluded = 0
    vtk_positive_pairs = 0
    sat_fallback_positive_pairs = 0

    for current in order.tolist():
        if not valid_triangle[current]:
            continue
        current_min_x = minimum[current, 0]
        active = [candidate for candidate in active if maximum[candidate, 0] >= current_min_x]
        current_face = faces[current]
        for candidate in active:
            if (
                maximum[candidate, 1] < minimum[current, 1]
                or maximum[current, 1] < minimum[candidate, 1]
                or maximum[candidate, 2] < minimum[current, 2]
                or maximum[current, 2] < minimum[candidate, 2]
            ):
                continue
            candidate_face = faces[candidate]
            if np.any(current_face[:, None] == candidate_face[None, :]):
                adjacent_pairs_excluded += 1
                continue
            broad_phase_pairs += 1
            first_triangle = triangles[candidate]
            second_triangle = triangles[current]
            vtk_hit = _vtk_triangles_intersect(
                first_triangle, second_triangle
            )
            sat_hit = (
                False
                if vtk_hit
                else triangles_intersect(
                    first_triangle,
                    second_triangle,
                    tolerance,
                    use_vtk=False,
                )
            )
            if vtk_hit or sat_hit:
                intersecting_pairs.append((candidate, current))
                vtk_positive_pairs += int(vtk_hit)
                sat_fallback_positive_pairs += int(sat_hit)
        active.append(current)

    affected = (
        np.unique(np.asarray(intersecting_pairs, dtype=np.int64))
        if intersecting_pairs
        else np.empty(0, dtype=np.int64)
    )
    return {
        "self_intersection_algorithm": (
            "aabb_sweep_plus_vtk_triangle_intersection_with_sat_fallback"
            if vtkTriangle is not None
            else "aabb_sweep_plus_triangle_sat"
        ),
        "self_intersection_computation_status": "ok",
        "self_intersection_candidate_pair_count": int(broad_phase_pairs),
        "self_intersection_vtk_positive_pair_count": int(vtk_positive_pairs),
        "self_intersection_sat_fallback_positive_pair_count": int(
            sat_fallback_positive_pairs
        ),
        "self_intersection_adjacent_pair_exclusion_count": int(
            adjacent_pairs_excluded
        ),
        "self_intersection_pair_count": int(len(intersecting_pairs)),
        "self_intersection_pair_indices_sample": ";".join(
            f"{left}-{right}" for left, right in intersecting_pairs[:100]
        ),
        "self_intersection_pair_indices_sample_truncated": bool(
            len(intersecting_pairs) > 100
        ),
        "self_intersection_affected_triangle_count": int(affected.size),
        "self_intersection_affected_triangle_fraction": (
            float(affected.size / len(faces)) if len(faces) else math.nan
        ),
        "self_intersection_present": bool(intersecting_pairs),
    }


def _edge_and_component_metrics(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if len(faces) == 0:
        return {
            "edge_count": 0,
            "boundary_edge_count": 0,
            "non_manifold_edge_count": 0,
            "orientation_inconsistent_edge_count": 0,
            "connected_component_count": 0,
            "largest_component_face_fraction": math.nan,
            "isolated_vertex_count": int(len(points)),
            "watertight": False,
            "orientation_consistent": False,
        }, np.empty((0, 2), dtype=np.int64)

    directed_edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    undirected_edges = np.sort(directed_edges, axis=1)
    unique_edges, inverse, counts = np.unique(
        undirected_edges, axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)
    orientation_sum = np.bincount(
        inverse, weights=signs, minlength=len(unique_edges)
    )
    manifold = counts == 2
    inconsistent = manifold & (np.abs(orientation_sum) == 2)

    face_ids = np.concatenate(
        (np.arange(len(faces)), np.arange(len(faces)), np.arange(len(faces)))
    )
    edge_order = np.argsort(inverse, kind="mergesort")
    group_starts = np.cumsum(counts) - counts

    parent = np.arange(len(faces))
    sizes = np.ones(len(faces), dtype=np.int64)

    def find_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left = find_root(left)
        root_right = find_root(right)
        if root_left == root_right:
            return
        if sizes[root_left] < sizes[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        sizes[root_left] += sizes[root_right]

    for start, count in zip(group_starts.tolist(), counts.tolist()):
        if count < 2:
            continue
        incident = face_ids[edge_order[start : start + count]]
        first = int(incident[0])
        for other in incident[1:]:
            union(first, int(other))
    roots = np.asarray([find_root(index) for index in range(len(faces))])
    _, component_sizes = np.unique(roots, return_counts=True)

    used_vertices = np.unique(faces)
    metrics: dict[str, Any] = {
        "edge_count": int(len(unique_edges)),
        "boundary_edge_count": int(np.sum(counts == 1)),
        "non_manifold_edge_count": int(np.sum(counts > 2)),
        "orientation_inconsistent_edge_count": int(np.sum(inconsistent)),
        "connected_component_count": int(component_sizes.size),
        "largest_component_face_fraction": float(
            np.max(component_sizes) / len(faces)
        ),
        "isolated_vertex_count": int(len(points) - len(used_vertices)),
        "watertight": bool(np.all(counts == 2)),
        "orientation_consistent": bool(
            not np.any(inconsistent) and not np.any(counts > 2)
        ),
    }

    manifold_indices = np.flatnonzero(manifold)
    if manifold_indices.size:
        first_positions = group_starts[manifold_indices]
        first_faces = face_ids[edge_order[first_positions]]
        second_faces = face_ids[edge_order[first_positions + 1]]
        dots = np.sum(normals[first_faces] * normals[second_faces], axis=1)
        angles_degrees = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        metrics.update(_distribution(angles_degrees, "dihedral_angle_degrees"))
        normal_variation = 1.0 - np.clip(dots, -1.0, 1.0)
        metrics.update(_distribution(normal_variation, "normal_variation"))
    else:
        metrics.update(_distribution(np.empty(0), "dihedral_angle_degrees"))
        metrics.update(_distribution(np.empty(0), "normal_variation"))
    return metrics, unique_edges


def _shape_metrics(
    points: np.ndarray,
    faces: np.ndarray,
    triangles: np.ndarray,
    areas: np.ndarray,
    unique_edges: np.ndarray,
    bbox_diagonal: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    triangle_edges = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    squared_sum = np.sum(triangle_edges**2, axis=1)
    quality = np.divide(
        4.0 * math.sqrt(3.0) * areas,
        squared_sum,
        out=np.full_like(areas, np.nan),
        where=squared_sum > 0,
    )
    result.update(_distribution(quality, "triangle_quality"))

    if len(unique_edges):
        edge_vectors = points[unique_edges[:, 1]] - points[unique_edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        finite_lengths = edge_lengths[np.isfinite(edge_lengths)]
    else:
        finite_lengths = np.empty(0)
    result.update(_distribution(finite_lengths, "edge_length"))
    edge_mean = (
        float(np.mean(finite_lengths)) if finite_lengths.size else math.nan
    )
    result["edge_length_cv"] = (
        float(np.std(finite_lengths) / edge_mean)
        if finite_lengths.size and edge_mean > 0
        else math.nan
    )
    result["mean_edge_length_over_bbox_diagonal"] = (
        edge_mean / bbox_diagonal
        if math.isfinite(edge_mean) and bbox_diagonal > 0
        else math.nan
    )

    if len(unique_edges) and len(points):
        neighbor_sum = np.zeros_like(points, dtype=np.float64)
        degree = np.zeros(len(points), dtype=np.int64)
        left = unique_edges[:, 0]
        right = unique_edges[:, 1]
        np.add.at(neighbor_sum, left, points[right])
        np.add.at(neighbor_sum, right, points[left])
        np.add.at(degree, left, 1)
        np.add.at(degree, right, 1)
        supported = degree > 0
        laplacian = points[supported] - (
            neighbor_sum[supported] / degree[supported, None]
        )
        numerator = float(np.mean(np.sum(laplacian**2, axis=1)))
        denominator = float(np.mean(finite_lengths**2))
        result["normalized_umbrella_laplacian_energy"] = (
            numerator / denominator if denominator > 0 else math.nan
        )
        result["laplacian_supported_vertex_count"] = int(np.sum(supported))
    else:
        result["normalized_umbrella_laplacian_energy"] = math.nan
        result["laplacian_supported_vertex_count"] = 0
    return result


def mesh_quality_metrics(
    points: np.ndarray,
    faces: np.ndarray,
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
    *,
    compute_self_intersections: bool = True,
) -> dict[str, Any]:
    """Compute intrinsic validity, topology, intersection, and quality metrics."""

    config.validate()
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (V, 3), got {points.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")

    finite_coordinate_mask = np.isfinite(points)
    finite_vertex_mask = finite_coordinate_mask.all(axis=1)
    valid_indices = (
        (faces >= 0).all(axis=1) & (faces < len(points)).all(axis=1)
        if len(faces)
        else np.empty(0, dtype=bool)
    )
    repeated_vertex_face = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
        if len(faces)
        else np.empty(0, dtype=bool)
    )
    valid_faces = faces[valid_indices]
    finite_face_geometry = (
        finite_vertex_mask[valid_faces].all(axis=1)
        if len(valid_faces)
        else np.empty(0, dtype=bool)
    )
    geometric_faces = valid_faces[finite_face_geometry]

    duplicate_vertices, duplicate_vertex_groups = _exact_vertex_duplicates(points)
    duplicate_faces, duplicate_face_groups = _canonical_face_duplicates(valid_faces)
    finite_points = points[finite_vertex_mask]
    if len(finite_points):
        bbox_extent = np.ptp(finite_points, axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_extent))
    else:
        bbox_diagonal = math.nan
    scale_squared = bbox_diagonal**2 if math.isfinite(bbox_diagonal) else math.nan
    near_area_threshold = (
        config.near_degenerate_area_ratio * scale_squared
        if math.isfinite(scale_squared)
        else math.nan
    )

    result: dict[str, Any] = {
        "qc_definition_version": QC_DEFINITION_VERSION,
        "vertex_count": int(len(points)),
        "face_count": int(len(faces)),
        "finite_coordinates": bool(np.all(finite_coordinate_mask)),
        "nonfinite_coordinate_count": int(np.sum(~finite_coordinate_mask)),
        "nonfinite_vertex_count": int(np.sum(~finite_vertex_mask)),
        "duplicate_vertex_count": duplicate_vertices,
        "duplicate_vertex_group_count": duplicate_vertex_groups,
        "duplicate_face_count": duplicate_faces,
        "duplicate_face_group_count": duplicate_face_groups,
        "invalid_face_index_count": int(np.sum(~valid_indices)),
        "repeated_vertex_face_count": int(np.sum(repeated_vertex_face)),
        "bbox_diagonal": bbox_diagonal,
        "near_degenerate_area_ratio_threshold": (
            config.near_degenerate_area_ratio
        ),
        "near_degenerate_area_threshold_mesh_units_squared": near_area_threshold,
    }

    if len(geometric_faces):
        triangles, areas, normals = _face_geometry(points, geometric_faces)
        degenerate = areas == 0
        near_degenerate = (
            (~degenerate) & (areas <= near_area_threshold)
            if math.isfinite(near_area_threshold)
            else np.zeros_like(degenerate)
        )
        result["surface_area_mesh_units_squared"] = float(np.sum(areas))
        result["degenerate_face_count"] = int(np.sum(degenerate))
        result["near_degenerate_face_count"] = int(np.sum(near_degenerate))
        result["degenerate_or_near_degenerate_face_fraction"] = float(
            np.mean(degenerate | near_degenerate)
        )
        normalized_areas = (
            areas / scale_squared
            if math.isfinite(scale_squared) and scale_squared > 0
            else np.full_like(areas, np.nan)
        )
        result.update(_distribution(normalized_areas, "normalized_face_area"))
        edge_metrics, unique_edges = _edge_and_component_metrics(
            points, valid_faces, _face_geometry(points, valid_faces)[2]
        )
        result.update(edge_metrics)
        result.update(
            _shape_metrics(
                points,
                geometric_faces,
                triangles,
                areas,
                unique_edges,
                bbox_diagonal,
            )
        )
    else:
        result.update(
            {
                "surface_area_mesh_units_squared": math.nan,
                "degenerate_face_count": 0,
                "near_degenerate_face_count": 0,
                "degenerate_or_near_degenerate_face_fraction": math.nan,
            }
        )
        result.update(_distribution(np.empty(0), "normalized_face_area"))
        edge_metrics, _ = _edge_and_component_metrics(
            points, valid_faces, np.empty((len(valid_faces), 3))
        )
        result.update(edge_metrics)
        result.update(_distribution(np.empty(0), "triangle_quality"))
        result.update(_distribution(np.empty(0), "edge_length"))
        result.update(
            {
                "edge_length_cv": math.nan,
                "mean_edge_length_over_bbox_diagonal": math.nan,
                "normalized_umbrella_laplacian_energy": math.nan,
                "laplacian_supported_vertex_count": 0,
            }
        )

    can_intersect = bool(
        len(faces)
        and np.all(valid_indices)
        and np.all(finite_vertex_mask)
        and not np.any(repeated_vertex_face)
    )
    if compute_self_intersections and can_intersect:
        tolerance = config.intersection_tolerance_ratio * max(
            bbox_diagonal, np.finfo(np.float64).tiny
        )
        result["self_intersection_tolerance_mesh_units"] = tolerance
        result.update(
            self_intersection_metrics(points, faces, tolerance=tolerance)
        )
    else:
        status = (
            "disabled"
            if not compute_self_intersections
            else "not_computed_invalid_mesh"
        )
        result.update(
            {
                "self_intersection_algorithm": (
                    "aabb_sweep_plus_vtk_triangle_intersection_with_sat_fallback"
                    if vtkTriangle is not None
                    else "aabb_sweep_plus_triangle_sat"
                ),
                "self_intersection_computation_status": status,
                "self_intersection_tolerance_mesh_units": math.nan,
                "self_intersection_candidate_pair_count": 0,
                "self_intersection_vtk_positive_pair_count": 0,
                "self_intersection_sat_fallback_positive_pair_count": 0,
                "self_intersection_adjacent_pair_exclusion_count": 0,
                "self_intersection_pair_count": math.nan,
                "self_intersection_pair_indices_sample": "",
                "self_intersection_pair_indices_sample_truncated": False,
                "self_intersection_affected_triangle_count": math.nan,
                "self_intersection_affected_triangle_fraction": math.nan,
                "self_intersection_present": None,
            }
        )
    return result


def deformation_quality_metrics(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    deformed_points: np.ndarray,
    deformed_faces: np.ndarray,
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
) -> dict[str, Any]:
    """Measure the indexed source-to-deformed triangular surface map."""

    config.validate()
    source_points = np.asarray(source_points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    deformed_points = np.asarray(deformed_points, dtype=np.float64)
    deformed_faces = np.asarray(deformed_faces, dtype=np.int64)
    topology_match = bool(
        source_points.shape == deformed_points.shape
        and source_faces.shape == deformed_faces.shape
        and np.array_equal(source_faces, deformed_faces)
    )
    result: dict[str, Any] = {
        "surface_jacobian_definition": (
            "J_A = deformed triangle area / source triangle area"
        ),
        "surface_jacobian_units": "dimensionless area ratio",
        "surface_jacobian_supported": topology_match,
        "surface_jacobian_topology_requirement": (
            "identical vertex indexing and identical oriented triangle array"
        ),
        "surface_jacobian_collapse_threshold": config.collapse_threshold,
        "surface_jacobian_expansion_threshold": config.expansion_threshold,
        "ambient_3d_jacobian_determinant_supported": False,
        "ambient_3d_jacobian_note": (
            "Unsupported from endpoint surface vertices alone; no unique 3x3 "
            "ambient derivative or divergence-integrated flow determinant is "
            "available in these artifacts."
        ),
        "fold_indicator_note": (
            "J_A is unsigned and is not used as a fold determinant. Intrinsic "
            "orientation inconsistency and non-adjacent self-intersection are "
            "reported separately as mesh/fold warning indicators."
        ),
    }
    if not topology_match or not np.isfinite(source_points).all() or not np.isfinite(
        deformed_points
    ).all():
        result["surface_jacobian_status"] = (
            "unsupported_topology"
            if not topology_match
            else "unsupported_nonfinite_coordinates"
        )
        result.update(_distribution(np.empty(0), "surface_jacobian"))
        result["surface_jacobian_log_mean"] = math.nan
        result["surface_jacobian_log_std"] = math.nan
        result["surface_jacobian_collapsed_face_count"] = 0
        result["surface_jacobian_collapsed_face_fraction"] = math.nan
        result["surface_jacobian_expanded_face_count"] = 0
        result["surface_jacobian_expanded_face_fraction"] = math.nan
        result["surface_jacobian_invalid_source_face_count"] = int(
            len(source_faces)
        )
        return result

    _, source_areas, _ = _face_geometry(source_points, source_faces)
    _, deformed_areas, _ = _face_geometry(deformed_points, deformed_faces)
    finite_source = source_points[np.isfinite(source_points).all(axis=1)]
    bbox_diagonal = (
        float(np.linalg.norm(np.ptp(finite_source, axis=0)))
        if len(finite_source)
        else math.nan
    )
    source_area_tolerance = (
        config.near_degenerate_area_ratio * bbox_diagonal**2
        if math.isfinite(bbox_diagonal)
        else math.nan
    )
    valid = (
        np.isfinite(source_areas)
        & np.isfinite(deformed_areas)
        & (source_areas > source_area_tolerance)
    )
    jacobian = np.divide(
        deformed_areas,
        source_areas,
        out=np.full_like(source_areas, np.nan),
        where=valid,
    )
    valid_jacobian = jacobian[np.isfinite(jacobian) & (jacobian > 0)]
    result["surface_jacobian_status"] = "ok"
    result["surface_jacobian_source_area_tolerance"] = source_area_tolerance
    result["surface_jacobian_invalid_source_face_count"] = int(np.sum(~valid))
    result["surface_jacobian_nonpositive_deformed_face_count"] = int(
        np.sum(valid & (deformed_areas <= 0))
    )
    result.update(_distribution(valid_jacobian, "surface_jacobian"))
    if valid_jacobian.size:
        log_jacobian = np.log(valid_jacobian)
        collapsed = valid_jacobian < config.collapse_threshold
        expanded = valid_jacobian > config.expansion_threshold
        result["surface_jacobian_log_mean"] = float(np.mean(log_jacobian))
        result["surface_jacobian_log_std"] = float(np.std(log_jacobian))
        result["surface_jacobian_collapsed_face_count"] = int(np.sum(collapsed))
        result["surface_jacobian_collapsed_face_fraction"] = float(
            np.mean(collapsed)
        )
        result["surface_jacobian_expanded_face_count"] = int(np.sum(expanded))
        result["surface_jacobian_expanded_face_fraction"] = float(
            np.mean(expanded)
        )
    else:
        result["surface_jacobian_log_mean"] = math.nan
        result["surface_jacobian_log_std"] = math.nan
        result["surface_jacobian_collapsed_face_count"] = 0
        result["surface_jacobian_collapsed_face_fraction"] = math.nan
        result["surface_jacobian_expanded_face_count"] = 0
        result["surface_jacobian_expanded_face_fraction"] = math.nan
    return result


def target_screening_decision(
    metrics: Mapping[str, Any],
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
) -> dict[str, str]:
    """Apply prespecified target-only pass/fail/review criteria."""

    fail: list[str] = []
    review: list[str] = []
    if not metrics.get("finite_coordinates", False):
        fail.append("nonfinite_coordinates")
    for field, reason in (
        ("invalid_face_index_count", "invalid_face_indices"),
        ("repeated_vertex_face_count", "faces_with_repeated_vertex_indices"),
        ("duplicate_face_count", "duplicate_faces"),
        ("degenerate_face_count", "zero_area_faces"),
        ("boundary_edge_count", "boundary_edges_on_expected_closed_surface"),
        ("non_manifold_edge_count", "non_manifold_edges"),
        ("orientation_inconsistent_edge_count", "inconsistent_face_orientation"),
    ):
        if float(metrics.get(field, 0) or 0) > 0:
            fail.append(reason)
    if int(metrics.get("connected_component_count", 0) or 0) != 1:
        fail.append("surface_is_not_one_edge_connected_component")
    if not metrics.get("watertight", False):
        fail.append("not_watertight")
    if metrics.get("self_intersection_computation_status") != "ok":
        fail.append("self_intersection_test_unavailable")
    elif bool(metrics.get("self_intersection_present")):
        fail.append("nonadjacent_triangle_self_intersection")

    if float(metrics.get("duplicate_vertex_count", 0) or 0) > 0:
        review.append("exact_duplicate_vertex_coordinates")
    if float(metrics.get("near_degenerate_face_count", 0) or 0) > 0:
        review.append("near_degenerate_faces")
    quality_min = float(metrics.get("triangle_quality_min", math.nan))
    if math.isfinite(quality_min) and quality_min < config.triangle_quality_review_threshold:
        review.append("extremely_low_triangle_quality")

    fail = list(dict.fromkeys(fail))
    review = list(dict.fromkeys(review))
    status = "fail" if fail else ("review" if review else "pass")
    return {
        "target_qc_status": status,
        "target_qc_fail_reasons": ";".join(fail),
        "target_qc_review_reasons": ";".join(review),
    }


def screen_target_meshes(
    targets: Mapping[str, Path],
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
) -> list[dict[str, Any]]:
    """Evaluate targets without using any training outcome."""

    rows: list[dict[str, Any]] = []
    for subject_id, path in targets.items():
        row: dict[str, Any] = {
            "mesh_role": "target",
            "subject_id": str(subject_id),
            "target_vtk": str(path),
            "target_qc_status": "fail",
        }
        try:
            _, points, faces = load_triangle_mesh(path)
            metrics = mesh_quality_metrics(points, faces, config)
            row.update(metrics)
            row.update(target_screening_decision(metrics, config))
        except Exception as exc:
            row["target_qc_fail_reasons"] = "unreadable_or_qc_exception"
            row["target_qc_review_reasons"] = ""
            row["target_qc_error"] = repr(exc)
        rows.append(row)
    return rows


def target_qc_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        status: sum(row.get("target_qc_status") == status for row in rows)
        for status in ("pass", "review", "fail")
    }
    return {
        "qc_definition_version": QC_DEFINITION_VERSION,
        "intended_target_count": len(rows),
        "intended_subjects": [str(row.get("subject_id")) for row in rows],
        "status_counts": counts,
        "all_pass": counts["pass"] == len(rows),
        "training_gate": (
            "pass"
            if counts["fail"] == 0 and counts["review"] == 0
            else "stop_before_training"
        ),
        "replacement_rule": REPLACEMENT_RULE,
        "failed_subjects": [
            str(row.get("subject_id"))
            for row in rows
            if row.get("target_qc_status") == "fail"
        ],
        "review_subjects": [
            str(row.get("subject_id"))
            for row in rows
            if row.get("target_qc_status") == "review"
        ],
    }


def target_qc_protocol(
    subject_ids: Iterable[str],
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
) -> dict[str, Any]:
    return {
        "qc_definition_version": QC_DEFINITION_VERSION,
        "screening_population": "intended targets selected before training",
        "outcome_independence": (
            "No training artifact, endpoint mismatch, or method outcome is read "
            "by the target screening decision."
        ),
        "intended_subjects": list(subject_ids),
        "config": asdict(config),
        "hard_fail_criteria": [
            "non-finite coordinates or unreadable geometry",
            "invalid/repeated face indices, duplicate faces, or zero-area faces",
            "any boundary edge on the expected closed surface",
            "any non-manifold edge or inconsistent incident-face orientation",
            "not exactly one edge-connected component or not watertight",
            "self-intersection test unavailable or any intersecting non-adjacent triangle pair",
        ],
        "manual_review_criteria": [
            "exact duplicate vertex coordinates",
            "positive-area face with area/bbox_diagonal^2 <= 1e-12",
            "minimum q = 4*sqrt(3)*area/sum(edge_length^2) below 1e-4",
        ],
        "training_gate": (
            "Hard failures always stop before training. Review findings stop by "
            "default and require an explicit audited override."
        ),
        "replacement_rule": REPLACEMENT_RULE,
    }


def propose_target_replacements(
    available: Mapping[str, Path],
    selected: Mapping[str, Path],
    selected_qc_rows: list[dict[str, Any]],
    config: MeshQCConfig = DEFAULT_QC_CONFIG,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen deterministic replacement candidates, but never substitute them.

    Candidates are ordered by distance from the failed subject's rank in the
    sorted full cohort; ties go to the numerically smaller subject ID.  Only a
    candidate receiving an unqualified ``pass`` is proposed.
    """

    cohort = sorted(available)
    cohort_rank = {subject_id: rank for rank, subject_id in enumerate(cohort)}
    selected_ids = set(selected)
    failures = [
        str(row["subject_id"])
        for row in selected_qc_rows
        if row.get("target_qc_status") == "fail"
    ]
    proposals: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    reserved_proposals: set[str] = set()
    for failed_subject in failures:
        failed_rank = cohort_rank[failed_subject]
        candidates = [
            subject_id
            for subject_id in cohort
            if subject_id not in selected_ids and subject_id not in reserved_proposals
        ]
        candidates.sort(
            key=lambda subject_id: (
                abs(cohort_rank[subject_id] - failed_rank),
                int(subject_id) if subject_id.isdigit() else subject_id,
            )
        )
        proposed_subject = ""
        screened_count = 0
        for candidate in candidates:
            screened_count += 1
            screened = screen_target_meshes(
                {candidate: available[candidate]}, config
            )[0]
            screened["proposal_for_failed_subject"] = failed_subject
            screened["candidate_sorted_cohort_rank_zero_based"] = cohort_rank[
                candidate
            ]
            screened["failed_sorted_cohort_rank_zero_based"] = failed_rank
            screened["absolute_rank_distance"] = abs(
                cohort_rank[candidate] - failed_rank
            )
            candidate_rows.append(screened)
            if screened["target_qc_status"] == "pass":
                proposed_subject = candidate
                reserved_proposals.add(candidate)
                break
        proposals.append(
            {
                "failed_subject_id": failed_subject,
                "failed_sorted_cohort_rank_zero_based": failed_rank,
                "proposed_replacement_subject_id": proposed_subject,
                "proposed_replacement_vtk": (
                    str(available[proposed_subject]) if proposed_subject else ""
                ),
                "proposed_replacement_sorted_cohort_rank_zero_based": (
                    cohort_rank[proposed_subject]
                    if proposed_subject
                    else None
                ),
                "absolute_rank_distance": (
                    abs(cohort_rank[proposed_subject] - failed_rank)
                    if proposed_subject
                    else None
                ),
                "candidates_screened": screened_count,
                "proposal_only_no_replacement_performed": True,
                "replacement_rule": REPLACEMENT_RULE,
            }
        )
    return proposals, candidate_rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _add_excel_sheet(workbook: Any, title: str, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet(title)
    if not rows:
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    sheet.append(headers)
    for row in rows:
        sheet.append([_json_ready(row.get(header)) for header in headers])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.alignment = Alignment(wrap_text=True)
    for index, header in enumerate(headers, 1):
        width = max(
            len(header),
            *(
                len(str(sheet.cell(row=row, column=index).value or ""))
                for row in range(2, min(sheet.max_row, 200) + 1)
            ),
        )
        sheet.column_dimensions[get_column_letter(index)].width = min(width + 2, 45)


def write_target_qc_outputs(
    output_root: Path,
    rows: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    *,
    workbook_name: str = "target_qc_report.xlsx",
    replacement_proposals: list[dict[str, Any]] | None = None,
    replacement_candidate_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write auditable target screening CSV, JSON, protocol, summary, and Excel."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary = target_qc_summary(rows)
    replacement_proposals = replacement_proposals or []
    replacement_candidate_rows = replacement_candidate_rows or []
    summary["replacement_proposals"] = replacement_proposals
    paths = {
        "csv": output_root / "target_qc.csv",
        "json": output_root / "target_qc.json",
        "protocol": output_root / "target_qc_protocol.json",
        "summary": output_root / "target_qc_summary.json",
        "replacement_proposals": (
            output_root / "target_qc_replacement_proposals.json"
        ),
        "replacement_candidates_csv": (
            output_root / "target_qc_replacement_candidates.csv"
        ),
        "replacement_candidates_json": (
            output_root / "target_qc_replacement_candidates.json"
        ),
        "excel": output_root / workbook_name,
    }
    _write_csv(paths["csv"], rows)
    paths["json"].write_text(
        json.dumps(_json_ready(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    paths["protocol"].write_text(
        json.dumps(_json_ready(dict(protocol)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["summary"].write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["replacement_proposals"].write_text(
        json.dumps(
            _json_ready(replacement_proposals), indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    _write_csv(paths["replacement_candidates_csv"], replacement_candidate_rows)
    paths["replacement_candidates_json"].write_text(
        json.dumps(
            _json_ready(replacement_candidate_rows), indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    if Workbook is None:
        raise RuntimeError("openpyxl is required for target QC Excel output")
    workbook = Workbook()
    workbook.remove(workbook.active)
    _add_excel_sheet(workbook, "Target QC", rows)
    _add_excel_sheet(
        workbook,
        "QC Summary",
        [{"parameter": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}
         for key, value in summary.items()],
    )
    _add_excel_sheet(
        workbook,
        "QC Protocol",
        [{"parameter": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}
         for key, value in protocol.items()],
    )
    _add_excel_sheet(
        workbook, "Replacement Proposals", replacement_proposals
    )
    _add_excel_sheet(
        workbook, "Replacement Candidates", replacement_candidate_rows
    )
    workbook.save(paths["excel"])
    return paths


def discover_target_paths(target_root: Path) -> dict[str, Path]:
    """Discover ``<id>_sm0.2/<id>_sm0.2_deformed.vtk`` target meshes."""

    pattern = re.compile(r"^(?P<id>\d+)_sm0\.2$")
    found: dict[str, Path] = {}
    for folder in sorted(target_root.iterdir()):
        match = pattern.match(folder.name) if folder.is_dir() else None
        if not match:
            continue
        subject_id = match.group("id")
        target = folder / f"{folder.name}_deformed.vtk"
        if target.is_file():
            found[subject_id] = target
    return found


def select_target_paths(
    available: Mapping[str, Path],
    explicit: list[str] | None,
    count: int,
    strategy: str,
    seed: int,
) -> dict[str, Path]:
    """Mirror the study's deterministic target selection without training."""

    if explicit:
        missing = [subject for subject in explicit if subject not in available]
        if missing:
            raise FileNotFoundError(f"Targets missing for subjects: {missing}")
        return {subject: available[subject] for subject in explicit}
    ids = sorted(available)
    if count > len(ids):
        raise ValueError(f"Requested {count} subjects but only {len(ids)} are available")
    if strategy == "first":
        selected = ids[:count]
    elif strategy == "systematic":
        indices = np.linspace(0, len(ids) - 1, count).round().astype(int)
        selected = [ids[index] for index in indices]
    elif strategy == "random":
        selected = sorted(
            np.random.default_rng(seed).choice(ids, size=count, replace=False).tolist()
        )
    else:
        raise ValueError(f"Unsupported target selection strategy: {strategy}")
    return {subject: available[subject] for subject in selected}
