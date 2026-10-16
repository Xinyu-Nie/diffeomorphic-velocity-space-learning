"""Legacy endpoint attachments, retaining each original normalization."""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn

import contextlib
from scipy.spatial import cKDTree
from .kernels import wendland_phi

def support_varifold_inner(c1,n1,a1,c2,n2,a2,radius,kind,tile_i,tile_j,amp):
    if amp:
        raise ValueError("Stable support backend requires full precision.")
    if kind not in ("c2","c4"):
        raise ValueError("Only compact-support Wendland kernels are supported.")
    left=c1.detach().double().cpu().numpy()
    right=c2.detach().double().cpu().numpy()
    padding=1.+8.*torch.finfo(c1.dtype).eps
    pairs=cKDTree(left).sparse_distance_matrix(cKDTree(right),float(radius)*padding,output_type="coo_matrix")
    ii=torch.as_tensor(pairs.row.astype(np.int64),device=c1.device)
    jj=torch.as_tensor(pairs.col.astype(np.int64),device=c1.device)
    # A differentiable zero also handles completely separated surfaces.
    total=(c1.sum()+c2.sum()).double()*0.
    for begin in range(0,len(ii),262144):
        i,j=ii[begin:begin+262144],jj[begin:begin+262144]
        distances=torch.linalg.vector_norm(c1[i]-c2[j],dim=-1)
        dot=(n1[i]*n2[j]).sum(-1).clamp(-1.,1.)
        values=a1[i]*a2[j]*wendland_phi(distances/float(radius),kind)*dot.square()
        total=total+values.double().sum()
    return total

def compute_face_attributes(
    points: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triangles = points[faces]
    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]
    cross = torch.cross(v1, v2, dim=1)
    double_area = cross.norm(dim=1)
    areas = 0.5 * double_area
    normals = cross / (double_area.unsqueeze(1) + 1.0e-8)
    centers = triangles.mean(dim=1)
    return centers, normals, areas


def deterministic_face_subset(
    faces: torch.Tensor,
    maximum: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if maximum is None or maximum <= 0 or faces.shape[0] <= maximum:
        return faces, None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices_cpu = torch.randperm(faces.shape[0], generator=generator)[: int(maximum)]
    indices = indices_cpu.to(device=faces.device)
    return faces[indices], indices


def varifold_inner_tiled(
    c1: torch.Tensor,
    n1: torch.Tensor,
    a1: torch.Tensor,
    c2: torch.Tensor,
    n2: torch.Tensor,
    a2: torch.Tensor,
    radius: float,
    kind: str,
    tile_i: int,
    tile_j: int,
    amp: bool,
) -> torch.Tensor:
    accumulator = torch.zeros((), dtype=torch.float64, device=c1.device)
    use_amp = bool(amp and c1.is_cuda and c1.dtype == torch.float32)

    for i0 in range(0, c1.shape[0], int(tile_i)):
        i1 = min(i0 + int(tile_i), c1.shape[0])
        ci = c1[i0:i1]
        ni = n1[i0:i1]
        ai = a1[i0:i1]
        for j0 in range(0, c2.shape[0], int(tile_j)):
            j1 = min(j0 + int(tile_j), c2.shape[0])
            cj = c2[j0:j1]
            nj = n2[j0:j1]
            aj = a2[j0:j1]
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                distances = torch.cdist(ci, cj)
                q = distances / float(radius)
                positional = wendland_phi(q, kind=kind)
                normal_dot = (ni @ nj.t()).clamp(-1.0, 1.0)
                directional = normal_dot.pow(2)
                tile = (
                    ai.unsqueeze(1)
                    * aj.unsqueeze(0)
                    * positional
                    * directional
                )
            accumulator = accumulator + tile.to(torch.float64).sum()
    return accumulator


class VarifoldLoss(nn.Module):
    """Orientation-free surface varifold loss with a cached target self-term."""

    def __init__(
        self,
        source_faces: torch.Tensor,
        target_points: torch.Tensor,
        target_faces: torch.Tensor,
        radius: float,
        kind: str,
        normalize_areas: bool,
        tile_i: int,
        tile_j: int,
        amp: bool,
        detach_geometry: bool,
        max_faces: Optional[int],
        seed: int,
        backend: str = "tiled",
    ) -> None:
        super().__init__()
        if backend not in ("tiled", "stable"):
            raise ValueError("Varifold backend must be tiled or stable.")
        if backend == "stable" and amp:
            raise ValueError("Stable varifold requires full precision; omit --var_amp.")
        self.inner_product = support_varifold_inner if backend == "stable" else varifold_inner_tiled
        if radius <= 0.0:
            raise ValueError("Varifold radius must be positive.")
        source_faces, source_idx = deterministic_face_subset(
            source_faces, max_faces, seed=seed
        )
        target_faces, target_idx = deterministic_face_subset(
            target_faces, max_faces, seed=seed + 1
        )
        self.register_buffer("source_faces", source_faces)
        self.register_buffer("target_points", target_points.detach())
        self.register_buffer("target_faces", target_faces)
        self.radius = float(radius)
        self.kind = str(kind)
        self.normalize_areas = bool(normalize_areas)
        self.tile_i = int(tile_i)
        self.tile_j = int(tile_j)
        self.amp = bool(amp)
        self.detach_geometry = bool(detach_geometry)
        self.source_subset_size = int(source_faces.shape[0])
        self.target_subset_size = int(target_faces.shape[0])
        self.source_subset_used = source_idx is not None
        self.target_subset_used = target_idx is not None

        with torch.no_grad():
            c2, n2, a2 = compute_face_attributes(self.target_points, self.target_faces)
            if self.normalize_areas:
                a2 = a2 / (a2.sum() + 1.0e-12)
            target_self = self.inner_product(
                c2,
                n2,
                a2,
                c2,
                n2,
                a2,
                radius=self.radius,
                kind=self.kind,
                tile_i=self.tile_i,
                tile_j=self.tile_j,
                amp=self.amp,
            )
        self.register_buffer("target_centers", c2.detach())
        self.register_buffer("target_normals", n2.detach())
        self.register_buffer("target_areas", a2.detach())
        self.register_buffer("target_self", target_self.detach())

    def forward(self, source_points: torch.Tensor) -> torch.Tensor:
        c1, n1, a1 = compute_face_attributes(source_points, self.source_faces)
        if self.normalize_areas:
            a1 = a1 / (a1.sum() + 1.0e-12)
        if self.detach_geometry:
            n1 = n1.detach()
            a1 = a1.detach()

        source_self = self.inner_product(
            c1,
            n1,
            a1,
            c1,
            n1,
            a1,
            radius=self.radius,
            kind=self.kind,
            tile_i=self.tile_i,
            tile_j=self.tile_j,
            amp=self.amp,
        )
        cross = self.inner_product(
            c1,
            n1,
            a1,
            self.target_centers,
            self.target_normals,
            self.target_areas,
            radius=self.radius,
            kind=self.kind,
            tile_i=self.tile_i,
            tile_j=self.tile_j,
            amp=self.amp,
        )
        return (source_self + self.target_self - 2.0 * cross).to(source_points.dtype)

def directed_squared_chamfer(a,b,chunk=512):
    if not len(a) or not len(b): raise ValueError('Chamfer requires nonempty point sets')
    indices=[]
    with torch.no_grad():
        for start in range(0,len(a),chunk):
            d=torch.cdist(a[start:start+chunk],b,compute_mode='donot_use_mm_for_euclid_dist')
            indices.append(d.argmin(dim=1))
    nearest=b[torch.cat(indices)]
    return (a-nearest).square().sum(dim=1).mean()


def symmetric_squared_chamfer(a,b):
    return directed_squared_chamfer(a,b)+directed_squared_chamfer(b,a)

def sample_surface_barycentric(
    points: torch.Tensor,
    faces: torch.Tensor,
    sample_count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic area-weighted face IDs and barycentric weights."""
    if sample_count <= 0:
        raise ValueError("Surface sample counts must be positive.")
    point_array = points.detach().cpu().numpy()
    face_array = faces.detach().cpu().numpy()
    triangles = point_array[face_array]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("Cannot area-sample a mesh with zero total triangle area.")
    generator = np.random.default_rng(int(seed))
    face_ids = generator.choice(
        face_array.shape[0],
        size=int(sample_count),
        replace=True,
        p=areas / total_area,
    )
    random_1 = generator.random(int(sample_count))
    random_2 = generator.random(int(sample_count))
    root = np.sqrt(random_1)
    barycentric = np.stack(
        [1.0 - root, root * (1.0 - random_2), root * random_2], axis=1
    )
    return (
        torch.as_tensor(face_ids, dtype=torch.long, device=points.device),
        torch.as_tensor(barycentric, dtype=points.dtype, device=points.device),
    )


def realize_surface_samples(
    points: torch.Tensor,
    faces: torch.Tensor,
    face_ids: torch.Tensor,
    barycentric: torch.Tensor,
) -> torch.Tensor:
    triangles = points[faces[face_ids]]
    return (triangles * barycentric.unsqueeze(-1)).sum(dim=1)


def directed_chamfer_squared(
    queries: torch.Tensor,
    references: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    minima = []
    chunk = max(1, int(chunk))
    for start in range(0, queries.shape[0], chunk):
        distances = torch.cdist(queries[start : start + chunk], references)
        minima.append(distances.square().min(dim=1).values)
    return torch.cat(minima).mean()


class AreaSampledSurfaceChamfer(nn.Module):
    """Symmetric squared Chamfer on deterministic area-weighted samples."""

    def __init__(
        self,
        source_points: torch.Tensor,
        source_faces: torch.Tensor,
        target_points: torch.Tensor,
        target_faces: torch.Tensor,
        source_sample_count: int,
        target_sample_count: int,
        chunk: int,
        seed: int,
    ) -> None:
        super().__init__()
        source_face_ids, source_barycentric = sample_surface_barycentric(
            source_points, source_faces, source_sample_count, seed
        )
        target_face_ids, target_barycentric = sample_surface_barycentric(
            target_points, target_faces, target_sample_count, seed + 1
        )
        target_samples = realize_surface_samples(
            target_points, target_faces, target_face_ids, target_barycentric
        )
        self.register_buffer("source_faces", source_faces.detach())
        self.register_buffer("source_face_ids", source_face_ids)
        self.register_buffer("source_barycentric", source_barycentric)
        self.register_buffer("target_samples", target_samples.detach())
        self.chunk = max(1, int(chunk))

    def forward(self, deformed_source: torch.Tensor) -> torch.Tensor:
        source_samples = realize_surface_samples(
            deformed_source,
            self.source_faces,
            self.source_face_ids,
            self.source_barycentric,
        )
        source_to_target = directed_chamfer_squared(
            source_samples, self.target_samples, self.chunk
        )
        target_to_source = directed_chamfer_squared(
            self.target_samples, source_samples, self.chunk
        )
        return 0.5 * (source_to_target + target_to_source)

