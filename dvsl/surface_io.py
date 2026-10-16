"""Surface/basis I/O and legacy endpoint diagnostics."""
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

def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Nonfinite values detected in {name}.")


@dataclass
class MeshData:
    path: str
    mesh: pv.PolyData
    points: torch.Tensor
    faces: torch.Tensor


def load_mesh(path: str, device: torch.device, dtype: torch.dtype) -> MeshData:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Mesh not found: {resolved}")
    mesh = pv.read(resolved).triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh has no usable triangular surface: {resolved}")
    face_array = np.asarray(mesh.faces, dtype=np.int64)
    if face_array.size % 4 != 0:
        raise ValueError(f"Triangulated face array is malformed: {resolved}")
    faces_np = face_array.reshape(-1, 4)
    if not np.all(faces_np[:, 0] == 3):
        raise ValueError(f"Mesh is not purely triangular after triangulation: {resolved}")
    points = torch.as_tensor(
        np.asarray(mesh.points, dtype=np.float64).copy(), dtype=dtype, device=device
    )
    faces = torch.as_tensor(
        faces_np[:, 1:].copy(), dtype=torch.long, device=device
    )
    finite_or_raise("mesh points", points)
    return MeshData(resolved, mesh, points, faces)


def load_mercer_basis(
    path: str,
    basis_dim: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Mercer basis not found: {resolved}")
    data = np.load(resolved, allow_pickle=True)
    if "U" not in data or "lam" not in data:
        raise KeyError("Mercer basis must contain arrays 'U' and 'lam'.")
    U_np = np.asarray(data["U"])
    lam_np = np.asarray(data["lam"]).reshape(-1)
    if U_np.ndim != 2 or U_np.shape[1] != lam_np.shape[0]:
        raise ValueError(
            f"Incompatible Mercer shapes: U={U_np.shape}, lam={lam_np.shape}."
        )

    meta: Dict = {}
    if "meta" in data:
        meta_raw = data["meta"]
        if isinstance(meta_raw, np.ndarray) and meta_raw.shape == ():
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, (str, bytes)):
            meta = json.loads(meta_raw)
        elif isinstance(meta_raw, dict):
            meta = dict(meta_raw)

    order_np = np.argsort(lam_np)[::-1]
    if basis_dim is None or basis_dim <= 0:
        selected = U_np.shape[1]
    else:
        selected = min(int(basis_dim), U_np.shape[1])
    order_np = order_np[:selected].copy()
    U = torch.as_tensor(U_np[:, order_np].copy(), dtype=dtype, device=device)
    lam = torch.as_tensor(lam_np[order_np].copy(), dtype=dtype, device=device)
    finite_or_raise("Mercer eigenvectors", U)
    finite_or_raise("Mercer eigenvalues", lam)
    if bool(torch.any(lam <= 0.0)):
        smallest = float(lam.min().detach().cpu())
        raise ValueError(
            "Selected Mercer modes must have positive eigenvalues; "
            f"smallest selected value is {smallest:.6e}. Reduce --basis_dim."
        )
    return U, lam, meta


def smoothness_loss(displacement: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0
    )
    return (displacement[edges[:, 0]] - displacement[edges[:, 1]]).pow(2).mean()


@torch.no_grad()
def directed_nearest_distances(
    source: torch.Tensor, target: torch.Tensor, chunk: int
) -> torch.Tensor:
    blocks: List[torch.Tensor] = []
    for start in range(0, source.shape[0], int(chunk)):
        blocks.append(
            torch.cdist(source[start : start + int(chunk)], target).min(dim=1).values
        )
    return torch.cat(blocks)


def distribution(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {
            f"{prefix}_min": math.nan,
            f"{prefix}_p01": math.nan,
            f"{prefix}_p05": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_p99": math.nan,
            f"{prefix}_max": math.nan,
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor(
            [0.01, 0.05, 0.50, 0.95, 0.99], dtype=finite.dtype, device=finite.device
        ),
    )
    return {
        f"{prefix}_min": float(finite.min().cpu()),
        f"{prefix}_p01": float(quantiles[0].cpu()),
        f"{prefix}_p05": float(quantiles[1].cpu()),
        f"{prefix}_median": float(quantiles[2].cpu()),
        f"{prefix}_p95": float(quantiles[3].cpu()),
        f"{prefix}_p99": float(quantiles[4].cpu()),
        f"{prefix}_max": float(finite.max().cpu()),
    }


@torch.no_grad()
def endpoint_metrics(
    endpoint: torch.Tensor,
    target: torch.Tensor,
    pointwise_available: bool,
    chunk: int,
) -> Dict[str, float]:
    forward = directed_nearest_distances(endpoint, target, chunk)
    backward = directed_nearest_distances(target, endpoint, chunk)
    result = {
        "chamfer_squared": float((forward.pow(2).mean() + backward.pow(2).mean()).cpu()),
        "symmetric_hausdorff": float(torch.maximum(forward.max(), backward.max()).cpu()),
        "endpoint_to_target_mean": float(forward.mean().cpu()),
        "target_to_endpoint_mean": float(backward.mean().cpu()),
    }
    if pointwise_available:
        squared = (endpoint - target).pow(2)
        result["pointwise_mse"] = float(squared.mean().cpu())
        result["pointwise_rmse"] = float(squared.mean().sqrt().cpu())
        result["pointwise_vertex_l2_mean"] = float(
            torch.linalg.norm(endpoint - target, dim=1).mean().cpu()
        )
    return result

