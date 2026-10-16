#!/usr/bin/env python3
"""Geodesic Eulerian Shooting (GES) surface matching.

Optimizes source-supported initial momentum density m0, converts to the discrete
covector p0 = M m0 using vertex areas, and integrates Hamiltonian dynamics with
the original fixed-step RK4/adjoint implementation. Single or normalized mixed
Wendland C2/C4 kernels are supported. Metric and numerical defaults are retained.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from .kernels import wendland_phi, wendland_phi_dphi, build_radius_graph
from .metrics import VarifoldLoss
from .matching_cli import validate_comparator_arguments

try:
    from torch_cluster import radius_graph
except Exception as exc:  # pragma: no cover - depends on local PyTorch build
    radius_graph = None
    _TORCH_CLUSTER_IMPORT_ERROR = exc
else:
    _TORCH_CLUSTER_IMPORT_ERROR = None

try:
    from torchdiffeq import odeint as odeint_direct
    from torchdiffeq import odeint_adjoint
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torchdiffeq is required. Install it with: pip install torchdiffeq"
    ) from exc


# =============================================================================
# Reproducibility and basic utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Nonfinite values detected in {name}.")


# =============================================================================
# Mesh I/O and source quadrature masses
# =============================================================================


@dataclass
class MeshData:
    path: str
    mesh: pv.PolyData
    points: torch.Tensor
    faces: torch.Tensor


def load_mesh(path: str, device: torch.device, dtype: torch.dtype) -> MeshData:
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mesh = pv.read(path).triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh is empty: {path}")

    points_np = np.asarray(mesh.points, dtype=np.float64)
    # PyVista may expose a read-only view; avoid aliasing it in torch.as_tensor.
    faces_np = np.array(mesh.faces, dtype=np.int64, copy=True)
    if faces_np.size % 4 != 0:
        raise ValueError(f"Triangulated mesh has an unexpected face layout: {path}")
    faces_np = faces_np.reshape(-1, 4)
    if not np.all(faces_np[:, 0] == 3):
        raise ValueError(f"Non-triangular cells remain after triangulation: {path}")
    faces_np = faces_np[:, 1:]

    if not np.isfinite(points_np).all():
        raise ValueError(f"Mesh contains nonfinite coordinates: {path}")

    points = torch.as_tensor(points_np, dtype=dtype, device=device)
    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
    return MeshData(path=path, mesh=mesh, points=points, faces=faces)


def compute_vertex_areas(mesh: pv.PolyData, dtype: torch.dtype) -> torch.Tensor:
    """Barycentric vertex areas, identical in convention to the DSMN script."""
    tri = mesh.triangulate()
    pts = np.asarray(tri.points, dtype=np.float64)
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    areas = np.zeros((pts.shape[0],), dtype=np.float64)
    a = pts[faces[:, 0]]
    b = pts[faces[:, 1]]
    c = pts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    for k in range(3):
        np.add.at(areas, faces[:, k], face_areas / 3.0)
    areas[areas <= 0.0] = np.finfo(np.float64).eps
    return torch.as_tensor(areas, dtype=dtype)


def load_or_compute_masses(
    source: MeshData,
    mass_npy: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mass_npy is None:
        masses = compute_vertex_areas(source.mesh, dtype=dtype)
    else:
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(mass_npy)))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if arr.shape[0] != source.points.shape[0]:
            raise ValueError(
                f"Mass length mismatch: {arr.shape[0]} versus "
                f"{source.points.shape[0]} source vertices."
            )
        if not np.isfinite(arr).all() or np.any(arr <= 0.0):
            raise ValueError("Mass array must be finite and strictly positive.")
        masses = torch.as_tensor(arr, dtype=dtype)
    return masses.to(device=device, dtype=dtype)


# =============================================================================
# Wendland RKHS and DSMN Hamiltonian dynamics
# =============================================================================










def resolve_shooting_kernel(
    radius: float,
    shooting_radii: Optional[Sequence[float]],
    shooting_weights: Optional[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    radii = [float(radius)] if shooting_radii is None else [float(r) for r in shooting_radii]
    if len(radii) == 0 or any(r <= 0.0 for r in radii):
        raise ValueError("All shooting radii must be positive.")

    if shooting_weights is None:
        weights = [1.0 / len(radii)] * len(radii)
    else:
        weights = [float(w) for w in shooting_weights]
        if len(weights) != len(radii):
            raise ValueError(
                "--shooting_weights must have the same length as --shooting_radii."
            )
        if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
            raise ValueError("Shooting weights must be nonnegative with positive sum.")
        total = sum(weights)
        weights = [w / total for w in weights]
    return radii, weights


def wendland_kernel_sparse(
    x: torch.Tensor,
    p: torch.Tensor,
    radius: float,
    kind: str,
    max_num_neighbors: int,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = build_radius_graph(x, radius, max_num_neighbors)
    if edges.numel() == 0:
        return torch.zeros_like(x), torch.zeros_like(p)

    i, j = edges[0], edges[1]
    use_amp = bool(amp and x.is_cuda and x.dtype == torch.float32)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )
    with amp_ctx:
        diff = x[i] - x[j]
        r = diff.norm(dim=-1)
        q = r / float(radius)
        phi, dphi = wendland_phi_dphi(q, kind)

        dx_edge = phi.unsqueeze(-1) * p[j]
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_edge.to(dx.dtype))

        qeps = 1.0e-6
        safe_q = torch.clamp(q, min=qeps)
        coeff = dphi / (float(radius) * float(radius) * safe_q)
        if kind == "c2":
            coeff = torch.where(
                q < qeps,
                torch.full_like(coeff, -20.0 / (float(radius) ** 2)),
                coeff,
            )
        # For C4 the self-edge contributes zero because diff=0.
        grad_kernel = coeff.unsqueeze(-1) * diff
        pij = (p[i] * p[j]).sum(dim=-1, keepdim=True)
        dp_edge = -(pij * grad_kernel)
        dp = torch.zeros_like(p)
        dp.index_add_(0, i, dp_edge.to(dp.dtype))

    return dx, dp


class HamiltonianDynamics(nn.Module):
    """Same shared-control, normalized sum-kernel dynamics used by DSMN."""

    def __init__(
        self,
        radii: Sequence[float],
        weights: Sequence[float],
        kind: str,
        max_neighbors: int,
        amp: bool,
    ) -> None:
        super().__init__()
        self.radii = [float(r) for r in radii]
        self.weights = [float(w) for w in weights]
        self.kind = str(kind)
        self.max_neighbors = int(max_neighbors)
        self.amp = bool(amp)

    def forward(
        self,
        t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del t
        x, p = state
        dx = torch.zeros_like(x)
        dp = torch.zeros_like(p)
        for radius, weight in zip(self.radii, self.weights):
            dx_s, dp_s = wendland_kernel_sparse(
                x,
                p,
                radius=radius,
                kind=self.kind,
                max_num_neighbors=self.max_neighbors,
                amp=self.amp,
            )
            dx = dx + weight * dx_s
            dp = dp + weight * dp_s
        return dx, dp


def build_sparse_gram_wendland(
    points: torch.Tensor,
    radius: float,
    kind: str,
    max_neighbors: int,
) -> torch.Tensor:
    edges = build_radius_graph(points, radius, max_neighbors)
    if edges.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=points.device),
            torch.empty((0,), dtype=points.dtype, device=points.device),
            (points.shape[0], points.shape[0]),
            device=points.device,
            dtype=points.dtype,
        ).coalesce()
    row, col = edges[0], edges[1]
    q = (points[row] - points[col]).norm(dim=-1) / float(radius)
    values = wendland_phi(q, kind=kind)
    return torch.sparse_coo_tensor(
        edges,
        values,
        (points.shape[0], points.shape[0]),
        device=points.device,
        dtype=points.dtype,
    ).coalesce()


def build_sparse_gram_multi_wendland(
    points: torch.Tensor,
    radii: Sequence[float],
    weights: Sequence[float],
    kind: str,
    max_neighbors: int,
) -> torch.Tensor:
    all_indices: List[torch.Tensor] = []
    all_values: List[torch.Tensor] = []
    for radius, weight in zip(radii, weights):
        gram = build_sparse_gram_wendland(
            points, radius=radius, kind=kind, max_neighbors=max_neighbors
        ).coalesce()
        all_indices.append(gram.indices())
        all_values.append(float(weight) * gram.values())
    return torch.sparse_coo_tensor(
        torch.cat(all_indices, dim=1),
        torch.cat(all_values, dim=0),
        (points.shape[0], points.shape[0]),
        device=points.device,
        dtype=points.dtype,
    ).coalesce()


def rkhs_momentum_norm2(gram: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    gram = gram.coalesce()
    total = torch.zeros((), dtype=p0.dtype, device=p0.device)
    for dim in range(3):
        vec = p0[:, dim : dim + 1]
        gram_vec = torch.sparse.mm(gram, vec)
        total = total + torch.mm(vec.t(), gram_vec).squeeze()
    return total


def smoothness_loss(displacement: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    differences = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    return differences.pow(2).sum(dim=1).mean()


# =============================================================================
# Endpoint discrepancies
# =============================================================================


def pointwise_distance(
    deformed: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if deformed.shape != target.shape:
        raise ValueError(
            "Pointwise loss requires identical [N,3] source and target shapes; "
            f"received {tuple(deformed.shape)} and {tuple(target.shape)}."
        )
    # Same convention as DSMN Stage A: mean over vertices and coordinates.
    return (deformed - target).pow(2).mean()


def directed_squared_nearest_mean(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("Chamfer distance requires nonempty point sets.")
    total = torch.zeros((), dtype=a.dtype, device=a.device)
    count = 0
    for start in range(0, a.shape[0], int(chunk_size)):
        chunk = a[start : start + int(chunk_size)]
        distances = torch.cdist(chunk, b)
        nearest2 = distances.min(dim=1).values.pow(2)
        total = total + nearest2.sum()
        count += nearest2.numel()
    return total / max(1, count)


def chamfer_squared(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return directed_squared_nearest_mean(source, target, chunk_size) + directed_squared_nearest_mean(
        target, source, chunk_size
    )










# =============================================================================
# Initialization loader
# =============================================================================


def _extract_array_from_object(obj, preferred_key: Optional[str]) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        keys = []
        if preferred_key:
            keys.append(preferred_key)
        keys.extend(
            [
                "m0_density",
                "m0",
                "m0_pred",
                "density",
                "p0",
                "momentum",
                "arr_0",
            ]
        )
        for key in keys:
            if key in obj:
                value = obj[key]
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                return np.asarray(value)
        raise KeyError(
            "Could not locate a momentum array in the supplied dictionary. "
            f"Available keys: {list(obj.keys())}"
        )
    raise TypeError(f"Unsupported initialization object type: {type(obj)}")


def load_initial_array(path: str, preferred_key: Optional[str]) -> np.ndarray:
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        obj = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if preferred_key and preferred_key in archive.files:
            obj = archive[preferred_key]
        else:
            candidates = [
                "m0_density",
                "m0",
                "m0_pred",
                "density",
                "p0",
                "momentum",
                "arr_0",
            ]
            found = next((key for key in candidates if key in archive.files), None)
            if found is None:
                if len(archive.files) != 1:
                    raise KeyError(
                        "Could not infer the momentum key in NPZ. "
                        f"Available arrays: {archive.files}. Use --init_key."
                    )
                found = archive.files[0]
            obj = archive[found]
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
    else:
        raise ValueError("Initialization must be .npy, .npz, .pt, or .pth.")
    return np.asarray(_extract_array_from_object(obj, preferred_key), dtype=np.float64)


def prepare_initial_density(
    init_path: Optional[str],
    init_key: Optional[str],
    representation: str,
    init_scale: float,
    n_vertices: int,
    masses: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if init_path is None:
        return torch.zeros((n_vertices, 3), dtype=dtype, device=device)

    array = load_initial_array(init_path, init_key)
    array = np.squeeze(array)
    if array.shape == (3, n_vertices):
        array = array.T
    if array.shape != (n_vertices, 3):
        raise ValueError(
            f"Initial momentum must have shape ({n_vertices}, 3); got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Initial momentum contains nonfinite values.")

    tensor = torch.as_tensor(array, dtype=dtype, device=device) * float(init_scale)
    if representation == "density":
        density = tensor
    elif representation == "p0":
        density = tensor / masses.view(-1, 1).clamp_min(torch.finfo(dtype).eps)
    else:
        raise ValueError("--init_representation must be density or p0.")
    return density


# =============================================================================
# Direct m0 optimization
# =============================================================================


class GeodesicEulerianObjective(nn.Module):
    def __init__(
        self,
        source_points: torch.Tensor,
        source_faces: torch.Tensor,
        target_points: torch.Tensor,
        masses: torch.Tensor,
        dynamics: HamiltonianDynamics,
        n_steps: int,
        loss_name: str,
        chamfer_chunk: int,
        varifold_loss: Optional[VarifoldLoss],
        w_data: float,
        lambda_smooth: float,
        m0_penalty: str,
        m0_reg: float,
        m0_rkhs_scale: float,
        gram: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.register_buffer("source_points", source_points.detach())
        self.register_buffer("source_faces", source_faces)
        self.register_buffer("target_points", target_points.detach())
        self.register_buffer("masses", masses.detach())
        self.dynamics = dynamics
        self.n_steps = int(n_steps)
        self.loss_name = str(loss_name)
        self.chamfer_chunk = int(chamfer_chunk)
        self.varifold_loss = varifold_loss
        self.w_data = float(w_data)
        self.lambda_smooth = float(lambda_smooth)
        self.m0_penalty = str(m0_penalty)
        self.m0_reg = float(m0_reg)
        self.m0_rkhs_scale = float(m0_rkhs_scale)
        self.gram = gram
        self.register_buffer(
            "t_span",
            torch.tensor(
                [0.0, 1.0],
                dtype=source_points.dtype,
                device=source_points.device,
            ),
        )

    def shoot(self, m0_density: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.masses.view(-1, 1) * m0_density
        x_solution, p_solution = odeint_adjoint(
            self.dynamics,
            (self.source_points, p0),
            self.t_span,
            method="rk4",
            options={"step_size": 1.0 / float(self.n_steps)},
        )
        return x_solution[-1], p_solution[-1], p0

    def data_loss(self, endpoint: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "pointwise":
            return pointwise_distance(endpoint, self.target_points)
        if self.loss_name == "chamfer":
            return chamfer_squared(endpoint, self.target_points, self.chamfer_chunk)
        if self.loss_name == "varifold":
            if self.varifold_loss is None:
                raise RuntimeError("Varifold loss was not initialized.")
            return self.varifold_loss(endpoint)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def forward(self, m0_density: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        endpoint, p_endpoint, p0 = self.shoot(m0_density)
        finite_or_raise("deformed endpoint", endpoint)
        finite_or_raise("terminal momentum", p_endpoint)

        data = self.data_loss(endpoint)
        displacement = endpoint - self.source_points
        smooth = smoothness_loss(displacement, self.source_faces)

        if self.m0_penalty == "l2":
            momentum_raw = m0_density.pow(2).mean()
        elif self.m0_penalty == "rkhs":
            if self.gram is None:
                raise RuntimeError("RKHS momentum penalty requested without a Gram matrix.")
            momentum_raw = rkhs_momentum_norm2(self.gram, p0) / self.source_points.shape[0]
        else:
            raise ValueError("m0_penalty must be l2 or rkhs.")

        momentum_weighted = self.m0_reg * momentum_raw
        if self.m0_penalty == "rkhs":
            momentum_weighted = momentum_weighted * self.m0_rkhs_scale

        total = self.w_data * data + self.lambda_smooth * smooth + momentum_weighted
        components = {
            "total": total,
            "data": data,
            "smooth": smooth,
            "m0_raw": momentum_raw,
            "m0_weighted": momentum_weighted,
            "endpoint": endpoint,
            "p0": p0,
        }
        return total, components


@dataclass
class BestState:
    value: float = math.inf
    density: Optional[torch.Tensor] = None
    stage: str = ""
    evaluation: int = -1


@dataclass
class OptimizationTermination:
    status: str = "success"
    reason: Optional[str] = None
    evaluation: Optional[int] = None


class ObjectiveDivergenceError(FloatingPointError):
    """Raised when a finite objective exceeds the configured safety bound."""


def components_to_row(
    stage: str,
    evaluation: int,
    elapsed: float,
    components: Dict[str, torch.Tensor],
    grad_norm: float,
) -> Dict[str, float | int | str]:
    return {
        "stage": stage,
        "evaluation": int(evaluation),
        "elapsed_sec": float(elapsed),
        "total": float(components["total"].detach().cpu()),
        "data": float(components["data"].detach().cpu()),
        "smooth": float(components["smooth"].detach().cpu()),
        "m0_raw": float(components["m0_raw"].detach().cpu()),
        "m0_weighted": float(components["m0_weighted"].detach().cpu()),
        "grad_norm": float(grad_norm),
    }


def write_history(path: str, rows: List[Dict]) -> None:
    ensure_parent(path)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def optimize_initial_momentum(
    objective: GeodesicEulerianObjective,
    initial_density: torch.Tensor,
    adam_iters: int,
    adam_lr: float,
    adam_weight_decay: float,
    lbfgs_iters: int,
    lbfgs_lr: float,
    lbfgs_history_size: int,
    grad_clip: float,
    log_every: int,
    keep_best_on_failure: bool = False,
    divergence_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, List[Dict], BestState, OptimizationTermination]:
    parameter = nn.Parameter(initial_density.clone())
    history: List[Dict] = []
    best = BestState()
    start_time = time.perf_counter()
    global_eval = 0
    initial_value: Optional[float] = None

    def evaluate_and_backward(stage: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], float]:
        nonlocal global_eval, best, initial_value
        loss, components = objective(parameter)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite objective during {stage} evaluation {global_eval}."
            )
        value = float(loss.detach().cpu())
        if initial_value is None:
            initial_value = value
        elif divergence_factor is not None:
            bound = max(abs(initial_value), 1.0e-30) * float(divergence_factor)
            if abs(value) > bound:
                raise ObjectiveDivergenceError(
                    f"Objective magnitude {abs(value):.6e} exceeded the safety "
                    f"bound {bound:.6e} during {stage} evaluation {global_eval}."
                )
        loss.backward()
        if parameter.grad is None:
            raise RuntimeError("No gradient was produced for m0.")
        finite_or_raise("m0 gradient", parameter.grad)
        # Gradient clipping is useful for Adam, but clipping inside an L-BFGS
        # strong-Wolfe closure changes the gradient used by the line search.
        if grad_clip > 0.0 and stage != "lbfgs":
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_([parameter], grad_clip)
            grad_norm = float(grad_norm_tensor.detach().cpu())
        else:
            grad_norm = float(parameter.grad.norm().detach().cpu())

        if value < best.value:
            best.value = value
            best.density = parameter.detach().clone()
            best.stage = stage
            best.evaluation = global_eval

        row = components_to_row(
            stage=stage,
            evaluation=global_eval,
            elapsed=time.perf_counter() - start_time,
            components=components,
            grad_norm=grad_norm,
        )
        history.append(row)
        if global_eval % max(1, log_every) == 0:
            print(
                f"[{stage:7s} eval {global_eval:05d}] "
                f"total={row['total']:.6e} data={row['data']:.6e} "
                f"smooth={row['smooth']:.6e} m0={row['m0_weighted']:.6e} "
                f"|grad|={row['grad_norm']:.3e}"
            )
        global_eval += 1
        return loss, components, grad_norm

    termination = OptimizationTermination()

    def evaluate_safely(
        stage: str,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor], float]]:
        nonlocal termination
        try:
            return evaluate_and_backward(stage)
        except FloatingPointError as exc:
            if not keep_best_on_failure or best.density is None:
                raise
            status = (
                "stopped_divergence"
                if isinstance(exc, ObjectiveDivergenceError)
                else "stopped_nonfinite"
            )
            termination = OptimizationTermination(
                status=status,
                reason=str(exc),
                evaluation=global_eval,
            )
            print(
                f"[{status}] {exc} Saving the best finite state from "
                f"{best.stage} evaluation {best.evaluation}.",
                file=sys.stderr,
            )
            with torch.no_grad():
                parameter.copy_(best.density)
            return None

    # Evaluate the supplied/zero initialization once.
    parameter.grad = None
    evaluate_safely("initial")
    parameter.grad = None

    if adam_iters > 0:
        adam = torch.optim.AdamW(
            [parameter], lr=float(adam_lr), weight_decay=float(adam_weight_decay)
        )
        for _ in range(int(adam_iters)):
            adam.zero_grad(set_to_none=True)
            if evaluate_safely("adam") is None:
                break
            adam.step()

    if lbfgs_iters > 0 and termination.status == "success":
        lbfgs = torch.optim.LBFGS(
            [parameter],
            lr=float(lbfgs_lr),
            max_iter=int(lbfgs_iters),
            history_size=int(lbfgs_history_size),
            line_search_fn="strong_wolfe",
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            if termination.status != "success":
                return parameter.sum() * 0.0
            result = evaluate_safely("lbfgs")
            if result is None:
                return parameter.sum() * 0.0
            loss, _, _ = result
            return loss

        lbfgs.step(closure)

    # Explicitly evaluate the final accepted optimizer state. This matters in
    # Adam-only runs because the last Adam update occurs after its last closure.
    if termination.status == "success":
        parameter.grad = None
        evaluate_safely("final")
        parameter.grad = None

    if best.density is None:
        raise RuntimeError("Optimization produced no finite state.")
    return best.density, history, best, termination


# =============================================================================
# Evaluation and output
# =============================================================================


@torch.no_grad()
def directed_nearest_distances(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, a.shape[0], int(chunk_size)):
        values.append(
            torch.cdist(a[start : start + int(chunk_size)], b).min(dim=1).values
        )
    return torch.cat(values, dim=0)


@torch.no_grad()
def endpoint_metrics(
    endpoint: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
    pointwise_available: bool,
    varifold_loss: Optional[VarifoldLoss] = None,
) -> Dict[str, Optional[float]]:
    d_st = directed_nearest_distances(endpoint, target, chunk_size)
    d_ts = directed_nearest_distances(target, endpoint, chunk_size)
    metrics: Dict[str, Optional[float]] = {
        "chamfer_squared": float(
            (d_st.pow(2).mean() + d_ts.pow(2).mean()).detach().cpu()
        ),
        "assd": float((0.5 * (d_st.mean() + d_ts.mean())).detach().cpu()),
        "hd95": float(
            torch.maximum(
                torch.quantile(d_st.float(), 0.95),
                torch.quantile(d_ts.float(), 0.95),
            )
            .detach()
            .cpu()
        ),
        "hausdorff": float(torch.maximum(d_st.max(), d_ts.max()).detach().cpu()),
    }
    if pointwise_available:
        delta = endpoint - target
        metrics["pointwise_mse_coordinate"] = float(delta.pow(2).mean().detach().cpu())
        metrics["pointwise_rmse_vertex"] = float(
            delta.pow(2).sum(dim=1).mean().sqrt().detach().cpu()
        )
    else:
        metrics["pointwise_mse_coordinate"] = None
        metrics["pointwise_rmse_vertex"] = None
    metrics["varifold_distance"] = (
        float(varifold_loss(endpoint).detach().cpu())
        if varifold_loss is not None
        else None
    )
    return metrics


@torch.no_grad()
def final_shoot(
    dynamics: HamiltonianDynamics,
    source_points: torch.Tensor,
    p0: torch.Tensor,
    n_steps: int,
    trajectory_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if trajectory_samples < 1:
        trajectory_samples = 1
    times = torch.linspace(
        0.0,
        1.0,
        int(trajectory_samples) + 1,
        dtype=source_points.dtype,
        device=source_points.device,
    )
    x_path, p_path = odeint_direct(
        dynamics,
        (source_points, p0),
        times,
        method="rk4",
        options={"step_size": 1.0 / float(n_steps)},
    )
    return x_path[-1], p_path[-1], x_path, p_path


def save_outputs(
    output_dir: str,
    source: MeshData,
    target: MeshData,
    endpoint: torch.Tensor,
    m0_density: torch.Tensor,
    p0: torch.Tensor,
    x_path: torch.Tensor,
    p_path: torch.Tensor,
    save_trajectory: bool,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    target_stem = Path(target.path).stem

    density_np = m0_density.detach().cpu().numpy().astype(np.float32)
    p0_np = p0.detach().cpu().numpy().astype(np.float32)
    endpoint_np = endpoint.detach().cpu().numpy().astype(np.float64)

    density_path = os.path.join(output_dir, "m0.npy")
    density_explicit_path = os.path.join(output_dir, "m0_density.npy")
    p0_path = os.path.join(output_dir, "p0_discrete.npy")
    np.save(density_path, density_np)
    np.save(density_explicit_path, density_np)
    np.save(p0_path, p0_np)

    # Keep the deformed VTK geometry-only. Momentum and displacement arrays can
    # become active scalars in ParaView and color the surface unintentionally;
    # the momentum arrays remain available in the separate NPY outputs above.
    output_mesh = pv.PolyData(
        endpoint_np,
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    )
    vtk_path = os.path.join(output_dir, f"{target_stem}_optimized_deformed.vtk")
    output_mesh.save(vtk_path)

    paths = {
        "deformed_vtk": os.path.abspath(vtk_path),
        "m0_density_npy": os.path.abspath(density_explicit_path),
        "m0_compatibility_npy": os.path.abspath(density_path),
        "p0_discrete_npy": os.path.abspath(p0_path),
    }

    if save_trajectory:
        x_path_file = os.path.join(output_dir, "x_path.npy")
        p_path_file = os.path.join(output_dir, "p_path.npy")
        np.save(x_path_file, x_path.detach().cpu().numpy().astype(np.float32))
        np.save(p_path_file, p_path.detach().cpu().numpy().astype(np.float32))
        paths["x_path_npy"] = os.path.abspath(x_path_file)
        paths["p_path_npy"] = os.path.abspath(p_path_file)

        frames_dir = os.path.join(output_dir, "trajectory_vtk")
        os.makedirs(frames_dir, exist_ok=True)
        for index, frame in enumerate(x_path.detach().cpu().numpy()):
            frame_mesh = pv.PolyData(
                frame.astype(np.float64),
                np.asarray(source.mesh.faces, dtype=np.int64).copy(),
            )
            frame_mesh.save(os.path.join(frames_dir, f"frame_{index:03d}.vtk"))
        paths["trajectory_vtk_dir"] = os.path.abspath(frames_dir)

    return paths


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Geodesic Eulerian Shooting (GES): optimize initial momentum for one "
            "source-target surface pair."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pairs",
        nargs=2,
        metavar=("SOURCE_VTK", "TARGET_VTK"),
        required=True,
        help="One source-target VTK pair; no OT file is required.",
    )
    parser.add_argument(
        "--metric", "--loss", dest="loss",
        choices=["pointwise", "chamfer", "varifold"],
        required=True,
        help="Endpoint discrepancy to optimize.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=2026)

    # Optional initialization from DSMN or another optimization.
    parser.add_argument(
        "--init_m0",
        type=str,
        default=None,
        help="Optional .npy/.npz/.pt/.pth initial momentum.",
    )
    parser.add_argument(
        "--init_representation",
        choices=["density", "p0"],
        default="density",
        help=(
            "Use density for a DSMN m0.npy; use p0 only when the supplied "
            "array has already been multiplied by source vertex areas."
        ),
    )
    parser.add_argument("--init_key", type=str, default=None)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument(
        "--mass_npy",
        type=str,
        default=None,
        help="Optional DSMN M.npy. Otherwise vertex areas are recomputed.",
    )

    # Hamiltonian shooting; defaults mirror the uploaded DSMN script.
    parser.add_argument("--radius", "--kernel_radius", type=float, default=0.2)
    parser.add_argument(
        "--shooting_radii", "--kernel_radii",
        nargs="+",
        type=float,
        default=None,
        help="Optional one or more radii for the normalized sum kernel.",
    )
    parser.add_argument(
        "--shooting_weights", "--kernel_weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional nonnegative kernel coefficients; normalized to sum to one.",
    )
    parser.add_argument("--kernel", choices=["c2", "c4"], default="c2")
    parser.add_argument("--max_neighbors", type=int, default=128)
    parser.add_argument("--n_steps", "--integration_steps", type=int, default=50)
    parser.add_argument("--amp", action="store_true")

    # Objective.
    parser.add_argument("--w_data", "--attachment_weight", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", "--smoothness_weight", type=float, default=25.0)
    parser.add_argument("--m0_penalty", choices=["l2", "rkhs"], default="l2")
    parser.add_argument("--m0_reg", "--momentum_weight", type=float, default=0.05)
    parser.add_argument("--m0_rkhs_scale", type=float, default=1.0)
    parser.add_argument("--chamfer_chunk", type=int, default=1024)

    # Pointwise safety check.
    parser.add_argument(
        "--strict_pointwise_topology",
        action="store_true",
        help=(
            "For pointwise loss, additionally require exactly identical face "
            "connectivity. Equal vertex count is always required."
        ),
    )

    # Varifold options.
    parser.add_argument(
        "--varifold_radius",
        type=float,
        default=None,
        help="Defaults to --radius when omitted.",
    )
    parser.add_argument(
        "--varifold_kernel",
        choices=["c2", "c4"],
        default=None,
        help="Defaults to --kernel when omitted.",
    )
    parser.add_argument("--var_Ti", type=int, default=1024)
    parser.add_argument("--var_Tj", type=int, default=2048)
    parser.add_argument("--var_amp", action="store_true")
    parser.add_argument("--var_detach_geom", action="store_true")
    parser.add_argument(
        "--evaluate_varifold",
        action="store_true",
        help="Report the varifold endpoint score even when optimizing another loss.",
    )
    parser.add_argument(
        "--var_face_max",
        type=int,
        default=None,
        help=(
            "Optional fixed number of source and target faces used by the "
            "varifold loss. Use only when full-face computation is too costly."
        ),
    )
    parser.add_argument(
        "--no_var_normalize_areas",
        dest="var_normalize_areas",
        action="store_false",
        help="Disable separate unit-area normalization of source and target.",
    )
    parser.set_defaults(var_normalize_areas=True)

    # Optimization.
    parser.add_argument("--adam_iters", type=int, default=300)
    parser.add_argument("--adam_lr", type=float, default=1.0e-2)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--lbfgs_iters", type=int, default=50)
    parser.add_argument("--lbfgs_lr", type=float, default=1.0)
    parser.add_argument("--lbfgs_history_size", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--keep_best_on_failure",
        action="store_true",
        help="On a non-finite/divergent evaluation, save the best finite state and partial history.",
    )
    parser.add_argument(
        "--divergence_factor",
        type=float,
        default=None,
        help="Stop when |objective| exceeds this multiple of its initial magnitude.",
    )

    # Outputs.
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument(
        "--trajectory_samples",
        type=int,
        default=20,
        help="Number of intervals in the saved trajectory.",
    )
    parser.add_argument("--varifold_backend", choices=["tiled", "stable"], default="tiled")
    return parser


def validate_pointwise_pair(
    source: MeshData,
    target: MeshData,
    strict_topology: bool,
) -> None:
    n_source = source.points.shape[0]
    n_target = target.points.shape[0]
    if n_source != n_target:
        raise ValueError(
            "Pointwise loss requires one-to-one vertex correspondence and equal "
            f"vertex counts, but source has {n_source} vertices and target has "
            f"{n_target}. Use --loss chamfer or --loss varifold."
        )
    if strict_topology:
        same_shape = source.faces.shape == target.faces.shape
        same_faces = same_shape and torch.equal(source.faces, target.faces)
        if not same_faces:
            raise ValueError(
                "--strict_pointwise_topology was requested, but source and target "
                "face connectivity are not identical."
            )
    print(
        "Pointwise mode: vertex i is assumed to correspond to target vertex i. "
        "Equal vertex count is verified, but semantic ordering cannot be "
        "inferred automatically."
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        validate_comparator_arguments(args, "ges")
    except ValueError as exc:
        parser.error(str(exc))
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if radius_graph is None:
        raise RuntimeError(
            "torch_cluster could not be imported. The script intentionally uses "
            "the same sparse radius-graph Hamiltonian as DSMN. Install the "
            "torch_cluster wheel matching the local PyTorch/CUDA build. "
            f"Original error: {_TORCH_CLUSTER_IMPORT_ERROR}"
        )
    if args.n_steps <= 0:
        raise ValueError("--n_steps must be positive.")
    if args.adam_iters < 0 or args.lbfgs_iters < 0:
        raise ValueError("Optimizer iteration counts cannot be negative.")
    if args.divergence_factor is not None and args.divergence_factor <= 1.0:
        raise ValueError("--divergence_factor must be greater than 1 when supplied.")
    if args.adam_iters == 0 and args.lbfgs_iters == 0:
        print("Warning: no optimization iterations requested; only initialization is evaluated.")

    source_path, target_path = args.pairs
    source = load_mesh(source_path, device=device, dtype=dtype)
    target = load_mesh(target_path, device=device, dtype=dtype)

    if args.loss == "pointwise":
        validate_pointwise_pair(source, target, args.strict_pointwise_topology)

    masses = load_or_compute_masses(
        source,
        mass_npy=args.mass_npy,
        device=device,
        dtype=dtype,
    )
    initial_density = prepare_initial_density(
        init_path=args.init_m0,
        init_key=args.init_key,
        representation=args.init_representation,
        init_scale=args.init_scale,
        n_vertices=source.points.shape[0],
        masses=masses,
        device=device,
        dtype=dtype,
    )

    radii, weights = resolve_shooting_kernel(
        radius=args.radius,
        shooting_radii=args.shooting_radii,
        shooting_weights=args.shooting_weights,
    )
    dynamics = HamiltonianDynamics(
        radii=radii,
        weights=weights,
        kind=args.kernel,
        max_neighbors=args.max_neighbors,
        amp=args.amp,
    ).to(device)

    gram: Optional[torch.Tensor] = None
    if args.m0_penalty == "rkhs":
        print("Building the static source RKHS Gram matrix...")
        gram = build_sparse_gram_multi_wendland(
            source.points,
            radii=radii,
            weights=weights,
            kind=args.kernel,
            max_neighbors=args.max_neighbors,
        )

    varifold_module: Optional[VarifoldLoss] = None
    varifold_radius = args.radius if args.varifold_radius is None else args.varifold_radius
    varifold_kernel = args.kernel if args.varifold_kernel is None else args.varifold_kernel
    if args.loss == "varifold" or args.evaluate_varifold:
        varifold_module = VarifoldLoss(
            source_faces=source.faces,
            target_points=target.points,
            target_faces=target.faces,
            radius=float(varifold_radius),
            kind=str(varifold_kernel),
            normalize_areas=bool(args.var_normalize_areas),
            tile_i=args.var_Ti,
            tile_j=args.var_Tj,
            amp=args.var_amp,
            detach_geometry=args.var_detach_geom,
            max_faces=args.var_face_max,
            seed=args.seed,
            backend=args.varifold_backend,
        ).to(device)
        print(
            "Varifold faces: "
            f"source={varifold_module.source_subset_size}, "
            f"target={varifold_module.target_subset_size}; "
            f"radius={varifold_radius}, kernel={varifold_kernel}"
        )

    objective = GeodesicEulerianObjective(
        source_points=source.points,
        source_faces=source.faces,
        target_points=target.points,
        masses=masses,
        dynamics=dynamics,
        n_steps=args.n_steps,
        loss_name=args.loss,
        chamfer_chunk=args.chamfer_chunk,
        varifold_loss=varifold_module,
        w_data=args.w_data,
        lambda_smooth=args.lambda_smooth,
        m0_penalty=args.m0_penalty,
        m0_reg=args.m0_reg,
        m0_rkhs_scale=args.m0_rkhs_scale,
        gram=gram,
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args).copy()
    config["method"] = "Geodesic Eulerian Shooting (GES)"
    config["metric"] = args.loss
    config.update(
        {
            "source_vtk": source.path,
            "target_vtk": target.path,
            "device_resolved": str(device),
            "dtype_resolved": str(dtype),
            "source_vertices": int(source.points.shape[0]),
            "target_vertices": int(target.points.shape[0]),
            "source_faces": int(source.faces.shape[0]),
            "target_faces": int(target.faces.shape[0]),
            "shooting_radii_resolved": radii,
            "shooting_weights_resolved": weights,
            "varifold_radius_resolved": float(varifold_radius),
            "varifold_kernel_resolved": str(varifold_kernel),
        }
    )
    save_json(os.path.join(args.output_dir, "config.json"), config)
    np.save(
        os.path.join(args.output_dir, "initial_m0_density.npy"),
        initial_density.detach().cpu().numpy().astype(np.float32),
    )

    print("\nGeodesic Eulerian Shooting (GES)")
    print(f"  source: {source.path}")
    print(f"  target: {target.path}")
    print(f"  loss: {args.loss}")
    print(f"  device/dtype: {device}/{dtype}")
    print(f"  shooting radii: {radii}")
    print(f"  shooting weights: {weights}")
    print(f"  initialization: {'zeros' if args.init_m0 is None else args.init_m0}")
    print()

    optimized_density, history, best, termination = optimize_initial_momentum(
        objective=objective,
        initial_density=initial_density,
        adam_iters=args.adam_iters,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_iters=args.lbfgs_iters,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_history_size=args.lbfgs_history_size,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        keep_best_on_failure=args.keep_best_on_failure,
        divergence_factor=args.divergence_factor,
    )

    history_path = os.path.join(args.output_dir, "optimization_history.csv")
    write_history(history_path, history)

    p0 = masses.view(-1, 1) * optimized_density
    endpoint, terminal_p, x_path, p_path = final_shoot(
        dynamics=dynamics,
        source_points=source.points,
        p0=p0,
        n_steps=args.n_steps,
        trajectory_samples=args.trajectory_samples,
    )
    finite_or_raise("final endpoint", endpoint)
    finite_or_raise("final terminal momentum", terminal_p)

    output_paths = save_outputs(
        output_dir=args.output_dir,
        source=source,
        target=target,
        endpoint=endpoint,
        m0_density=optimized_density,
        p0=p0,
        x_path=x_path,
        p_path=p_path,
        save_trajectory=args.save_trajectory,
    )

    pointwise_available = source.points.shape[0] == target.points.shape[0]
    initial_metrics = endpoint_metrics(
        endpoint=source.points,
        target=target.points,
        chunk_size=args.chamfer_chunk,
        pointwise_available=pointwise_available,
        varifold_loss=varifold_module,
    )
    metrics = endpoint_metrics(
        endpoint=endpoint,
        target=target.points,
        chunk_size=args.chamfer_chunk,
        pointwise_available=pointwise_available,
        varifold_loss=varifold_module,
    )
    with torch.no_grad():
        final_data = float(objective.data_loss(endpoint).detach().cpu())
        final_smooth = float(
            smoothness_loss(endpoint - source.points, source.faces).detach().cpu()
        )
        if args.m0_penalty == "l2":
            final_m0_raw = float(optimized_density.pow(2).mean().detach().cpu())
        else:
            assert gram is not None
            final_m0_raw = float(
                (
                    rkhs_momentum_norm2(gram, p0) / source.points.shape[0]
                )
                .detach()
                .cpu()
            )

    summary = {
        "status": termination.status,
        "termination_reason": termination.reason,
        "termination_evaluation": termination.evaluation,
        "best_objective": float(best.value),
        "best_stage": best.stage,
        "best_evaluation": int(best.evaluation),
        "final_selected_data_loss": final_data,
        "final_smoothness": final_smooth,
        "final_m0_raw": final_m0_raw,
        "initial_metrics": initial_metrics,
        "metrics": metrics,
        "outputs": output_paths,
        "history_csv": os.path.abspath(history_path),
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)

    print(f"\nOptimization finished with status: {termination.status}.")
    print(f"Best objective: {best.value:.6e} ({best.stage}, eval {best.evaluation})")
    print(f"Final selected data loss: {final_data:.6e}")
    print(f"Endpoint metrics: {metrics}")
    print(f"Deformed VTK: {output_paths['deformed_vtk']}")
    print(f"Optimized DSMN-compatible m0 density: {output_paths['m0_density_npy']}")
    print(f"Area-weighted discrete p0: {output_paths['p0_discrete_npy']}")
    print(f"Summary: {os.path.join(os.path.abspath(args.output_dir), 'summary.json')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise
