#!/usr/bin/env python3
"""Direct Template-Body Matching (DTM) surface matching.

Optimizes a time-dependent template-body Mercer coefficient path alpha using
the original approximate forward body decoder xdot = F b(X), Fdot = F Db(X).
This comparator is distinct from the endpoint-equivalent Eulerian body decoder.
Pointwise, vertex Chamfer and legacy varifold-adapter attachments are supported.
All numerical defaults, parameterizations, RK4 integration and outputs are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from .surface_io import (finite_or_raise, MeshData, load_mesh, load_mercer_basis,
                         smoothness_loss, directed_nearest_distances, distribution, endpoint_metrics)
from .metrics import VarifoldLoss
from .matching_cli import validate_comparator_arguments


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Basic utilities and I/O
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




def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)








# =============================================================================
# Wendland kernel derivatives and fixed Mercer gradient basis
# =============================================================================


def wendland_dphi(q: torch.Tensor, kind: str) -> torch.Tensor:
    s = torch.clamp(1.0 - q, min=0.0)
    inside = (q >= 0.0) & (q <= 1.0)
    if kind == "c2":
        value = -20.0 * q * s.pow(3)
    elif kind == "c4":
        poly = (35.0 * q.pow(2) + 18.0 * q + 3.0) / 3.0
        poly_prime = (70.0 * q + 18.0) / 3.0
        value = s.pow(5) * (-6.0 * poly + s * poly_prime)
    else:
        raise ValueError(f"Unsupported Wendland kernel: {kind}")
    return torch.where(inside, value, torch.zeros_like(value))


@torch.no_grad()
def precompute_template_gradient_basis(
    source_points: torch.Tensor,
    U: torch.Tensor,
    radius: float,
    kind: str,
    tile_q: int,
    tile_s: int,
) -> torch.Tensor:
    """Precompute G[i,c,j] = sum_l d_c K(X_i,X_l) U[l,j]."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("--radius must be positive.")
    vertex_count, basis_count = U.shape
    result = torch.zeros(
        (vertex_count, 3, basis_count),
        dtype=source_points.dtype,
        device=source_points.device,
    )
    eps = torch.finfo(source_points.dtype).eps
    for query_start in range(0, vertex_count, int(tile_q)):
        query_stop = min(query_start + int(tile_q), vertex_count)
        query = source_points[query_start:query_stop]
        accumulator = torch.zeros(
            (query_stop - query_start, 3, basis_count),
            dtype=source_points.dtype,
            device=source_points.device,
        )
        for source_start in range(0, vertex_count, int(tile_s)):
            source_stop = min(source_start + int(tile_s), vertex_count)
            centers = source_points[source_start:source_stop]
            modes = U[source_start:source_stop]
            difference = query[:, None, :] - centers[None, :, :]
            distance = torch.linalg.norm(difference, dim=-1)
            q = distance / float(radius)
            coefficient = wendland_dphi(q, kind) / (
                float(radius) * torch.clamp(distance, min=eps)
            )
            gradient = coefficient[..., None] * difference
            accumulator.add_(torch.einsum("qsc,sb->qcb", gradient, modes))
        result[query_start:query_stop] = accumulator
    finite_or_raise("precomputed Mercer gradient basis", result)
    return result


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_or_build_gradient_basis(
    cache_path: Optional[str],
    source_points: torch.Tensor,
    U: torch.Tensor,
    radius: float,
    kind: str,
    tile_q: int,
    tile_s: int,
) -> Tuple[torch.Tensor, str]:
    expected_shape = (source_points.shape[0], 3, U.shape[1])
    source_digest = _tensor_digest(source_points)
    basis_digest = _tensor_digest(U)
    if cache_path:
        resolved = os.path.abspath(os.path.expanduser(cache_path))
        if os.path.isfile(resolved):
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "gradient_basis" not in payload:
                raise ValueError(f"Invalid gradient-basis cache: {resolved}")
            cached = payload["gradient_basis"]
            checks = {
                "shape": tuple(cached.shape) == expected_shape,
                "radius": math.isclose(float(payload.get("radius", math.nan)), radius),
                "kind": payload.get("kind") == kind,
                "source_digest": payload.get("source_digest") == source_digest,
                "basis_digest": payload.get("basis_digest") == basis_digest,
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise ValueError(
                    f"Gradient-basis cache does not match this run ({', '.join(failed)}): "
                    f"{resolved}"
                )
            print(f"Loading precomputed Mercer gradient basis: {resolved}")
            return cached.to(device=source_points.device, dtype=source_points.dtype), resolved

    print("Precomputing the fixed Mercer velocity-gradient basis...")
    started = time.perf_counter()
    gradient_basis = precompute_template_gradient_basis(
        source_points, U, radius, kind, tile_q=tile_q, tile_s=tile_s
    )
    print(
        f"Gradient basis ready: shape={tuple(gradient_basis.shape)}, "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    if cache_path:
        resolved = os.path.abspath(os.path.expanduser(cache_path))
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        torch.save(
            {
                "gradient_basis": gradient_basis.detach().cpu(),
                "radius": float(radius),
                "kind": kind,
                "source_digest": source_digest,
                "basis_digest": basis_digest,
            },
            resolved,
        )
        print(f"Saved Mercer gradient-basis cache: {resolved}")
        return gradient_basis, resolved
    return gradient_basis, "not_saved"


# =============================================================================
# Approximate direct left-invariant decoder (Eqs. 41, 42, 126, and 130)
# =============================================================================


def interpolate_knots(values: torch.Tensor, t: float) -> torch.Tensor:
    knot_count = values.shape[0]
    if knot_count == 1:
        return values[0]
    position = min(max(float(t), 0.0), 1.0) * (knot_count - 1)
    lower = min(int(math.floor(position)), knot_count - 1)
    upper = min(lower + 1, knot_count - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _left_rhs(
    state: Tuple[torch.Tensor, torch.Tensor],
    velocity: torch.Tensor,
    jacobian_transposed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_t, F_t = state
    del x_t  # The approximate system is driven in fixed body coordinates.
    dx = torch.einsum("vij,vj->vi", F_t, velocity)
    # jacobian_transposed[v,c,d] = d v_d / d X_c.  Therefore this
    # contraction is F @ Dv in the conventional output-by-input layout.
    dF = torch.einsum("vij,vkj->vik", F_t, jacobian_transposed)
    return dx, dF


def _state_add(
    state: Tuple[torch.Tensor, torch.Tensor],
    derivative: Tuple[torch.Tensor, torch.Tensor],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return state[0] + scale * derivative[0], state[1] + scale * derivative[1]


def integrate_template_body_path(
    source_points: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    gradient_basis: torch.Tensor,
    alpha_knots: torch.Tensor,
    n_steps: int,
    return_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Integrate the approximate direct left-invariant system with RK4."""
    U_lam = U * lam.view(1, -1)
    velocity_knots = torch.einsum("vb,tbd->tvd", U_lam, alpha_knots)
    jacobian_knots = torch.einsum(
        "vcb,tbd->tvcd", gradient_basis, alpha_knots
    )

    vertex_count = source_points.shape[0]
    identity = torch.eye(
        3, dtype=source_points.dtype, device=source_points.device
    ).expand(vertex_count, -1, -1).clone()
    state = (source_points, identity)
    x_frames: List[torch.Tensor] = [source_points] if return_path else []
    F_frames: List[torch.Tensor] = [identity] if return_path else []
    dt = 1.0 / float(n_steps)

    for step in range(int(n_steps)):
        t0 = step * dt
        tm = t0 + 0.5 * dt
        t1 = t0 + dt
        v0 = interpolate_knots(velocity_knots, t0)
        vm = interpolate_knots(velocity_knots, tm)
        v1 = interpolate_knots(velocity_knots, t1)
        J0 = interpolate_knots(jacobian_knots, t0)
        Jm = interpolate_knots(jacobian_knots, tm)
        J1 = interpolate_knots(jacobian_knots, t1)

        k1 = _left_rhs(state, v0, J0)
        k2 = _left_rhs(_state_add(state, k1, 0.5 * dt), vm, Jm)
        k3 = _left_rhs(_state_add(state, k2, 0.5 * dt), vm, Jm)
        k4 = _left_rhs(_state_add(state, k3, dt), v1, J1)
        x_next = state[0] + (dt / 6.0) * (
            k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]
        )
        F_next = state[1] + (dt / 6.0) * (
            k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]
        )
        state = (x_next, F_next)
        if return_path:
            x_frames.append(x_next)
            F_frames.append(F_next)

    x_path = torch.stack(x_frames) if return_path else None
    F_path = torch.stack(F_frames) if return_path else None
    return state[0], state[1], x_path, F_path


def exact_piecewise_linear_quadratic(
    alpha_knots: torch.Tensor,
    weights: Optional[torch.Tensor],
    denominator: float,
) -> torch.Tensor:
    """Exact integral of a diagonal quadratic along a linear knot path."""
    if weights is None:
        weighted = alpha_knots
    else:
        weighted = alpha_knots * weights.sqrt().view(1, -1, 1)
    if weighted.shape[0] == 1:
        return weighted.pow(2).sum() / float(denominator)
    left = weighted[:-1]
    right = weighted[1:]
    interval_values = (left.pow(2) + left * right + right.pow(2)).sum(dim=(1, 2)) / 3.0
    return interval_values.mean() / float(denominator)


# =============================================================================
# Objective and optimization
# =============================================================================




def directed_squared_nearest_mean(
    source: torch.Tensor, target: torch.Tensor, chunk: int
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, source.shape[0], int(chunk)):
        distance = torch.cdist(source[start : start + int(chunk)], target)
        values.append(distance.pow(2).min(dim=1).values)
    return torch.cat(values).mean()


def chamfer_squared(a: torch.Tensor, b: torch.Tensor, chunk: int) -> torch.Tensor:
    return directed_squared_nearest_mean(a, b, chunk) + directed_squared_nearest_mean(
        b, a, chunk
    )


class DirectTemplateBodyObjective(nn.Module):
    def __init__(
        self,
        source_points: torch.Tensor,
        source_faces: torch.Tensor,
        target_points: torch.Tensor,
        U: torch.Tensor,
        lam: torch.Tensor,
        gradient_basis: torch.Tensor,
        time_knots: int,
        n_steps: int,
        loss_name: str,
        chamfer_chunk: int,
        parameterization: str,
        lambda_rec: float,
        lambda_smooth: float,
        lambda_velocity: float,
        lambda_moment: float,
        lambda_time: float,
        lambda_jac: float,
        varifold_loss: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("source_points", source_points.detach())
        self.register_buffer("source_faces", source_faces)
        self.register_buffer("target_points", target_points.detach())
        self.register_buffer("U", U.detach())
        self.register_buffer("lam", lam.detach())
        self.register_buffer("gradient_basis", gradient_basis.detach())
        self.time_knots = int(time_knots)
        self.n_steps = int(n_steps)
        self.varifold_loss = varifold_loss
        self.loss_name = loss_name
        self.chamfer_chunk = int(chamfer_chunk)
        self.parameterization = parameterization
        self.lambda_rec = float(lambda_rec)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_moment = float(lambda_moment)
        self.lambda_time = float(lambda_time)
        self.lambda_jac = float(lambda_jac)

    @property
    def parameter_shape(self) -> Tuple[int, int, int]:
        return self.time_knots, self.U.shape[1], 3

    def alpha_to_parameter(self, alpha: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "alpha":
            return alpha
        if self.parameterization == "energy":
            return alpha * self.lam.sqrt().view(1, -1, 1)
        raise ValueError(f"Unsupported parameterization: {self.parameterization}")

    def parameter_to_alpha(self, parameter: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "alpha":
            return parameter
        if self.parameterization == "energy":
            return parameter / self.lam.sqrt().view(1, -1, 1)
        raise ValueError(f"Unsupported parameterization: {self.parameterization}")

    def data_loss(self, endpoint: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "varifold":
            if self.varifold_loss is None:
                raise RuntimeError("Varifold loss was not initialized.")
            return self.varifold_loss(endpoint)
        if self.loss_name == "pointwise":
            return (endpoint - self.target_points).pow(2).mean()
        if self.loss_name == "chamfer":
            return chamfer_squared(endpoint, self.target_points, self.chamfer_chunk)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def forward(self, parameter: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        alpha = self.parameter_to_alpha(parameter)
        endpoint, F_T, _, _ = integrate_template_body_path(
            self.source_points,
            self.U,
            self.lam,
            self.gradient_basis,
            alpha,
            self.n_steps,
            return_path=False,
        )
        finite_or_raise("left-invariant endpoint", endpoint)
        finite_or_raise("left-invariant local Jacobians", F_T)

        data = self.data_loss(endpoint)
        smooth = smoothness_loss(endpoint - self.source_points, self.source_faces)
        velocity = exact_piecewise_linear_quadratic(
            alpha,
            weights=self.lam,
            denominator=self.source_points.shape[0] * 3,
        )
        # The stored U columns are Euclidean-orthonormal Mercer eigenvectors, so
        # ||U alpha||_F^2 = ||alpha||_F^2 for the discrete moment penalty.
        moment = exact_piecewise_linear_quadratic(
            alpha,
            weights=None,
            denominator=self.source_points.shape[0] * 3,
        )
        if alpha.shape[0] > 1:
            temporal = (alpha[1:] - alpha[:-1]).pow(2).mean()
        else:
            temporal = alpha.new_zeros(())
        identity = torch.eye(3, dtype=F_T.dtype, device=F_T.device).view(1, 3, 3)
        jacobian = (F_T - identity).pow(2).mean()

        total = (
            self.lambda_rec * data
            + self.lambda_smooth * smooth
            + self.lambda_velocity * velocity
            + self.lambda_moment * moment
            + self.lambda_time * temporal
            + self.lambda_jac * jacobian
        )
        components = {
            "total": total,
            "data": data,
            "smooth": smooth,
            "velocity": velocity,
            "moment": moment,
            "temporal": temporal,
            "jacobian": jacobian,
            "endpoint": endpoint,
            "F_T": F_T,
            "alpha": alpha,
        }
        return total, components


@dataclass
class BestState:
    value: float = math.inf
    parameter: Optional[torch.Tensor] = None
    stage: str = ""
    evaluation: int = -1


def history_row(
    stage: str,
    evaluation: int,
    elapsed: float,
    components: Dict[str, torch.Tensor],
    grad_norm: float,
) -> Dict:
    row: Dict = {
        "stage": stage,
        "evaluation": int(evaluation),
        "elapsed_sec": float(elapsed),
        "grad_norm": float(grad_norm),
    }
    for name in (
        "total",
        "data",
        "smooth",
        "velocity",
        "moment",
        "temporal",
        "jacobian",
    ):
        row[name] = float(components[name].detach().cpu())
    return row


def write_history(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def optimize_body_coefficients(
    objective: DirectTemplateBodyObjective,
    initial_alpha: torch.Tensor,
    adam_iters: int,
    adam_lr: float,
    adam_weight_decay: float,
    lbfgs_iters: int,
    lbfgs_lr: float,
    lbfgs_history_size: int,
    grad_clip: float,
    log_every: int,
) -> Tuple[torch.Tensor, List[Dict], BestState]:
    initial_parameter = objective.alpha_to_parameter(initial_alpha)
    parameter = nn.Parameter(initial_parameter.clone())
    history: List[Dict] = []
    best = BestState()
    started = time.perf_counter()
    evaluation = 0

    def evaluate(stage: str, clip_gradient: bool) -> torch.Tensor:
        nonlocal evaluation, best
        loss, components = objective(parameter)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite objective during {stage} evaluation {evaluation}."
            )
        loss.backward()
        if parameter.grad is None:
            raise RuntimeError("No gradient was produced for the coefficient path.")
        finite_or_raise("coefficient gradient", parameter.grad)
        if clip_gradient and grad_clip > 0.0:
            raw_norm = torch.nn.utils.clip_grad_norm_([parameter], grad_clip)
            grad_norm = float(raw_norm.detach().cpu())
        else:
            grad_norm = float(parameter.grad.norm().detach().cpu())

        value = float(loss.detach().cpu())
        if value < best.value:
            best.value = value
            best.parameter = parameter.detach().clone()
            best.stage = stage
            best.evaluation = evaluation
        row = history_row(
            stage, evaluation, time.perf_counter() - started, components, grad_norm
        )
        history.append(row)
        if evaluation % max(1, int(log_every)) == 0:
            print(
                f"[{stage:7s} eval {evaluation:05d}] total={row['total']:.6e} "
                f"rec={row['data']:.6e} smooth={row['smooth']:.6e} "
                f"vel={row['velocity']:.6e} time={row['temporal']:.6e} "
                f"jac={row['jacobian']:.6e} |grad|={row['grad_norm']:.3e}"
            )
        evaluation += 1
        return loss

    parameter.grad = None
    evaluate("initial", clip_gradient=False)
    parameter.grad = None

    if adam_iters > 0:
        optimizer = torch.optim.AdamW(
            [parameter], lr=float(adam_lr), weight_decay=float(adam_weight_decay)
        )
        for _ in range(int(adam_iters)):
            optimizer.zero_grad(set_to_none=True)
            evaluate("adam", clip_gradient=True)
            optimizer.step()

    if lbfgs_iters > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            [parameter],
            lr=float(lbfgs_lr),
            max_iter=int(lbfgs_iters),
            history_size=int(lbfgs_history_size),
            line_search_fn="strong_wolfe",
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            return evaluate("lbfgs", clip_gradient=False)

        optimizer_lbfgs.step(closure)

    parameter.grad = None
    evaluate("final", clip_gradient=False)
    parameter.grad = None
    if best.parameter is None:
        raise RuntimeError("Optimization produced no finite coefficient path.")
    return objective.parameter_to_alpha(best.parameter), history, best


# =============================================================================
# Evaluation and outputs
# =============================================================================








@torch.no_grad()
def left_numerical_diagnostics(
    source: torch.Tensor,
    endpoint: torch.Tensor,
    F_T: torch.Tensor,
    faces: torch.Tensor,
) -> Dict[str, float]:
    determinant = torch.linalg.det(F_T)
    singular_values = torch.linalg.svdvals(F_T)
    condition = singular_values[:, 0] / torch.clamp(
        singular_values[:, -1], min=torch.finfo(F_T.dtype).eps
    )
    result: Dict[str, float] = {}
    result.update(distribution(determinant, "detF"))
    result.update(distribution(condition, "conditionF"))
    result["detF_nonpositive_fraction"] = float((determinant <= 0.0).float().mean().cpu())
    result["detF_below_0p1_fraction"] = float((determinant < 0.1).float().mean().cpu())
    result["conditionF_above_100_fraction"] = float((condition > 100.0).float().mean().cpu())

    directed_edges = torch.cat(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0
    )
    tail = directed_edges[:, 0]
    head = directed_edges[:, 1]
    body_edge = source[head] - source[tail]
    geometric_edge = endpoint[head] - endpoint[tail]
    locally_predicted_edge = torch.einsum("eij,ej->ei", F_T[tail], body_edge)
    mismatch = torch.linalg.norm(geometric_edge - locally_predicted_edge, dim=1)
    relative_mismatch = mismatch / torch.clamp(
        torch.linalg.norm(geometric_edge, dim=1),
        min=torch.finfo(endpoint.dtype).eps,
    )
    result.update(distribution(relative_mismatch, "local_global_edge_mismatch"))
    result["local_global_edge_mismatch_mean"] = float(relative_mismatch.mean().cpu())
    result["local_global_edge_mismatch_rms"] = float(
        relative_mismatch.pow(2).mean().sqrt().cpu()
    )

    def triangle_areas(points: torch.Tensor) -> torch.Tensor:
        tri = points[faces]
        return 0.5 * torch.linalg.norm(
            torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1
        )

    source_area = triangle_areas(source)
    endpoint_area = triangle_areas(endpoint)
    area_ratio = endpoint_area / torch.clamp(
        source_area, min=torch.finfo(endpoint.dtype).eps
    )
    result.update(distribution(area_ratio, "surface_area_ratio"))
    result["surface_area_ratio_below_0p1_fraction"] = float(
        (area_ratio < 0.1).float().mean().cpu()
    )
    result["surface_area_ratio_above_10_fraction"] = float(
        (area_ratio > 10.0).float().mean().cpu()
    )
    return result


def save_outputs(
    output_dir: str,
    source: MeshData,
    target: MeshData,
    alpha: torch.Tensor,
    endpoint: torch.Tensor,
    F_T: torch.Tensor,
    x_path: Optional[torch.Tensor],
    F_path: Optional[torch.Tensor],
    trajectory_samples: int,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    coeff_dir = os.path.join(output_dir, "coeff_outputs")
    os.makedirs(coeff_dir, exist_ok=True)
    target_stem = Path(target.path).stem

    alpha_path = os.path.join(output_dir, "alpha_knots.npy")
    alpha_explicit_path = os.path.join(coeff_dir, "final_alpha_knots.npy")
    endpoint_path = os.path.join(output_dir, "final_forward_source.npy")
    jacobian_path = os.path.join(output_dir, "final_jacobian.npy")
    determinant_path = os.path.join(output_dir, "final_detF.npy")
    np.save(alpha_path, alpha.detach().cpu().numpy().astype(np.float32))
    np.save(alpha_explicit_path, alpha.detach().cpu().numpy().astype(np.float32))
    np.save(endpoint_path, endpoint.detach().cpu().numpy().astype(np.float32))
    np.save(jacobian_path, F_T.detach().cpu().numpy().astype(np.float32))
    np.save(
        determinant_path,
        torch.linalg.det(F_T).detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(output_dir, "final_source.npy"),
        source.points.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(output_dir, "final_target.npy"),
        target.points.detach().cpu().numpy().astype(np.float32),
    )

    output_mesh = pv.PolyData(
        endpoint.detach().cpu().numpy().astype(np.float64),
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    )
    vtk_path = os.path.join(
        output_dir, f"{target_stem}_left_alpha_optimized_deformed.vtk"
    )
    output_mesh.save(vtk_path)
    paths = {
        "deformed_vtk": os.path.abspath(vtk_path),
        "alpha_knots_npy": os.path.abspath(alpha_path),
        "alpha_knots_explicit_npy": os.path.abspath(alpha_explicit_path),
        "endpoint_npy": os.path.abspath(endpoint_path),
        "jacobian_npy": os.path.abspath(jacobian_path),
        "detF_npy": os.path.abspath(determinant_path),
    }

    if x_path is not None and F_path is not None:
        sample_count = max(1, min(int(trajectory_samples), x_path.shape[0] - 1))
        indices = torch.linspace(
            0, x_path.shape[0] - 1, sample_count + 1, device=x_path.device
        ).round().long().unique(sorted=True)
        sampled_x = x_path[indices]
        sampled_F = F_path[indices]
        x_path_file = os.path.join(output_dir, "x_path.npy")
        F_path_file = os.path.join(output_dir, "F_path.npy")
        np.save(x_path_file, sampled_x.detach().cpu().numpy().astype(np.float32))
        np.save(F_path_file, sampled_F.detach().cpu().numpy().astype(np.float32))
        paths["x_path_npy"] = os.path.abspath(x_path_file)
        paths["F_path_npy"] = os.path.abspath(F_path_file)

        frames_dir = os.path.join(output_dir, "trajectory_vtk")
        os.makedirs(frames_dir, exist_ok=True)
        for frame_index, frame in enumerate(sampled_x.detach().cpu().numpy()):
            frame_mesh = pv.PolyData(
                frame.astype(np.float64),
                np.asarray(source.mesh.faces, dtype=np.int64).copy(),
            )
            frame_mesh.save(os.path.join(frames_dir, f"frame_{frame_index:03d}.vtk"))
        paths["trajectory_vtk_dir"] = os.path.abspath(frames_dir)
    return paths


def load_initial_alpha(
    path: Optional[str],
    shape: Tuple[int, int, int],
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if path is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    resolved = os.path.abspath(os.path.expanduser(path))
    array = np.load(resolved)
    tensor = torch.as_tensor(array, dtype=dtype, device=device)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"Initial alpha shape {tuple(tensor.shape)} does not match expected {shape}."
        )
    finite_or_raise("initial alpha", tensor)
    return tensor * float(scale)


# =============================================================================
# CLI and main
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Direct Template-Body Matching (DTM): optimize alpha(t) through the "
            "approximate forward left-invariant decoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        nargs=2,
        metavar=("SOURCE_VTK", "TARGET_VTK"),
        required=True,
        help="One source-target VTK pair.",
    )
    parser.add_argument("--basis", required=True, help="Mercer basis .npz file.")
    parser.add_argument("--basis_dim", "--body_rank", type=int, default=None)
    parser.add_argument(
        "--output_dir", "--save_dir", dest="output_dir", required=True
    )
    parser.add_argument(
        "--metric", "--loss",
        "--rec_loss",
        dest="loss",
        choices=["pointwise", "chamfer", "varifold"],
        default="pointwise",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Compatibility option; direct per-pair optimization requires 1.",
    )

    parser.add_argument("--time_knots", "--body_knots", type=int, default=30)
    parser.add_argument("--n_steps", "--integration_steps", type=int, default=30)
    parser.add_argument("--kernel", choices=["auto", "c2", "c4"], default="auto")
    parser.add_argument("--radius", "--kernel_radius", type=float, default=None)
    parser.add_argument("--gradient_tile_q", type=int, default=64)
    parser.add_argument("--gradient_tile_s", type=int, default=256)
    parser.add_argument(
        "--gradient_basis_cache",
        type=str,
        default=None,
        help="Optional .pt cache shared by runs with the same source, basis, and kernel.",
    )

    parser.add_argument("--lambda_rec", "--attachment_weight", type=float, default=200.0)
    parser.add_argument("--lambda_smooth", "--smoothness_weight", type=float, default=5.0)
    parser.add_argument("--lambda_velocity", "--body_energy_weight", type=float, default=1.0)
    parser.add_argument("--lambda_moment", "--body_moment_weight", type=float, default=0.0)
    parser.add_argument("--lambda_time", "--body_temporal_weight", type=float, default=0.0)
    parser.add_argument("--lambda_jac", "--jacobian_weight", type=float, default=0.0)
    parser.add_argument("--chamfer_chunk", type=int, default=1024)
    parser.add_argument("--strict_pointwise_topology", action="store_true")

    parser.add_argument(
        "--parameterization",
        choices=["alpha", "energy"],
        default="alpha",
        help=(
            "alpha optimizes Eq. (126) literally; energy uses the bijective "
            "z=sqrt(lambda)*alpha coordinates only as an optimizer preconditioner."
        ),
    )
    parser.add_argument("--init_alpha", type=str, default=None)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument(
        "--adam_iters", "--epochs", dest="adam_iters", type=int, default=500
    )
    parser.add_argument("--adam_lr", "--lr", dest="adam_lr", type=float, default=1.0e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--lbfgs_iters", type=int, default=0)
    parser.add_argument("--lbfgs_lr", type=float, default=1.0)
    parser.add_argument("--lbfgs_history_size", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--trajectory_samples", type=int, default=20)
    parser.add_argument("--varifold_radius", type=float, default=None)
    parser.add_argument("--varifold_kernel", choices=["c2", "c4"], default=None)
    parser.add_argument("--var_Ti", type=int, default=1024)
    parser.add_argument("--var_Tj", type=int, default=2048)
    parser.add_argument("--var_amp", action="store_true")
    parser.add_argument("--var_detach_geom", action="store_true")
    parser.add_argument("--var_face_max", type=int, default=None)
    parser.add_argument("--no_var_normalize_areas", dest="var_normalize_areas", action="store_false")
    parser.set_defaults(var_normalize_areas=True)
    parser.add_argument("--varifold_backend", choices=["tiled", "stable"], default="tiled")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        validate_comparator_arguments(args, "dtm")
    except ValueError as exc:
        parser.error(str(exc))
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.batch_size != 1:
        raise ValueError("Direct per-pair optimization requires --batch_size 1.")
    if args.time_knots <= 0 or args.n_steps <= 0:
        raise ValueError("--time_knots and --n_steps must be positive.")
    if args.adam_iters < 0 or args.lbfgs_iters < 0:
        raise ValueError("Optimizer iteration counts cannot be negative.")
    if args.gradient_tile_q <= 0 or args.gradient_tile_s <= 0:
        raise ValueError("Gradient tile sizes must be positive.")

    source = load_mesh(args.pairs[0], device=device, dtype=dtype)
    target = load_mesh(args.pairs[1], device=device, dtype=dtype)
    U, lam, basis_meta = load_mercer_basis(
        args.basis, args.basis_dim, device=device, dtype=dtype
    )
    if U.shape[0] != source.points.shape[0]:
        raise ValueError(
            f"Source has {source.points.shape[0]} vertices, but the Mercer basis "
            f"has {U.shape[0]} rows."
        )
    if args.loss == "pointwise" and source.points.shape[0] != target.points.shape[0]:
        raise ValueError(
            "Pointwise loss requires equal source and target vertex counts. "
            "Use --loss chamfer otherwise."
        )
    if args.strict_pointwise_topology and args.loss == "pointwise":
        same_shape = source.faces.shape == target.faces.shape
        same_faces = same_shape and torch.equal(source.faces, target.faces)
        if not same_faces:
            raise ValueError(
                "--strict_pointwise_topology was requested, but face connectivity differs."
            )

    kernel = str(basis_meta.get("kernel", "c2")) if args.kernel == "auto" else args.kernel
    if kernel not in ("c2", "c4"):
        raise ValueError(f"Unsupported resolved kernel in basis metadata: {kernel}")
    if args.radius is None:
        if "R" not in basis_meta:
            raise ValueError("--radius was omitted and basis metadata has no 'R'.")
        radius = float(basis_meta["R"])
    else:
        radius = float(args.radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("--radius must be positive.")

    gradient_basis, gradient_cache_status = load_or_build_gradient_basis(
        args.gradient_basis_cache,
        source.points,
        U,
        radius,
        kernel,
        args.gradient_tile_q,
        args.gradient_tile_s,
    )
    # Match the historical adapter's explicit attachment-radius/kernel fallback.
    # In particular, auto body kernel does not change the adapter's C2 default.
    varifold_module = None
    varifold_radius = args.varifold_radius if args.varifold_radius is not None else args.radius
    varifold_kernel = args.varifold_kernel or args.kernel
    if varifold_kernel == "auto":
        varifold_kernel = "c2"
    if args.loss == "varifold":
        if varifold_radius is None:
            raise ValueError("Varifold attachment requires --varifold_radius or --kernel_radius.")
        varifold_module = VarifoldLoss(
            source_faces=source.faces, target_points=target.points, target_faces=target.faces,
            radius=float(varifold_radius), kind=str(varifold_kernel),
            normalize_areas=args.var_normalize_areas, tile_i=args.var_Ti, tile_j=args.var_Tj,
            amp=args.var_amp, detach_geometry=args.var_detach_geom,
            max_faces=args.var_face_max, seed=args.seed, backend=args.varifold_backend,
        ).to(device)
    objective = DirectTemplateBodyObjective(
        source_points=source.points,
        source_faces=source.faces,
        target_points=target.points,
        U=U,
        lam=lam,
        gradient_basis=gradient_basis,
        time_knots=args.time_knots,
        n_steps=args.n_steps,
        loss_name=args.loss,
        chamfer_chunk=args.chamfer_chunk,
        parameterization=args.parameterization,
        lambda_rec=args.lambda_rec,
        lambda_smooth=args.lambda_smooth,
        lambda_velocity=args.lambda_velocity,
        lambda_moment=args.lambda_moment,
        lambda_time=args.lambda_time,
        lambda_jac=args.lambda_jac,
        varifold_loss=varifold_module,
    ).to(device)
    initial_alpha = load_initial_alpha(
        args.init_alpha,
        objective.parameter_shape,
        args.init_scale,
        device=device,
        dtype=dtype,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(
        os.path.join(args.output_dir, "initial_alpha_knots.npy"),
        initial_alpha.detach().cpu().numpy().astype(np.float32),
    )
    config = vars(args).copy()
    config["method"] = "Direct Template-Body Matching (DTM)"
    config["metric"] = args.loss
    config["varifold_radius_resolved"] = varifold_radius
    config["varifold_kernel_resolved"] = varifold_kernel
    config.update(
        {
            "source_vtk": source.path,
            "target_vtk": target.path,
            "basis_path": os.path.abspath(args.basis),
            "source_vertices": int(source.points.shape[0]),
            "target_vertices": int(target.points.shape[0]),
            "source_faces": int(source.faces.shape[0]),
            "target_faces": int(target.faces.shape[0]),
            "basis_dim_resolved": int(U.shape[1]),
            "kernel_resolved": kernel,
            "radius_resolved": radius,
            "device_resolved": str(device),
            "dtype_resolved": str(dtype),
            "gradient_basis_cache_status": gradient_cache_status,
            "optimized_variable": "full alpha path from Eq. (126)",
            "decoder": "approximate direct left-invariant Eqs. (41), (42), and (130)",
        }
    )
    save_json(os.path.join(args.output_dir, "config.json"), config)

    print("\nDirect Template-Body Matching (DTM)")
    print(f"  source: {source.path}")
    print(f"  target: {target.path}")
    print(f"  basis/time knots: {U.shape[1]}/{args.time_knots}")
    print(f"  kernel/radius: {kernel}/{radius}")
    print(f"  decoder steps: {args.n_steps}")
    print(f"  loss: {args.loss}")
    print(f"  parameterization: {args.parameterization}")
    print(f"  device/dtype: {device}/{dtype}\n")

    optimized_alpha, history, best = optimize_body_coefficients(
        objective,
        initial_alpha,
        adam_iters=args.adam_iters,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_iters=args.lbfgs_iters,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_history_size=args.lbfgs_history_size,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )
    history_path = os.path.join(args.output_dir, "optimization_history.csv")
    write_history(history_path, history)

    with torch.no_grad():
        endpoint, F_T, x_path, F_path = integrate_template_body_path(
            source.points,
            U,
            lam,
            gradient_basis,
            optimized_alpha,
            args.n_steps,
            return_path=args.save_trajectory,
        )
        finite_or_raise("final endpoint", endpoint)
        finite_or_raise("final local Jacobian", F_T)
        final_parameter = objective.alpha_to_parameter(optimized_alpha)
        final_total, final_components = objective(final_parameter)

    output_paths = save_outputs(
        args.output_dir,
        source,
        target,
        optimized_alpha,
        endpoint,
        F_T,
        x_path,
        F_path,
        args.trajectory_samples,
    )
    pointwise_available = source.points.shape[0] == target.points.shape[0]
    initial_metrics = endpoint_metrics(
        source.points, target.points, pointwise_available, args.chamfer_chunk
    )
    metrics = endpoint_metrics(
        endpoint, target.points, pointwise_available, args.chamfer_chunk
    )
    diagnostics = left_numerical_diagnostics(
        source.points, endpoint, F_T, source.faces
    )
    summary = {
        "status": "success",
        "optimized_variable": "alpha",
        "alpha_equation": 126,
        "residual_c_equation_not_used": 129,
        "best_objective": float(best.value),
        "best_stage": best.stage,
        "best_evaluation": int(best.evaluation),
        "final_objective": float(final_total.detach().cpu()),
        "final_components": {
            name: float(final_components[name].detach().cpu())
            for name in (
                "data",
                "smooth",
                "velocity",
                "moment",
                "temporal",
                "jacobian",
            )
        },
        "initial_metrics": initial_metrics,
        "metrics": metrics,
        "left_numerical_diagnostics": diagnostics,
        "outputs": output_paths,
        "history_csv": os.path.abspath(history_path),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary_path, summary)

    print("\nOptimization finished.")
    print(f"Best objective: {best.value:.6e} ({best.stage}, eval {best.evaluation})")
    print(f"Endpoint metrics: {metrics}")
    print(
        "Left numerical diagnostics: "
        f"detF_min={diagnostics['detF_min']:.6e}, "
        f"detF_nonpositive={diagnostics['detF_nonpositive_fraction']:.3%}, "
        f"edge_mismatch_p95={diagnostics['local_global_edge_mismatch_p95']:.6e}"
    )
    print(f"Deformed VTK: {output_paths['deformed_vtk']}")
    print(f"Optimized alpha path: {output_paths['alpha_knots_npy']}")
    print(f"Final local Jacobians: {output_paths['jacobian_npy']}")
    print(f"Summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise
