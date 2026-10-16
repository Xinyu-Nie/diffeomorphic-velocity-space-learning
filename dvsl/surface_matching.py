#!/usr/bin/env python3
"""Dual-Frame Pullback Surface Matching (DF-PSM).

The optimized variable is a time-dependent moving-support Eulerian control

    p(t) = U_R beta(t),
    K_mix = sum_s w_s K_{r_s},
    u_t(y) = sum_j K_mix(y, q_j(t)) p_j(t),

and the deformation is integrated as one ambient flow,

    qdot_i = u_t(q_i),        Jdot_i = Du_t(q_i) J_i.

At selected trajectory frames, the realized material/body velocity

    b_i(t) = J_i(t)^{-1} u_t(q_i(t))

is projected analytically onto the fixed-template Mercer space

    b_hat(t) = U_D Lambda_D alpha(t).

The component outside that space is the pullback-compressibility loss.  Thus
the saved alpha path is not an independent deformation decoder and is never
used to move the surface; it is the fixed-domain coordinate representation of
the Eulerian flow that actually performs matching.

This implements the beta-primary version of the pullback-consistent matching
formulation.  It is non-geodesic unless a Hamiltonian-defect regularizer is
added separately: beta(t) is freely time dependent and no EPDiff momentum
equation is imposed.
"""

from __future__ import annotations

from .metrics import symmetric_squared_chamfer, AreaSampledSurfaceChamfer

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .surface_io import (
    MeshData,
    endpoint_metrics,
    load_mercer_basis,
    load_mesh,
    smoothness_loss,
)
from .metrics import VarifoldLoss
from .kernels import (
    build_radius_graph,
    wendland_phi_dphi,
)
from .surface_quality import mesh_quality_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"Non-finite {name}.")


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def interpolate_knots(values: torch.Tensor, t: float) -> torch.Tensor:
    if values.shape[0] == 1:
        return values[0]
    position = min(max(float(t), 0.0), 1.0) * (values.shape[0] - 1)
    lower = min(int(math.floor(position)), values.shape[0] - 1)
    upper = min(lower + 1, values.shape[0] - 1)
    weight = position - lower
    return (1.0 - weight) * values[lower] + weight * values[upper]


def resample_knots(values: torch.Tensor, count: int) -> torch.Tensor:
    if values.shape[0] == count:
        return values
    return torch.stack(
        [interpolate_knots(values, i / float(max(1, count - 1))) for i in range(count)]
    )


def resolve_temporal_grid(args: argparse.Namespace) -> Tuple[int, int, str]:
    """Resolve primary intervals/samples while retaining legacy CLI behavior."""
    if args.time_intervals is not None:
        if args.time_intervals <= 0:
            raise ValueError("--time_intervals must be positive.")
        intervals = int(args.time_intervals)
        return intervals, intervals + 1, "time_intervals"
    if args.time_knots is not None:
        if args.time_knots < 2:
            raise ValueError("Legacy --time_knots must provide at least two samples.")
        samples = int(args.time_knots)
        print(
            "WARNING: --time_knots is deprecated and retains its legacy meaning "
            "of temporal sample count. Use --time_intervals M for M+1 samples.",
            flush=True,
        )
        return samples - 1, samples, "legacy_time_knots"
    return 30, 31, "default_time_intervals"


def resample_flow_path(
    path: "FlowPath",
    beta: torch.Tensor,
    U_control: torch.Tensor,
    radii: Sequence[float],
    weights: Sequence[float],
    kernel: str,
    max_neighbors: int,
    recompute_velocity: bool,
) -> "FlowPath":
    """Sample a dense integrated path on the primary beta/alpha knot grid."""
    sample_count = int(beta.shape[0])
    if sample_count == path.q.shape[0]:
        return FlowPath(
            q=path.q, J=path.J, velocity=path.velocity, control=path.control
        )
    q = resample_knots(path.q, sample_count)
    J = resample_knots(path.J, sample_count)
    control = torch.einsum("vr,trc->tvc", U_control, beta)
    if recompute_velocity:
        velocities = []
        controls = []
        for frame in range(sample_count):
            velocity, _, realized_control = moving_field_and_jacobian(
                q[frame],
                beta[frame],
                U_control,
                radii,
                weights,
                kernel,
                max_neighbors,
            )
            velocities.append(velocity)
            controls.append(realized_control)
        velocity = torch.stack(velocities)
        control = torch.stack(controls)
    else:
        velocity = resample_knots(path.velocity, sample_count)
    return FlowPath(q=q, J=J, velocity=velocity, control=control)


def load_control_initialization(
    beta_path: Optional[str],
    alpha_path: Optional[str],
    time_knots: int,
    control_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    path = beta_path or alpha_path
    if not path:
        return torch.zeros((time_knots, control_dim, 3), dtype=dtype, device=device)
    resolved = os.path.abspath(os.path.expanduser(path))
    value = np.load(resolved)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 2 and value.shape[-1] == 3:
        value = value[None]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            f"Initialization must have shape (T,D,3), got {value.shape}: {resolved}"
        )
    source = torch.as_tensor(value, dtype=dtype, device=device)
    source = resample_knots(source, time_knots)
    result = torch.zeros((time_knots, control_dim, 3), dtype=dtype, device=device)
    shared = min(control_dim, source.shape[1])
    result[:, :shared] = source[:, :shared]
    finite_or_raise("control initialization", result)
    return result


def resolve_kernel_mixture(
    radius: Optional[float],
    radii: Optional[Sequence[float]],
    weights: Optional[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    """Resolve one legacy radius or a normalized unified mixed kernel."""
    resolved_radii = (
        [float(radius)] if radii is None and radius is not None
        else [float(value) for value in (radii or [])]
    )
    if not resolved_radii or any(value <= 0.0 for value in resolved_radii):
        raise ValueError("At least one strictly positive RKHS radius is required.")
    if weights is None:
        resolved_weights = [1.0 / len(resolved_radii)] * len(resolved_radii)
    else:
        resolved_weights = [float(value) for value in weights]
        if len(resolved_weights) != len(resolved_radii):
            raise ValueError("--weights must have the same length as --radii.")
        if any(value <= 0.0 for value in resolved_weights):
            raise ValueError("All mixed-kernel weights must be strictly positive.")
        total = sum(resolved_weights)
        resolved_weights = [value / total for value in resolved_weights]
    return resolved_radii, resolved_weights


def validate_basis_metadata(
    metadata: Dict,
    radii: Sequence[float],
    weights: Sequence[float],
    kernel: str,
    allow_mismatch: bool,
) -> None:
    """Require the precomputed Mercer operator to match the realized kernel."""
    messages: List[str] = []
    stored_radii = metadata.get("mixed_radii")
    stored_weights = metadata.get("mixed_weights")
    if stored_radii is None and len(radii) == 1 and metadata.get("R") is not None:
        stored_radii = [metadata["R"]]
        stored_weights = [1.0]
    if stored_radii is None or stored_weights is None:
        messages.append("basis metadata does not identify its kernel radii/weights")
    else:
        stored_radii = list(stored_radii)
        stored_weights = list(stored_weights)
        if len(stored_radii) != len(radii) or any(
            not math.isclose(float(a), float(b), rel_tol=1.0e-6, abs_tol=1.0e-8)
            for a, b in zip(stored_radii, radii)
        ):
            messages.append(
                f"radii differ: stored={stored_radii}, requested={list(radii)}"
            )
        if len(stored_weights) != len(weights) or any(
            not math.isclose(float(a), float(b), rel_tol=1.0e-6, abs_tol=1.0e-8)
            for a, b in zip(stored_weights, weights)
        ):
            messages.append(
                f"weights differ: stored={stored_weights}, requested={list(weights)}"
            )
    stored_kernel = metadata.get("kernel")
    if stored_kernel is None:
        messages.append("basis metadata does not identify its kernel type")
    elif str(stored_kernel) != str(kernel):
        messages.append(f"kernel differs: stored={stored_kernel}, requested={kernel}")
    if messages:
        message = "Unified basis mismatch: " + "; ".join(messages)
        if not allow_mismatch:
            raise ValueError(
                message
                + ". Regenerate the basis with ComputeUnifiedMultiRKHSMercerBasis.py "
                "or pass --allow_basis_mismatch."
            )
        print("WARNING: " + message, flush=True)


def moving_field_and_jacobian(
    q: torch.Tensor,
    beta_t: torch.Tensor,
    U_control: torch.Tensor,
    radii: Sequence[float],
    weights: Sequence[float],
    kernel: str,
    max_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mixed-kernel u(q), Du(q), and the shared p=U_R beta."""
    if len(radii) != len(weights):
        raise ValueError("Kernel radii and weights must have the same length.")
    p = U_control @ beta_t
    velocity = torch.zeros_like(q)
    jacobian = torch.zeros(
        (q.shape[0], 3, 3), dtype=q.dtype, device=q.device
    )
    eps = torch.finfo(q.dtype).eps
    for radius, weight in zip(radii, weights):
        edges = build_radius_graph(q.detach(), float(radius), max_neighbors)
        if edges.numel() == 0:
            continue
        row, col = edges[0], edges[1]
        difference = q[row] - q[col]
        distance = torch.linalg.norm(difference, dim=-1)
        radial = distance / float(radius)
        phi, dphi = wendland_phi_dphi(radial, kernel)
        p_edge = p[col]
        velocity.index_add_(0, row, float(weight) * phi[:, None] * p_edge)

        radial_derivative = dphi / (
            float(radius) * torch.clamp(distance, min=eps)
        )
        grad_kernel = radial_derivative[:, None] * difference
        jacobian_edge = p_edge[:, :, None] * grad_kernel[:, None, :]
        jacobian.index_add_(0, row, float(weight) * jacobian_edge)
    return velocity, jacobian, p


@dataclass
class FlowPath:
    q: torch.Tensor
    J: torch.Tensor
    velocity: torch.Tensor
    control: torch.Tensor


def integrate_eulerian_path(
    source: torch.Tensor,
    beta: torch.Tensor,
    U_control: torch.Tensor,
    n_steps: int,
    radii: Sequence[float],
    weights: Sequence[float],
    kernel: str,
    max_neighbors: int,
    integrator: str,
) -> FlowPath:
    """Integrate q and the same-flow deformation Jacobian J."""
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    J = identity.expand(source.shape[0], -1, -1).clone()
    q = source
    q_frames: List[torch.Tensor] = [q]
    J_frames: List[torch.Tensor] = [J]
    velocity_frames: List[torch.Tensor] = []
    control_frames: List[torch.Tensor] = []
    dt = 1.0 / float(n_steps)

    for step in range(n_steps):
        t0 = step * dt
        tm = t0 + 0.5 * dt
        t1 = t0 + dt
        beta0 = interpolate_knots(beta, t0)
        velocity0, Du0, p0 = moving_field_and_jacobian(
            q, beta0, U_control, radii, weights, kernel, max_neighbors
        )
        dJ0 = torch.einsum("vij,vjk->vik", Du0, J)
        velocity_frames.append(velocity0)
        control_frames.append(p0)

        if integrator == "euler":
            q = q + dt * velocity0
            J = J + dt * dJ0
        elif integrator == "midpoint":
            q_mid = q + 0.5 * dt * velocity0
            J_mid = J + 0.5 * dt * dJ0
            velocity_mid, Du_mid, _ = moving_field_and_jacobian(
                q_mid,
                interpolate_knots(beta, tm),
                U_control,
                radii,
                weights,
                kernel,
                max_neighbors,
            )
            dJ_mid = torch.einsum("vij,vjk->vik", Du_mid, J_mid)
            q = q + dt * velocity_mid
            J = J + dt * dJ_mid
        elif integrator == "rk4":
            q2 = q + 0.5 * dt * velocity0
            J2 = J + 0.5 * dt * dJ0
            velocity2, Du2, _ = moving_field_and_jacobian(
                q2,
                interpolate_knots(beta, tm),
                U_control,
                radii,
                weights,
                kernel,
                max_neighbors,
            )
            dJ2 = torch.einsum("vij,vjk->vik", Du2, J2)
            q3 = q + 0.5 * dt * velocity2
            J3 = J + 0.5 * dt * dJ2
            velocity3, Du3, _ = moving_field_and_jacobian(
                q3,
                interpolate_knots(beta, tm),
                U_control,
                radii,
                weights,
                kernel,
                max_neighbors,
            )
            dJ3 = torch.einsum("vij,vjk->vik", Du3, J3)
            q4 = q + dt * velocity3
            J4 = J + dt * dJ3
            velocity4, Du4, _ = moving_field_and_jacobian(
                q4,
                interpolate_knots(beta, t1),
                U_control,
                radii,
                weights,
                kernel,
                max_neighbors,
            )
            dJ4 = torch.einsum("vij,vjk->vik", Du4, J4)
            q = q + (dt / 6.0) * (
                velocity0 + 2.0 * velocity2 + 2.0 * velocity3 + velocity4
            )
            J = J + (dt / 6.0) * (dJ0 + 2.0 * dJ2 + 2.0 * dJ3 + dJ4)
        else:
            raise ValueError(f"Unsupported integrator: {integrator}")

        q_frames.append(q)
        J_frames.append(J)

    velocity_T, _, control_T = moving_field_and_jacobian(
        q, beta[-1], U_control, radii, weights, kernel, max_neighbors
    )
    velocity_frames.append(velocity_T)
    control_frames.append(control_T)
    return FlowPath(
        q=torch.stack(q_frames),
        J=torch.stack(J_frames),
        velocity=torch.stack(velocity_frames),
        control=torch.stack(control_frames),
    )


def projection_indices(frame_count: int, stride: int) -> List[int]:
    indices = list(range(0, frame_count, max(1, int(stride))))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


@dataclass
class PullbackProjection:
    alpha: torch.Tensor
    body: torch.Tensor
    projected: torch.Tensor
    residual: torch.Tensor
    frame_indices: List[int]


def project_template_pullback(
    path: FlowPath,
    U_body: torch.Tensor,
    lam_body: torch.Tensor,
    stride: int,
    solve_ridge: float,
) -> PullbackProjection:
    indices = projection_indices(path.q.shape[0], stride)
    J = path.J[indices]
    velocity = path.velocity[indices]
    if solve_ridge > 0.0:
        identity = torch.eye(3, dtype=J.dtype, device=J.device)
        J = J + float(solve_ridge) * identity.view(1, 1, 3, 3)
    body = torch.linalg.solve(J, velocity.unsqueeze(-1)).squeeze(-1)
    coefficients = torch.einsum("vd,tvc->tdc", U_body, body)
    alpha = coefficients / lam_body.view(1, -1, 1)
    projected = torch.einsum("vd,tdc->tvc", U_body, coefficients)
    residual = body - projected
    return PullbackProjection(alpha, body, projected, residual, indices)


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def edge_distortion_loss(
    q_path: torch.Tensor,
    source: torch.Tensor,
    edges: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    if edges.numel() == 0:
        return q_path.new_zeros(())
    baseline = torch.linalg.norm(
        source[edges[:, 0]] - source[edges[:, 1]], dim=-1
    ).clamp_min(torch.finfo(source.dtype).eps)
    values = []
    for index in projection_indices(q_path.shape[0], stride)[1:]:
        current = torch.linalg.norm(
            q_path[index, edges[:, 0]] - q_path[index, edges[:, 1]], dim=-1
        )
        values.append(torch.log(current.clamp_min(1.0e-8) / baseline).pow(2).mean())
    return torch.stack(values).mean() if values else q_path.new_zeros(())


def validate_pointwise_pair(source: MeshData, target: MeshData, strict: bool) -> None:
    if source.points.shape[0] != target.points.shape[0]:
        raise ValueError("Pointwise loss requires equal source and target vertex counts.")
    if strict and (
        source.faces.shape != target.faces.shape
        or not torch.equal(source.faces, target.faces)
    ):
        raise ValueError("--strict_pointwise_topology requires identical connectivity.")


class DualFramePullbackObjective(nn.Module):
    def __init__(
        self,
        source: torch.Tensor,
        faces: torch.Tensor,
        target: torch.Tensor,
        U_control: torch.Tensor,
        lam_control: torch.Tensor,
        U_body: torch.Tensor,
        lam_body: torch.Tensor,
        varifold_loss: Optional[VarifoldLoss],
        args: argparse.Namespace,
        endpoint_attachment: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("source", source.detach())
        self.register_buffer("faces", faces)
        self.register_buffer("target", target.detach())
        self.register_buffer("U_control", U_control.detach())
        self.register_buffer("lam_control", lam_control.detach())
        self.register_buffer("U_body", U_body.detach())
        self.register_buffer("lam_body", lam_body.detach())
        self.register_buffer("edges", unique_edges(faces))
        self.endpoint_attachment = endpoint_attachment
        self.varifold_loss = varifold_loss
        self.args = args

    def forward(self, beta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        def integrate(beta_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            integrated = integrate_eulerian_path(
                self.source,
                beta_input,
                self.U_control,
                self.args.n_steps,
                self.args.radii_resolved,
                self.args.weights_resolved,
                self.args.kernel,
                self.args.max_neighbors,
                self.args.integrator,
            )
            return (
                integrated.q,
                integrated.J,
                integrated.velocity,
                integrated.control,
            )

        if self.args.checkpoint_path and torch.is_grad_enabled():
            tensors = checkpoint(
                integrate,
                beta,
                use_reentrant=False,
                determinism_check="none",
            )
        else:
            tensors = integrate(beta)
        path = FlowPath(*tensors)
        finite_or_raise("trajectory", path.q)
        finite_or_raise("trajectory Jacobian", path.J)

        if self.args.loss == "pointwise":
            data = (path.q[-1] - self.target).pow(2).mean()
        elif self.args.loss == "chamfer":
            data = symmetric_squared_chamfer(path.q[-1], self.target)
        elif self.args.loss == "surface_chamfer":
            if self.endpoint_attachment is None:
                raise RuntimeError("Surface Chamfer attachment was not initialized.")
            data = self.endpoint_attachment(path.q[-1])
        elif self.args.loss == "varifold":
            if self.varifold_loss is None:
                raise RuntimeError("Varifold loss was not initialized.")
            data = self.varifold_loss(path.q[-1])
        else:
            raise ValueError(f"Unsupported loss: {self.args.loss}")
        smooth = smoothness_loss(path.q[-1] - self.source, self.faces)
        action_frames = 0.5 * (path.control * path.velocity).sum(dim=(1, 2))
        action = action_frames.mean() / float(self.source.shape[0])

        primary_path = resample_flow_path(
            path,
            beta,
            self.U_control,
            self.args.radii_resolved,
            self.args.weights_resolved,
            self.args.kernel,
            self.args.max_neighbors,
            recompute_velocity=True,
        )
        # Equation (92) is evaluated on the same primary temporal grid as beta.
        projection = project_template_pullback(
            primary_path,
            self.U_body,
            self.lam_body,
            1,
            self.args.pullback_solve_ridge,
        )
        pullback_indices = projection_indices(
            projection.residual.shape[0], self.args.pullback_stride
        )
        pullback = projection.residual[pullback_indices].pow(2).mean()
        body_power = projection.body[pullback_indices].pow(2).mean()
        pullback_relative = pullback / torch.clamp(body_power, min=1.0e-12)

        alpha_energy = (
            projection.alpha.pow(2) * self.lam_body.view(1, -1, 1)
        ).sum() / float(
            max(1, projection.alpha.shape[0] * self.source.shape[0] * 3)
        )
        if projection.alpha.shape[0] > 1:
            alpha_delta = projection.alpha[1:] - projection.alpha[:-1]
            alpha_time = (
                alpha_delta.pow(2) * self.lam_body.view(1, -1, 1)
            ).mean()
        else:
            alpha_time = beta.new_zeros(())
        if beta.shape[0] > 1:
            beta_time = (
                (beta[1:] - beta[:-1]).pow(2)
                * self.lam_control.view(1, -1, 1)
            ).mean()
        else:
            beta_time = beta.new_zeros(())

        geometry_indices = projection_indices(
            path.J.shape[0], self.args.geometry_stride
        )
        detJ = torch.linalg.det(path.J[geometry_indices])
        jacobian = torch.relu(float(self.args.jacobian_floor) - detJ).pow(2).mean()
        edge = edge_distortion_loss(
            path.q,
            self.source,
            self.edges,
            self.args.geometry_stride,
        )
        total = (
            self.args.lambda_rec * data
            + self.args.lambda_smooth * smooth
            + self.args.lambda_action * action
            + self.args.lambda_pullback * pullback
            + self.args.lambda_alpha_energy * alpha_energy
            + self.args.lambda_alpha_time * alpha_time
            + self.args.lambda_beta_time * beta_time
            + self.args.lambda_jacobian * jacobian
            + self.args.lambda_edge * edge
        )
        components = {
            "total": total,
            "data": data,
            "smooth": smooth,
            "action": action,
            "pullback": pullback,
            "pullback_relative": pullback_relative,
            "alpha_energy": alpha_energy,
            "alpha_time": alpha_time,
            "beta_time": beta_time,
            "jacobian": jacobian,
            "edge": edge,
            "detJ_min": detJ.min(),
            "endpoint": path.q[-1],
        }
        return total, components


@dataclass
class BestState:
    value: float = math.inf
    beta: Optional[torch.Tensor] = None
    stage: str = ""
    evaluation: int = -1


def optimize_eulerian_controls(
    objective: DualFramePullbackObjective,
    initial_beta: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[Dict], BestState]:
    beta = nn.Parameter(initial_beta.clone())
    best = BestState()
    history: List[Dict] = []
    started = time.perf_counter()
    evaluation = 0

    def evaluate(stage: str, clip: bool) -> torch.Tensor:
        nonlocal evaluation, best
        loss, components = objective(beta)
        finite_or_raise("objective", loss)
        loss.backward()
        if beta.grad is None:
            raise RuntimeError("No beta gradient was produced.")
        finite_or_raise("beta gradient", beta.grad)
        if clip and args.grad_clip > 0.0:
            raw_norm = torch.nn.utils.clip_grad_norm_([beta], args.grad_clip)
            grad_norm = float(raw_norm.detach().cpu())
        else:
            grad_norm = float(beta.grad.norm().detach().cpu())
        value = float(loss.detach().cpu())
        if value < best.value:
            best = BestState(value, beta.detach().clone(), stage, evaluation)
        row = {
            "stage": stage,
            "evaluation": evaluation,
            "elapsed_sec": time.perf_counter() - started,
            "grad_norm": grad_norm,
        }
        for key in (
            "total", "data", "smooth", "action", "pullback",
            "pullback_relative", "alpha_energy", "alpha_time", "beta_time",
            "jacobian", "edge", "detJ_min",
        ):
            row[key] = float(components[key].detach().cpu())
        history.append(row)
        if evaluation % max(1, args.log_every) == 0:
            print(
                f"[{stage:7s} eval {evaluation:05d}] total={row['total']:.6e} "
                f"rec={row['data']:.6e} pb={row['pullback']:.3e} "
                f"pb_rel={row['pullback_relative']:.3e} action={row['action']:.3e} "
                f"detmin={row['detJ_min']:.3e} |grad|={row['grad_norm']:.3e}",
                flush=True,
            )
        evaluation += 1
        return loss

    if args.adam_iters > 0:
        optimizer = torch.optim.Adam(
            [beta], lr=args.adam_lr, weight_decay=args.adam_weight_decay
        )
        checkpoint_every = getattr(args, "checkpoint_every", 0)
        checkpoint_path = Path(args.output_dir) / "adam_checkpoint.pt"
        ignored = {"resume_checkpoint", "checkpoint_every", "output_dir", "log_every"}
        checkpoint_config = {k: v for k, v in vars(args).items() if k not in ignored}
        first_iteration = 0
        if getattr(args, "resume_checkpoint", False) and checkpoint_path.is_file():
            # Locally generated optimizer state, including RNG state.
            saved = torch.load(checkpoint_path, map_location=beta.device, weights_only=False)
            if saved["config"] != checkpoint_config:
                raise ValueError("Adam checkpoint configuration differs from this run.")
            first_iteration = int(saved["completed_iterations"])
            if not 0 <= first_iteration <= args.adam_iters:
                raise ValueError("Invalid completed iteration count in Adam checkpoint.")
            with torch.no_grad():
                beta.copy_(saved["beta"])
            optimizer.load_state_dict(saved["optimizer"])
            best = BestState(**saved["best"])
            history = saved["history"]
            evaluation = saved["evaluation"]
            started = time.perf_counter() - saved["elapsed_seconds"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if beta.is_cuda:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
            print(f"Resuming Adam after {first_iteration} completed iterations.", flush=True)
        for iteration in range(first_iteration, args.adam_iters):
            optimizer.zero_grad(set_to_none=True)
            evaluate("adam", clip=True)
            optimizer.step()
            if checkpoint_every > 0 and (
                (iteration + 1) % checkpoint_every == 0 or iteration + 1 == args.adam_iters
            ):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                torch.save({
                    "config": checkpoint_config,
                    "completed_iterations": iteration + 1,
                    "beta": beta.detach(),
                    "optimizer": optimizer.state_dict(),
                    "best": vars(best),
                    "history": history,
                    "evaluation": evaluation,
                    "elapsed_seconds": time.perf_counter() - started,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if beta.is_cuda else [],
                }, temporary_path)
                os.replace(temporary_path, checkpoint_path)
        optimizer.zero_grad(set_to_none=True)
        evaluate("adam_final", clip=False)

    if args.lbfgs_iters > 0:
        optimizer = torch.optim.LBFGS(
            [beta],
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_iters,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            return evaluate("lbfgs", clip=False)

        optimizer.step(closure)

    if best.beta is None:
        with torch.enable_grad():
            beta.grad = None
            evaluate("initial", clip=False)
    assert best.beta is not None
    return best.beta, history, best


def write_history(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def final_projection_all_frames(
    path: FlowPath,
    U_body: torch.Tensor,
    lam_body: torch.Tensor,
    solve_ridge: float,
) -> PullbackProjection:
    return project_template_pullback(path, U_body, lam_body, 1, solve_ridge)


def save_outputs(
    output_dir: str,
    source: MeshData,
    target: MeshData,
    beta: torch.Tensor,
    dense_path: FlowPath,
    dense_projection: PullbackProjection,
    primary_path: FlowPath,
    primary_projection: PullbackProjection,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    target_stem = Path(target.path).stem
    knot_times = torch.linspace(
        0.0, 1.0, beta.shape[0], dtype=beta.dtype, device=beta.device
    )
    arrays = {
        # Primary statistical paths: beta and alpha have identical time axes.
        "knot_times.npy": knot_times,
        "beta_knots.npy": beta,
        "alpha_knots.npy": primary_projection.alpha,
        "trajectory_q_knots.npy": primary_path.q,
        "trajectory_J_knots.npy": primary_path.J,
        "trajectory_velocity_knots.npy": primary_path.velocity,
        "trajectory_control_p_knots.npy": primary_path.control,
        "body_velocity_knots.npy": primary_projection.body,
        "body_projected_knots.npy": primary_projection.projected,
        "pullback_residual_knots.npy": primary_projection.residual,
        # Dense fixed-step integration diagnostics (legacy names retained).
        "trajectory_q.npy": dense_path.q,
        "trajectory_J.npy": dense_path.J,
        "trajectory_velocity.npy": dense_path.velocity,
        "trajectory_control_p.npy": dense_path.control,
        "trajectory_alpha_dense.npy": dense_projection.alpha,
        "trajectory_body_velocity.npy": dense_projection.body,
        "trajectory_body_projected.npy": dense_projection.projected,
        "trajectory_pullback_residual.npy": dense_projection.residual,
    }
    paths: Dict[str, str] = {}
    for name, tensor in arrays.items():
        destination = os.path.join(output_dir, name)
        np.save(destination, tensor.detach().cpu().numpy().astype(np.float32))
        paths[name.removesuffix(".npy")] = os.path.abspath(destination)

    mesh = pv.PolyData(
        dense_path.q[-1].detach().cpu().numpy().astype(np.float64),
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    )
    vtk_path = os.path.join(
        output_dir, f"{target_stem}_pullback_matched_deformed.vtk"
    )
    mesh.save(vtk_path)
    paths["deformed_vtk"] = os.path.abspath(vtk_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs=2, metavar=("SOURCE", "TARGET"), required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--body_basis_dim", "--pullback_rank", "--basis_dim", type=int, default=512)
    parser.add_argument(
        "--control_basis_dim", "--control_rank",
        type=int,
        default=0,
        help="Eulerian control modes; 0 uses body_basis_dim.",
    )
    kernel_group = parser.add_mutually_exclusive_group(required=True)
    kernel_group.add_argument(
        "--radius", "--kernel_radius",
        type=float,
        help="Legacy single Wendland support radius.",
    )
    kernel_group.add_argument(
        "--radii",
        "--shooting_radii",
        dest="radii",
        nargs="+",
        type=float,
        help="One or more support radii for a unified mixed kernel.",
    )
    parser.add_argument(
        "--weights",
        "--shooting_weights",
        dest="weights",
        nargs="+",
        type=float,
        default=None,
        help="Positive mixed-kernel weights; normalized to sum to one.",
    )
    parser.add_argument("--kernel", choices=["c2", "c4"], default="c2")
    parser.add_argument(
        "--allow_basis_mismatch",
        action="store_true",
        help="Allow basis metadata to differ from the realized kernel (diagnostics only).",
    )
    temporal_group = parser.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--time_intervals",
        type=int,
        default=None,
        help="Primary intervals on [0,1]; allocates time_intervals+1 beta/alpha samples.",
    )
    temporal_group.add_argument(
        "--time_knots", "--control_knots",
        type=int,
        default=None,
        help="Deprecated legacy sample count; use --time_intervals.",
    )
    parser.add_argument("--n_steps", "--integration_steps", type=int, default=30)
    parser.add_argument("--integrator", choices=["euler", "midpoint", "rk4"], default="midpoint")
    parser.add_argument("--max_neighbors", type=int, default=512)
    parser.add_argument(
        "--checkpoint_path",
        action="store_true",
        help="Recompute the integrated trajectory during backward to reduce GPU memory.",
    )
    parser.add_argument("--pullback_stride", type=int, default=2)
    parser.add_argument("--geometry_stride", type=int, default=3)
    parser.add_argument("--pullback_solve_ridge", "--pullback_ridge", type=float, default=1.0e-6)
    parser.add_argument("--jacobian_floor", type=float, default=0.05)

    parser.add_argument("--metric", "--loss", dest="loss", choices=["pointwise", "varifold", "chamfer", "surface_chamfer"], default="pointwise")
    parser.add_argument("--varifold_backend", choices=["tiled", "stable"], default="tiled")
    parser.add_argument("--chamfer_source_samples", type=int, default=4096)
    parser.add_argument("--chamfer_target_samples", type=int, default=4096)
    parser.add_argument("--chamfer_chunk", type=int, default=1024)
    parser.add_argument("--chamfer_seed", type=int, default=None)
    parser.add_argument("--strict_pointwise_topology", action="store_true")
    parser.add_argument(
        "--varifold_radius",
        type=float,
        default=None,
        help="Varifold discrepancy radius; defaults to the largest deformation-kernel radius.",
    )
    parser.add_argument(
        "--varifold_kernel",
        choices=["c2", "c4"],
        default=None,
        help="Varifold positional kernel; defaults to --kernel.",
    )
    parser.add_argument("--var_Ti", type=int, default=1024)
    parser.add_argument("--var_Tj", type=int, default=2048)
    parser.add_argument("--var_amp", action="store_true")
    parser.add_argument("--var_detach_geom", action="store_true")
    parser.add_argument(
        "--var_face_max",
        type=int,
        default=None,
        help="Optional deterministic face cap for each surface in the varifold loss.",
    )
    parser.add_argument(
        "--no_var_normalize_areas",
        dest="var_normalize_areas",
        action="store_false",
        help="Disable separate unit-area normalization of source and target.",
    )
    parser.set_defaults(var_normalize_areas=True)
    parser.add_argument("--lambda_rec", "--attachment_weight", type=float, default=1000.0)
    parser.add_argument("--lambda_smooth", "--smoothness_weight", type=float, default=0.01)
    parser.add_argument("--lambda_action", "--action_weight", type=float, default=1.0e-4)
    parser.add_argument("--lambda_pullback", "--pullback_weight", type=float, default=1.0)
    parser.add_argument("--lambda_alpha_energy", "--body_energy_weight", type=float, default=0.0)
    parser.add_argument("--lambda_alpha_time", "--body_temporal_weight", type=float, default=0.0)
    parser.add_argument("--lambda_beta_time", "--control_temporal_weight", type=float, default=1.0e-3)
    parser.add_argument("--lambda_jacobian", "--jacobian_weight", type=float, default=0.1)
    parser.add_argument("--lambda_edge", "--edge_weight", type=float, default=0.01)

    parser.add_argument("--init_beta", default="")
    parser.add_argument(
        "--init_alpha",
        default="",
        help="Warm-start beta by copying shared alpha modes; this is initialization only.",
    )
    parser.add_argument(
        "--ablation_zero_init",
        action="store_true",
        help="Require zero initialization and reject --init_beta/--init_alpha.",
    )
    parser.add_argument("--adam_iters", type=int, default=200)
    parser.add_argument("--adam_lr", type=float, default=1.0e-3)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Atomically save Adam state every N iterations; 0 disables.")
    parser.add_argument("--resume_checkpoint", action="store_true",
                        help="Resume a matching local Adam checkpoint in output_dir.")
    parser.add_argument("--lbfgs_iters", type=int, default=0)
    parser.add_argument("--lbfgs_lr", type=float, default=0.5)
    parser.add_argument("--lbfgs_history_size", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_exact_si", action="store_true")
    return parser



def validate_arguments(args: argparse.Namespace) -> None:
    """Reject ambiguous/invalid metric controls before reading meshes or writing outputs."""
    for name in ("n_steps", "max_neighbors", "body_basis_dim", "pullback_stride", "geometry_stride",
                 "var_Ti", "var_Tj", "chamfer_source_samples", "chamfer_target_samples", "chamfer_chunk"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    resolve_temporal_grid(args)
    resolve_kernel_mixture(args.radius, args.radii, args.weights)
    for name in ("radius", "varifold_radius"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name} must be finite and positive.")
    for name in ("radii", "weights"):
        values = getattr(args, name)
        if values is not None and any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"--{name} values must be finite and positive.")
    if args.adam_iters < 0 or args.lbfgs_iters < 0 or args.checkpoint_every < 0:
        raise ValueError("Iteration counts and checkpoint interval cannot be negative.")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name} must be finite.")
        if name.startswith("lambda_") and value < 0:
            raise ValueError(f"--{name} cannot be negative.")
    if args.adam_lr <= 0 or args.lbfgs_lr <= 0 or args.adam_weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative.")
    if args.pullback_solve_ridge < 0:
        raise ValueError("--pullback_solve_ridge cannot be negative.")
    if args.loss == "varifold" and args.varifold_backend == "stable" and args.var_amp:
        raise ValueError("Stable varifold requires full precision; omit --var_amp.")
    if args.strict_pointwise_topology and args.loss != "pointwise":
        raise ValueError("--strict_pointwise_topology only applies to pointwise attachment.")
    if args.init_beta and args.init_alpha:
        raise ValueError("Choose only one of --init_beta and --init_alpha.")
    if args.ablation_zero_init and (args.init_beta or args.init_alpha):
        raise ValueError("--ablation_zero_init forbids warm starts.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    radii, weights = resolve_kernel_mixture(args.radius, args.radii, args.weights)
    args.radii_resolved = radii
    args.weights_resolved = weights
    intervals, samples, temporal_source = resolve_temporal_grid(args)
    args.time_intervals_resolved = intervals
    args.time_samples_resolved = samples
    args.temporal_grid_source = temporal_source
    if args.n_steps <= 0:
        raise ValueError("--n_steps must be positive.")
    if args.adam_iters < 0 or args.lbfgs_iters < 0:
        raise ValueError("Optimizer iteration counts cannot be negative.")
    if args.ablation_zero_init and (args.init_beta or args.init_alpha):
        raise ValueError(
            "--ablation_zero_init forbids --init_beta and --init_alpha."
        )

    source = load_mesh(args.pairs[0], device, dtype)
    target = load_mesh(args.pairs[1], device, dtype)
    if args.loss == "pointwise":
        validate_pointwise_pair(source, target, args.strict_pointwise_topology)
    requested_control_dim = (
        args.body_basis_dim if args.control_basis_dim <= 0 else args.control_basis_dim
    )
    loaded_dim = max(args.body_basis_dim, requested_control_dim)
    U_all, lam_all, basis_meta = load_mercer_basis(
        args.basis, loaded_dim, device, dtype
    )
    validate_basis_metadata(
        basis_meta, radii, weights, args.kernel, args.allow_basis_mismatch
    )
    if U_all.shape[0] != source.points.shape[0]:
        raise ValueError(
            f"Basis has {U_all.shape[0]} vertices but source has {source.points.shape[0]}."
        )
    if args.body_basis_dim <= 0 or requested_control_dim <= 0:
        raise ValueError("Body and control basis dimensions must be positive.")
    if U_all.shape[1] < loaded_dim:
        raise ValueError(
            f"Basis provides {U_all.shape[1]} usable modes, but {loaded_dim} are required."
        )
    body_dim = args.body_basis_dim
    control_dim = requested_control_dim
    U_body, lam_body = U_all[:, :body_dim], lam_all[:body_dim]
    U_control, lam_control = U_all[:, :control_dim], lam_all[:control_dim]
    initial_beta = load_control_initialization(
        args.init_beta or None,
        args.init_alpha or None,
        args.time_samples_resolved,
        control_dim,
        device,
        dtype,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    varifold_radius = (
        max(radii) if args.varifold_radius is None else float(args.varifold_radius)
    )
    varifold_kernel = args.kernel if args.varifold_kernel is None else args.varifold_kernel
    varifold_module: Optional[VarifoldLoss] = None
    if args.loss == "varifold":
        varifold_module = VarifoldLoss(
            source_faces=source.faces,
            target_points=target.points,
            target_faces=target.faces,
            radius=varifold_radius,
            kind=varifold_kernel,
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
            f"radius={varifold_radius}, kernel={varifold_kernel}",
            flush=True,
        )

    endpoint_attachment = None
    if args.loss == "surface_chamfer":
        endpoint_attachment = AreaSampledSurfaceChamfer(
            source.points, source.faces, target.points, target.faces,
            args.chamfer_source_samples, args.chamfer_target_samples,
            args.chamfer_chunk, args.seed if args.chamfer_seed is None else args.chamfer_seed,
        ).to(device)

    configuration = vars(args).copy()
    configuration["method"] = "Dual-Frame Pullback Surface Matching (DF-PSM)"
    configuration["metric"] = args.loss
    configuration.update(
        {
            "source_vtk": source.path,
            "target_vtk": target.path,
            "device_resolved": str(device),
            "body_basis_dim_resolved": body_dim,
            "control_basis_dim_resolved": control_dim,
            "rkhs_radii_resolved": radii,
            "rkhs_weights_resolved": weights,
            "time_intervals": int(args.time_intervals_resolved),
            "time_samples": int(args.time_samples_resolved),
            # Compatibility: time_knots now records the resolved sample count.
            "varifold_radius_resolved": varifold_radius,
            "varifold_kernel_resolved": varifold_kernel,
            "time_knots": int(args.time_samples_resolved),
            "temporal_grid_source": args.temporal_grid_source,
            "optimization_variable_count": int(
                args.time_samples_resolved * control_dim * 3
            ),
            "initialization_resolved": (
                "zero" if not (args.init_beta or args.init_alpha) else "warm_start"
            ),
            "basis_meta": basis_meta,
        }
    )
    save_json(os.path.join(args.output_dir, "config.json"), configuration)
    np.save(
        os.path.join(args.output_dir, "initial_beta_knots.npy"),
        initial_beta.detach().cpu().numpy().astype(np.float32),
    )

    print("\nDual-Frame Pullback Surface Matching (DF-PSM)", flush=True)
    print(f"  source: {source.path}", flush=True)
    print(f"  target: {target.path}", flush=True)
    print(f"  loss: {args.loss}", flush=True)
    print(f"  device/dtype: {device}/{dtype}", flush=True)
    print(
        f"  vertices={source.points.shape[0]} body_D={body_dim} control_R={control_dim} "
        f"intervals={args.time_intervals_resolved} "
        f"samples={args.time_samples_resolved} integration_steps={args.n_steps}",
        flush=True,
    )

    print(f"  RKHS radii/weights: {radii}/{weights}", flush=True)
    objective = DualFramePullbackObjective(
        source.points,
        source.faces,
        target.points,
        U_control,
        lam_control,
        U_body,
        lam_body,
        varifold_module,
        args,
        endpoint_attachment=endpoint_attachment,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimized_beta, history, best = optimize_eulerian_controls(objective, initial_beta, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
    history_path = os.path.join(args.output_dir, "optimization_history.csv")
    write_history(history_path, history)

    with torch.no_grad():
        final_path = integrate_eulerian_path(
            source.points,
            optimized_beta,
            U_control,
            args.n_steps,
            radii,
            weights,
            args.kernel,
            args.max_neighbors,
            args.integrator,
        )
        dense_projection = final_projection_all_frames(
            final_path, U_body, lam_body, args.pullback_solve_ridge
        )
        primary_path = resample_flow_path(
            final_path,
            optimized_beta,
            U_control,
            radii,
            weights,
            args.kernel,
            args.max_neighbors,
            recompute_velocity=True,
        )
        projection = final_projection_all_frames(
            primary_path, U_body, lam_body, args.pullback_solve_ridge
        )
        final_total, final_components = objective(optimized_beta)
    outputs = save_outputs(
        args.output_dir,
        source,
        target,
        optimized_beta,
        final_path,
        dense_projection,
        primary_path,
        projection,
    )

    pointwise_available = args.loss == "pointwise"
    metrics = endpoint_metrics(
        final_path.q[-1], target.points, pointwise_available=pointwise_available, chunk=2048
    )
    if pointwise_available:
        metrics["pointwise_vertex_rmse"] = float(
            torch.linalg.norm(final_path.q[-1] - target.points, dim=1)
            .pow(2)
            .mean()
            .sqrt()
            .cpu()
        )
    detJ = torch.linalg.det(final_path.J)
    residual_power = dense_projection.residual.pow(2).mean(dim=(1, 2))
    body_power = dense_projection.body.pow(2).mean(dim=(1, 2)).clamp_min(1.0e-12)
    pullback_relative_by_frame = residual_power / body_power
    qc: Dict = {}
    if not args.skip_exact_si:
        qc = mesh_quality_metrics(
            final_path.q[-1].detach().cpu().numpy().astype(np.float64),
            source.faces.detach().cpu().numpy().astype(np.int64),
            compute_self_intersections=True,
        )
    summary = {
        "best_objective": best.value,
        "cuda_peak_allocated_bytes": peak_allocated_bytes,
        "cuda_peak_reserved_bytes": peak_reserved_bytes,
        "best_stage": best.stage,
        "best_evaluation": best.evaluation,
        "final_objective": float(final_total.detach().cpu()),
        "final_components": {
            key: float(value.detach().cpu())
            for key, value in final_components.items()
            if value.ndim == 0
        },
        "metrics": metrics,
        "pullback_relative_mean_all_frames": float(
            pullback_relative_by_frame.mean().cpu()
        ),
        "pullback_relative_max_all_frames": float(
            pullback_relative_by_frame.max().cpu()
        ),
        "detJ_min_all_frames": float(detJ.min().cpu()),
        "detJ_nonpositive_fraction_all_frames": float(
            (detJ <= 0.0).float().mean().cpu()
        ),
        "endpoint_qc": qc,
        "rkhs_radii": radii,
        "rkhs_weights": weights,
        "body_basis_dim": body_dim,
        "control_basis_dim": control_dim,
        "time_intervals": int(args.time_intervals_resolved),
        "time_samples": int(args.time_samples_resolved),
        "time_knots": int(args.time_samples_resolved),
        "optimization_variable_count": int(
            args.time_samples_resolved * control_dim * 3
        ),
        "elapsed_optimization_seconds": (
            float(history[-1]["elapsed_sec"]) if history else 0.0
        ),
        "outputs": outputs,
        "history_csv": os.path.abspath(history_path),
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("\nFinished.", flush=True)
    if device.type == "cuda":
        print(
            f"  CUDA peak allocated/reserved: {peak_allocated_bytes / 2**30:.3f}/"
            f"{peak_reserved_bytes / 2**30:.3f} GiB",
            flush=True,
        )

    if pointwise_available:
        print(
            f"  pointwise coordinate/vertex RMSE: "
            f"{metrics.get('pointwise_rmse', math.nan):.6e}/"
            f"{metrics.get('pointwise_vertex_rmse', math.nan):.6e}",
            flush=True,
        )
    else:
        print(
            f"  {args.loss} data loss: {float(final_components['data'].cpu()):.6e}; "
            f"endpoint/target nearest means: "
            f"{metrics['endpoint_to_target_mean']:.6e}/"
            f"{metrics['target_to_endpoint_mean']:.6e}",
            flush=True,
        )

    print(
        f"  pullback residual fraction mean/max: "
        f"{summary['pullback_relative_mean_all_frames']:.6e}/"
        f"{summary['pullback_relative_max_all_frames']:.6e}",
        flush=True,
    )
    print(
        f"  detJ min/nonpositive: {summary['detJ_min_all_frames']:.6e}/"
        f"{summary['detJ_nonpositive_fraction_all_frames']:.6e}",
        flush=True,
    )
    if qc:
        print(
            f"  endpoint SI pairs: {qc.get('self_intersection_pair_count', 'n/a')}",
            flush=True,
        )
    print(f"  deformed VTK: {outputs['deformed_vtk']}", flush=True)
    print(f"  summary: {os.path.join(os.path.abspath(args.output_dir), 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
