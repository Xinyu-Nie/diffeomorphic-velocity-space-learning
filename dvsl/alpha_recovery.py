"""Geometry-regularized sequential inverse recovery of DF-PSM beta from induced alpha.

This is a representation/inverse-recovery diagnostic, not an independent matcher.
The alpha fit drives recovery; saved beta/trajectory/target data provide diagnostics.
The original fixed-state inverse, local geometry penalties and Euler/midpoint
state updates are retained without module-global CLI parsing or monkey patches.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from .surface_matching import interpolate_knots, moving_field_and_jacobian, resample_knots, unique_edges
from .kernels import build_radius_graph, wendland_phi_dphi
from .surface_io import load_mercer_basis, load_mesh
from .surface_quality import mesh_quality_metrics
from .mercer_validation import legacy_precision
from .recovery_inputs import prepare_recovery_inputs, validate_recovery_arguments

def relative_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> float:
    return float(
        ((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(eps))
        .detach()
        .cpu()
    )


def source_scale(points: torch.Tensor) -> float:
    return float(
        torch.linalg.norm(points.max(0).values - points.min(0).values).cpu()
    )


def normalized_vertex_rmse(
    a: torch.Tensor, b: torch.Tensor, scale: float
) -> float:
    value = torch.linalg.norm(a - b, dim=-1).pow(2).mean().sqrt()
    return float(value.detach().cpu()) / max(scale, 1.0e-12)


class FixedStatePullback(nn.Module):
    """Linear beta -> alpha map with cached state and radius graph."""

    def __init__(
        self,
        q: torch.Tensor,
        J: torch.Tensor,
        U_control: torch.Tensor,
        U_body: torch.Tensor,
        lam_body: torch.Tensor,
        radii: Sequence[float],
        weights: Sequence[float],
        kernel: str,
        max_neighbors: int,
        solve_ridge: float,
    ) -> None:
        super().__init__()
        self.register_buffer("q", q.detach())
        self.register_buffer("J", J.detach())
        self.register_buffer("U_control", U_control.detach())
        self.register_buffer("U_body", U_body.detach())
        self.register_buffer("lam_body", lam_body.detach())
        self.solve_ridge = float(solve_ridge)
        cached: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for radius, weight in zip(radii, weights):
            edges = build_radius_graph(q.detach(), float(radius), max_neighbors)
            row, col = edges[0], edges[1]
            distance = torch.linalg.norm(q[row] - q[col], dim=-1)
            phi, _ = wendland_phi_dphi(distance / float(radius), kernel)
            cached.append((row, col, float(weight) * phi))
        self.cached = cached

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        p = self.U_control @ beta
        velocity = torch.zeros_like(self.q)
        for row, col, weighted_phi in self.cached:
            velocity.index_add_(
                0, row, weighted_phi[:, None] * p[col]
            )
        J = self.J
        if self.solve_ridge > 0.0:
            identity = torch.eye(3, dtype=J.dtype, device=J.device)
            J = J + self.solve_ridge * identity.view(1, 3, 3)
        body = torch.linalg.solve(J, velocity.unsqueeze(-1)).squeeze(-1)
        coefficients = self.U_body.T @ body
        return coefficients / self.lam_body[:, None]


def spectral_relative_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lam: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    delta2 = (predicted - target).pow(2)
    if norm == "raw":
        weight = torch.ones_like(lam)
    elif norm == "rkhs":
        weight = lam
    else:
        weight = lam.pow(2)
    weight = weight[:, None]
    return (weight * delta2).sum() / (weight * target.pow(2)).sum().clamp_min(1.0e-12)


def initialize_first_beta(
    alpha: torch.Tensor,
    control_dim: int,
    mode: str,
) -> torch.Tensor:
    initial = torch.zeros(
        (control_dim, 3), dtype=alpha.dtype, device=alpha.device
    )
    if mode == "shared_alpha":
        count = min(control_dim, alpha.shape[0])
        initial[:count] = alpha[:count]
    return initial


@torch.no_grad()
def state_derivative(
    q: torch.Tensor,
    J: torch.Tensor,
    beta: torch.Tensor,
    U_control: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    velocity, Du, _ = moving_field_and_jacobian(
        q,
        beta,
        U_control,
        config["radii_resolved"],
        config["weights_resolved"],
        config["kernel"],
        int(config["max_neighbors"]),
    )
    dJ = torch.einsum("vij,vjk->vik", Du, J)
    return velocity, dJ


class GeometryFixedStatePullback(FixedStatePullback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cached = []
        radii = args[5]
        weights = args[6]
        kernel = args[7]
        max_neighbors = args[8]
        q = args[0]
        for radius, weight in zip(radii, weights):
            edges = build_radius_graph(q.detach(), float(radius), max_neighbors)
            row, col = edges[0], edges[1]
            difference = q[row] - q[col]
            distance = torch.linalg.norm(difference, dim=-1)
            phi, dphi = wendland_phi_dphi(distance / float(radius), kernel)
            radial_derivative = dphi / (
                float(radius) * distance.clamp_min(torch.finfo(q.dtype).eps)
            )
            grad_kernel = radial_derivative[:, None] * difference
            cached.append(
                (row, col, float(weight) * phi, float(weight) * grad_kernel)
            )
        self.geometry_cached = cached

    def realize(self, beta):
        p = self.U_control @ beta
        velocity = torch.zeros_like(self.q)
        Du = torch.zeros(
            (self.q.shape[0], 3, 3), dtype=self.q.dtype, device=self.q.device
        )
        for row, col, weighted_phi, weighted_grad in self.geometry_cached:
            velocity.index_add_(0, row, weighted_phi[:, None] * p[col])
            Du.index_add_(
                0, row, p[col, :, None] * weighted_grad[:, None, :]
            )
        return velocity, Du

    def alpha_from_velocity(self, velocity):
        J = self.J
        if self.solve_ridge > 0.0:
            identity = torch.eye(3, dtype=J.dtype, device=J.device)
            J = J + self.solve_ridge * identity.view(1, 3, 3)
        body = torch.linalg.solve(J, velocity.unsqueeze(-1)).squeeze(-1)
        coefficients = self.U_body.T @ body
        return coefficients / self.lam_body[:, None]

    def forward(self, beta):
        velocity, _ = self.realize(beta)
        return self.alpha_from_velocity(velocity)


def make_operator(q, J, U_control, U_body, lam_body, config):
    return GeometryFixedStatePullback(
        q,
        J,
        U_control,
        U_body,
        lam_body,
        config["radii_resolved"],
        config["weights_resolved"],
        config["kernel"],
        int(config["max_neighbors"]),
        float(config.get("pullback_solve_ridge", 1.0e-6)),
    )


def solve_frame(
    operator,
    target_alpha,
    initial_beta,
    previous_beta,
    lam_control,
    args,
    label,
):
    beta = nn.Parameter(initial_beta.clone())
    optimizer = torch.optim.Adam([beta], lr=args.inner_lr)
    best_value = float("inf")
    best_beta = beta.detach().clone()
    best_values = None
    source = torch.as_tensor(
        args.reference_source_points_np, dtype=operator.q.dtype, device=operator.q.device
    )
    faces = torch.as_tensor(
        args.reference_source_faces_np, dtype=torch.long, device=operator.q.device
    )
    edges = unique_edges(faces)
    baseline = torch.linalg.norm(
        source[edges[:, 0]] - source[edges[:, 1]], dim=-1
    ).clamp_min(torch.finfo(source.dtype).eps)
    recovery_intervals = int(
        args.recovery_intervals
        if args.recovery_intervals is not None
        else args.saved_match_config.get("time_intervals", args.saved_match_config["n_steps"])
    )
    dt = 0.0 if label == "final" else 0.5 / float(recovery_intervals)

    for iteration in range(args.inner_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity, Du = operator.realize(beta)
        predicted = operator.alpha_from_velocity(velocity)
        fit = spectral_relative_loss(
            predicted, target_alpha, operator.lam_body, args.alpha_norm
        )
        energy = (lam_control[:, None] * beta.pow(2)).mean()
        temporal = (beta - previous_beta).pow(2).mean()
        if dt > 0.0:
            predicted_q = operator.q + dt * velocity
            predicted_J = operator.J + dt * torch.einsum(
                "vij,vjk->vik", Du, operator.J
            )
            detJ = torch.linalg.det(predicted_J)
            jacobian = torch.relu(args.jacobian_floor - detJ).pow(2).mean()
            current = torch.linalg.norm(
                predicted_q[edges[:, 0]] - predicted_q[edges[:, 1]], dim=-1
            ).clamp_min(1.0e-8)
            edge = torch.log(current / baseline).pow(2).mean()
        else:
            detJ = torch.linalg.det(operator.J)
            jacobian = beta.new_zeros(())
            edge = beta.new_zeros(())
        loss = (
            fit
            + args.lambda_energy * energy
            + args.lambda_temporal * temporal
            + args.lambda_jacobian * jacobian
            + args.lambda_edge * edge
        )
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_([beta], args.grad_clip).detach().cpu()
        )
        value = float(loss.detach().cpu())
        if value < best_value:
            best_value = value
            best_beta = beta.detach().clone()
            best_values = (
                float(fit.detach().cpu()),
                float(jacobian.detach().cpu()),
                float(edge.detach().cpu()),
                float(detJ.min().detach().cpu()),
                grad_norm,
            )
        if iteration < args.inner_iters:
            optimizer.step()

    assert best_values is not None
    with torch.no_grad():
        alpha_best = operator(best_beta)
    fit_value, jac_value, edge_value, det_min, grad_norm = best_values
    return best_beta, {
        "label": label,
        "objective": best_value,
        "alpha_relative_mse": fit_value,
        "alpha_relative_l2": float(np.sqrt(max(0.0, fit_value))),
        "beta_rms": float(best_beta.pow(2).mean().sqrt().cpu()),
        "beta_step_relative": relative_l2(best_beta, previous_beta),
        "final_grad_norm": grad_norm,
        "alpha_raw_relative_l2": relative_l2(alpha_best, target_alpha),
        "jacobian_penalty": jac_value,
        "edge_penalty": edge_value,
        "predicted_detJ_min": det_min,
    }


def sequential_decode(
    source: torch.Tensor,
    target_alpha: torch.Tensor,
    U_control: torch.Tensor,
    lam_control: torch.Tensor,
    U_body: torch.Tensor,
    lam_body: torch.Tensor,
    config: Dict,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    frame_count = target_alpha.shape[0]
    step_count = frame_count - 1
    dt = 1.0 / float(step_count)
    vertex_count = source.shape[0]
    q = source.clone()
    J = torch.eye(3, dtype=source.dtype, device=source.device).expand(
        vertex_count, 3, 3
    ).clone()
    q_frames = [q.clone()]
    J_frames = [J.clone()]
    beta_frames: List[torch.Tensor] = []
    midpoint_beta_frames: List[torch.Tensor] = []
    velocity_frames: List[torch.Tensor] = []
    rows: List[Dict] = []
    previous_beta = initialize_first_beta(
        target_alpha[0], U_control.shape[1], args.first_init
    )

    for step in range(step_count):
        operator = make_operator(
            q, J, U_control, U_body, lam_body, config
        )
        beta_left, row = solve_frame(
            operator,
            target_alpha[step],
            previous_beta,
            previous_beta,
            lam_control,
            args,
            f"left_{step:03d}",
        )
        rows.append(row)
        velocity_left, dJ_left = state_derivative(
            q, J, beta_left, U_control, config
        )
        beta_frames.append(beta_left)
        velocity_frames.append(velocity_left)

        if args.scheme == "euler":
            q = q + dt * velocity_left
            J = J + dt * dJ_left
            previous_beta = beta_left
        else:
            q_mid = q + 0.5 * dt * velocity_left
            J_mid = J + 0.5 * dt * dJ_left
            alpha_mid = 0.5 * (
                target_alpha[step] + target_alpha[step + 1]
            )
            midpoint_operator = make_operator(
                q_mid, J_mid, U_control, U_body, lam_body, config
            )
            beta_mid, mid_row = solve_frame(
                midpoint_operator,
                alpha_mid,
                beta_left,
                beta_left,
                lam_control,
                args,
                f"mid_{step:03d}",
            )
            rows.append(mid_row)
            velocity_mid, dJ_mid = state_derivative(
                q_mid, J_mid, beta_mid, U_control, config
            )
            midpoint_beta_frames.append(beta_mid)
            q = q + dt * velocity_mid
            J = J + dt * dJ_mid
            previous_beta = beta_mid

        q_frames.append(q.clone())
        J_frames.append(J.clone())
        print(
            f"[step {step + 1:02d}/{step_count:02d}] "
            f"left_alpha={row['alpha_relative_l2']:.3e} "
            f"detJ={float(torch.linalg.det(J).min().cpu()):.3e}",
            flush=True,
        )

    final_operator = make_operator(
        q, J, U_control, U_body, lam_body, config
    )
    beta_final, final_row = solve_frame(
        final_operator,
        target_alpha[-1],
        previous_beta,
        previous_beta,
        lam_control,
        args,
        "final",
    )
    rows.append(final_row)
    final_velocity, _ = state_derivative(
        q, J, beta_final, U_control, config
    )
    beta_frames.append(beta_final)
    velocity_frames.append(final_velocity)
    midpoint_beta = (
        torch.stack(midpoint_beta_frames)
        if midpoint_beta_frames
        else source.new_empty((0, U_control.shape[1], 3))
    )
    return (
        torch.stack(beta_frames),
        midpoint_beta,
        torch.stack(q_frames),
        torch.stack(J_frames),
        torch.stack(velocity_frames),
        rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_dir", "--match-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--scheme", choices=["euler", "midpoint"], default="midpoint")
    parser.add_argument("--alpha_norm", "--alpha-norm", choices=["raw", "rkhs", "body"], default="rkhs")
    parser.add_argument("--first_init", "--first-init", choices=["zero", "shared_alpha"], default="shared_alpha")
    parser.add_argument("--inner_iters", "--inner-iters", type=int, default=40)
    parser.add_argument(
        "--recovery_intervals", "--recovery-intervals", type=int, default=None,
        help=("Dense recovery/reconstruction intervals. Saved alpha knots "
              "are interpolated; generative dimension is unchanged."),
    )
    parser.add_argument("--inner_lr", "--inner-lr", type=float, default=1.0e-2)
    parser.add_argument("--lambda_energy", "--lambda-energy", type=float, default=1.0e-3)
    parser.add_argument("--lambda_temporal", "--lambda-temporal", type=float, default=1.0e-2)
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=10.0)
    parser.add_argument("--exact_si", "--exact-si", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda_jacobian", "--lambda-jacobian", type=float, default=100.0)
    parser.add_argument("--lambda_edge", "--lambda-edge", type=float, default=1.0)
    parser.add_argument("--jacobian_floor", "--jacobian-floor", type=float, default=0.1)
    parser.add_argument("--match-stage", choices=["stage1", "stage2"], default=None,
                        help="Select a two-stage root's stage; default is stage2. Omit for a leaf match directory.")
    parser.add_argument("--source", help="Override a relocated source path; saved initial geometry must agree.")
    parser.add_argument("--target", help="Override the saved diagnostic target path.")
    parser.add_argument("--basis", help="Override the path to the exact fitted Mercer basis (not a newly rotated basis).")
    parser.add_argument("--allow-basis-mismatch", action="store_true", help="Explicit diagnostic opt-out of kernel metadata matching.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing recovery outputs; input match directories stay protected.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_recovery_arguments(args)
    prepared = prepare_recovery_inputs(args)
    if args.inner_iters < 0 or args.inner_lr <= 0:
        raise ValueError("Inner iterations must be nonnegative and learning rate positive.")

    started = time.perf_counter()
    device = prepared["device"]
    dtype = torch.float32
    match_dir = prepared["match_dir"]
    output_dir = prepared["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepared["config"]
    args.saved_match_config = config
    source = load_mesh(config["source_vtk"], device, dtype)
    target = load_mesh(config["target_vtk"], device, dtype)
    args.reference_source_points_np = np.asarray(source.mesh.points, dtype=np.float32).copy()
    args.reference_source_faces_np = np.asarray(source.mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:].copy()
    D = int(config["body_basis_dim_resolved"])
    R = int(config["control_basis_dim_resolved"])
    U, lam, _ = load_mercer_basis(config["basis"], max(D, R), device, dtype)
    U_body, lam_body = U[:, :D], lam[:D]
    U_control, lam_control = U[:, :R], lam[:R]
    target_alpha = torch.as_tensor(
        np.load(match_dir / "alpha_knots.npy"), dtype=dtype, device=device
    )
    true_beta = torch.as_tensor(
        np.load(match_dir / "beta_knots.npy"), dtype=dtype, device=device
    )
    generated_alpha_samples = int(target_alpha.shape[0])
    if args.recovery_intervals is not None:
        if args.recovery_intervals <= 0:
            raise ValueError("--recovery_intervals must be positive.")
        target_alpha = resample_knots(
            target_alpha, int(args.recovery_intervals) + 1
        )
    primary_count = int(target_alpha.shape[0])
    if primary_count < 2:
        raise ValueError("alpha_knots.npy must contain at least two temporal samples.")
    q_primary_path = match_dir / "trajectory_q_knots.npy"
    velocity_primary_path = match_dir / "trajectory_velocity_knots.npy"
    true_q = torch.as_tensor(
        np.load(q_primary_path if q_primary_path.exists() else match_dir / "trajectory_q.npy"),
        dtype=dtype,
        device=device,
    )
    true_velocity = torch.as_tensor(
        np.load(
            velocity_primary_path
            if velocity_primary_path.exists()
            else match_dir / "trajectory_velocity.npy"
        ),
        dtype=dtype,
        device=device,
    )
    true_q = resample_knots(true_q, primary_count)
    true_velocity = resample_knots(true_velocity, primary_count)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    beta_frames, beta_midpoints, q_frames, J_frames, velocity_frames, rows = sequential_decode(
        source.points,
        target_alpha,
        U_control,
        lam_control,
        U_body,
        lam_body,
        config,
        args,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0

    scale = source_scale(source.points)
    true_displacement = true_q - source.points.unsqueeze(0)
    # New synchronized outputs have one beta and alpha value at every primary
    # knot. Legacy match directories are resampled only for compatibility.
    beta_frame_count = int(beta_frames.shape[0])
    true_beta_frames = (
        true_beta
        if true_beta.shape[0] == beta_frame_count
        else resample_knots(true_beta, beta_frame_count)
    )
    pointwise_target_available = bool(
        config.get("loss") == "pointwise"
        and true_q[-1].shape[0] == target.points.shape[0]
    )
    true_endpoint_mesh = pv.PolyData(
        true_q[-1].cpu().numpy().astype(np.float64),
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    ).triangulate()
    recovered_endpoint_mesh = pv.PolyData(
        q_frames[-1].cpu().numpy().astype(np.float64),
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    ).triangulate()
    target_mesh = target.mesh.extract_surface().triangulate()

    def surface_distances(first: pv.PolyData, second: pv.PolyData) -> Tuple[float, float]:
        first_to_second = np.abs(
            np.asarray(
                first.compute_implicit_distance(second, inplace=False).point_data[
                    "implicit_distance"
                ],
                dtype=np.float64,
            )
        )
        second_to_first = np.abs(
            np.asarray(
                second.compute_implicit_distance(first, inplace=False).point_data[
                    "implicit_distance"
                ],
                dtype=np.float64,
            )
        )
        assd = 0.5 * (first_to_second.mean() + second_to_first.mean())
        hd95 = max(
            np.percentile(first_to_second, 95.0),
            np.percentile(second_to_first, 95.0),
        )
        return float(assd), float(hd95)

    true_assd, true_hd95 = surface_distances(true_endpoint_mesh, target_mesh)
    recovered_assd, recovered_hd95 = surface_distances(
        recovered_endpoint_mesh, target_mesh
    )

    metrics: Dict = {
        # Keep the historical key for downstream CSV compatibility.
        "beta_relative_l2_knot_grid": relative_l2(beta_frames, true_beta_frames),
        "beta_relative_l2_first_30": relative_l2(beta_frames, true_beta_frames),
        "beta_relative_l2_interpolated_frames": relative_l2(
            beta_frames, true_beta_frames
        ),
        "velocity_relative_l2": relative_l2(velocity_frames, true_velocity),
        "trajectory_relative_to_true_displacement_l2": float(
            ((q_frames - true_q).pow(2).sum().sqrt()
             / true_displacement.pow(2).sum().sqrt().clamp_min(1.0e-12)).cpu()
        ),
        "trajectory_source_scale_nrmse": normalized_vertex_rmse(q_frames, true_q, scale),
        "endpoint_vs_true_source_scale_nrmse": normalized_vertex_rmse(q_frames[-1], true_q[-1], scale),
        "target_pointwise_available": pointwise_target_available,
        "true_endpoint_target_source_scale_nrmse": (
            normalized_vertex_rmse(true_q[-1], target.points, scale)
            if pointwise_target_available
            else math.nan
        ),
        "recovered_endpoint_target_source_scale_nrmse": (
            normalized_vertex_rmse(q_frames[-1], target.points, scale)
            if pointwise_target_available
            else math.nan
        ),
        "true_endpoint_target_surface_assd": true_assd,
        "true_endpoint_target_surface_hd95": true_hd95,
        "recovered_endpoint_target_surface_assd": recovered_assd,
        "recovered_endpoint_target_surface_hd95": recovered_hd95,
        "detJ_min_all_frames": float(torch.linalg.det(J_frames).min().cpu()),
        "detJ_nonpositive_fraction": float((torch.linalg.det(J_frames) <= 0).float().mean().cpu()),
        "alpha_frame_relative_l2_mean": float(np.mean([row["alpha_relative_l2"] for row in rows])),
        "alpha_frame_relative_l2_max": float(np.max([row["alpha_relative_l2"] for row in rows])),
    }
    if args.exact_si:
        metrics["endpoint_qc"] = mesh_quality_metrics(
            q_frames[-1].cpu().numpy().astype(np.float64),
            source.faces.cpu().numpy().astype(np.int64),
            compute_self_intersections=True,
        )

    knot_times = np.linspace(0.0, 1.0, beta_frames.shape[0], dtype=np.float32)
    midpoint_times = (
        (np.arange(beta_midpoints.shape[0], dtype=np.float32) + 0.5)
        / max(1, beta_midpoints.shape[0])
    )
    np.save(output_dir / "sequential_knot_times.npy", knot_times)
    np.save(output_dir / "sequential_beta_knots.npy", beta_frames.cpu().numpy())
    # Legacy alias retained for existing evaluation scripts.
    np.save(output_dir / "sequential_beta_frames.npy", beta_frames.cpu().numpy())
    np.save(output_dir / "sequential_midpoint_times.npy", midpoint_times)
    np.save(output_dir / "sequential_beta_midpoints.npy", beta_midpoints.cpu().numpy())
    np.save(output_dir / "sequential_trajectory_q.npy", q_frames.cpu().numpy())
    np.save(output_dir / "sequential_trajectory_J.npy", J_frames.cpu().numpy())
    np.save(output_dir / "sequential_velocity.npy", velocity_frames.cpu().numpy())
    mesh = pv.PolyData(
        q_frames[-1].cpu().numpy().astype(np.float64),
        np.asarray(source.mesh.faces, dtype=np.int64).copy(),
    )
    mesh.save(output_dir / "sequential_recovered_endpoint.vtk")
    with (output_dir / "frame_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "role": "DF-PSM induced-alpha inverse-recovery diagnostic",
        "inputs": {"requested_match_dir": str(prepared["requested_match_dir"]),
                   "source": config["source_vtk"], "target": config["target_vtk"], "basis": config["basis"]},
        "geometry_regularization": {
            "lambda_jacobian": args.lambda_jacobian,
            "lambda_edge": args.lambda_edge,
            "jacobian_floor": args.jacobian_floor,
            "local_prediction_dt": 0.5 / float(target_alpha.shape[0] - 1),
            "solver_prediction_dt": 0.5 / float(args.recovery_intervals if args.recovery_intervals is not None else config.get("time_intervals", config["n_steps"])),
        },
        "match_dir": str(match_dir),
        "body_dim": D,
        "control_dim": R,
        "scheme": args.scheme,
        "alpha_norm": args.alpha_norm,
        "inner_iters": args.inner_iters,
        "inner_lr": args.inner_lr,
        "lambda_energy": args.lambda_energy,
        "lambda_temporal": args.lambda_temporal,
        "time_intervals": int(target_alpha.shape[0] - 1),
        "time_samples": int(target_alpha.shape[0]),
        "generated_alpha_samples": generated_alpha_samples,
        "generated_alpha_intervals": generated_alpha_samples - 1,
        "recovery_intervals_requested": args.recovery_intervals,
        "midpoint_control_count": int(beta_midpoints.shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print("\nSequential recovery summary", flush=True)
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            print(f"  {key}: {value:.6e}", flush=True)
    print(f"  summary: {output_dir / 'summary.json'}", flush=True)



def cli():
    try:
        with legacy_precision():
            main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
