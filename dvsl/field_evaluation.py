"""Memory-bounded evaluation of the moving Wendland field and its differential."""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .kernels import radius_graph, wendland_phi_dphi


def scalar_key_symmetrize(edges: torch.Tensor, count: int) -> torch.Tensor:
    """Return the same sorted symmetric pair union as the legacy graph builder."""
    if edges.numel() == 0:
        return edges
    # int64 row*N+column preserves lexicographic pair order without a 2D sort.
    keys = torch.cat((edges[0] * count + edges[1], edges[1] * count + edges[0]))
    keys = torch.unique(keys, sorted=True)
    return torch.stack((torch.div(keys, count, rounding_mode="floor"), keys.remainder(count)))


def _edge_contribution(q, p, row, col, *, radius, weight, kernel):
    difference = q[row] - q[col]
    distance = torch.linalg.norm(difference, dim=-1)
    phi, dphi = wendland_phi_dphi(distance / float(radius), kernel)
    p_edge = p[col]
    velocity = torch.zeros_like(q).index_add(0, row, float(weight) * phi[:, None] * p_edge)
    derivative = dphi / (float(radius) * torch.clamp(distance, min=torch.finfo(q.dtype).eps))
    grad_kernel = derivative[:, None] * difference
    jacobian = torch.zeros((len(q), 3, 3), device=q.device, dtype=q.dtype).index_add(
        0, row, float(weight) * p_edge[:, :, None] * grad_kernel[:, None, :]
    )
    return velocity, jacobian


def _chunked_field(q, beta, U, radii, weights, kernel, max_neighbors, chunk_size):
    if radius_graph is None:
        raise RuntimeError("torch_cluster matching the installed PyTorch/CUDA build is required.")
    p = U @ beta
    velocity = torch.zeros_like(q)
    jacobian = torch.zeros((len(q), 3, 3), device=q.device, dtype=q.dtype)
    for radius, weight in zip(radii, weights):
        raw = radius_graph(q.detach(), r=float(radius), loop=True,
                           max_num_neighbors=int(max_neighbors))
        edges = scalar_key_symmetrize(raw, len(q))
        del raw
        contribution = partial(_edge_contribution, radius=radius, weight=weight, kernel=kernel)
        for start in range(0, edges.shape[1], chunk_size):
            row, col = edges[0, start:start + chunk_size], edges[1, start:start + chunk_size]
            # Pass edge tensors explicitly so outer checkpoint hooks can release
            # them. Capturing them in the callable keeps all trajectory graphs live.
            if torch.is_grad_enabled() and (q.requires_grad or p.requires_grad):
                v, j = checkpoint(contribution, q, p, row, col, use_reentrant=False,
                                  determinism_check="none")
            else:
                v, j = contribution(q, p, row, col)
            velocity = velocity + v
            jacobian = jacobian + j
    return velocity, jacobian, p


def bounded_field_and_jacobian(q, beta, U, radii, weights, kernel, max_neighbors, chunk_size):
    """Chunk edge tensors and recompute graphs during backward; retain all neighbors.

    This bounds field intermediates, not the complete solver's memory. Chunked
    floating-point accumulation is numerically equivalent, not bitwise identical.
    """
    if chunk_size <= 0:
        raise ValueError("Field chunk size must be positive.")
    if len(radii) != len(weights):
        raise ValueError("Kernel radii and weights must have the same length.")
    evaluate = partial(_chunked_field, radii=radii, weights=weights, kernel=kernel,
                       max_neighbors=max_neighbors, chunk_size=chunk_size)
    if torch.is_grad_enabled() and (q.requires_grad or beta.requires_grad or U.requires_grad):
        return checkpoint(evaluate, q, beta, U, use_reentrant=False, determinism_check="none")
    return evaluate(q, beta, U)
