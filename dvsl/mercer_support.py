"""Unchanged sparse Gram/eigensolver helpers, without unrelated solver imports."""
from __future__ import annotations
from typing import List, Optional, Sequence
import torch
try:
    from torch_cluster import radius_graph, radius as bipartite_radius
except Exception:
    radius_graph = None
    bipartite_radius = None
    HAS_CLUSTER = False
else:
    HAS_CLUSTER = True

def _phi_c2(q: torch.Tensor) -> torch.Tensor:
    s = torch.clamp(1.0 - q, min=0.0)
    return s**4 * (1.0 + 4.0*q)


def _phi_c4(q: torch.Tensor) -> torch.Tensor:
    s = torch.clamp(1.0 - q, min=0.0)
    poly = (35.0*q*q + 18.0*q + 3.0)/3.0
    return s**6 * poly


def wendland_phi(q: torch.Tensor, kind: str) -> torch.Tensor:
    k = kind.lower()
    if k == "c2": return _phi_c2(q)
    if k == "c4": return _phi_c4(q)
    raise ValueError("kernel must be 'c2' or 'c4'")


def _symmetrize_edges(edge_index: torch.Tensor) -> torch.Tensor:
    ei = edge_index
    e  = torch.stack([ei[0], ei[1]], dim=1)
    er = torch.stack([ei[1], ei[0]], dim=1)
    merged = torch.cat([e, er], dim=0)
    merged = torch.unique(merged, dim=0)
    return merged.t().contiguous()


def build_sparse_wendland_gram_cuda(x: torch.Tensor, R: float, kind: str, max_neighbors: int) -> torch.Tensor:
    """
    x: (V,3) on CUDA, fp32 by default. Returns coalesced sparse COO K on CUDA.
    """
    assert x.is_cuda, "points must be on CUDA for GPU graph build"
    if not HAS_CLUSTER:
        raise RuntimeError("torch_cluster is not installed with CUDA; install torch-cluster for your PyTorch/CUDA build.")

    V = x.shape[0]
    ei = radius_graph(x, r=R, loop=True, max_num_neighbors=max_neighbors)     # [2, E] on CUDA
    ei = _symmetrize_edges(ei)                                                # ensure symmetry
    i, j = ei[0], ei[1]
    diff = x[i] - x[j]
    r = torch.linalg.norm(diff, dim=-1)
    q = r / R
    phi = wendland_phi(q, kind=kind).to(x.dtype)

    # Build sparse K and coalesce to merge duplicates
    idx = torch.stack([i, j], dim=0)                                          # [2, E]
    K = torch.sparse_coo_tensor(idx, phi, (V, V), device=x.device, dtype=x.dtype).coalesce()
    return K


@torch.no_grad()
def randomized_top_eigs_sym(op_mv, V: int, k: int, oversample=64, iters=2, dtype=torch.float32, device="cuda", seed=12345):
    """
    Top-k eigenpairs of symmetric PSD operator using subspace iteration:
      Q0 ~ N(0,1); repeat iters: Q <- orth(op_mv(Q)); then eig(B=Q^T op_mv(Q)).
    Entirely on the chosen device.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    r = min(k + oversample, V)
    Q = torch.randn(V, r, device=device, dtype=dtype, generator=gen)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    for _ in range(max(1, iters)):
        Z = op_mv(Q)                              # (V,r)
        Q, _ = torch.linalg.qr(Z, mode="reduced")
    B = Q.T @ op_mv(Q)                            # (r,r) dense, stays on GPU
    S, U = torch.linalg.eigh(B)                   # ascending
    idx = torch.argsort(S, descending=True)[:k]
    lam = S[idx]
    Ubig = Q @ U[:, idx]                          # (V,k)
    # Final polish
    Ubig, _ = torch.linalg.qr(Ubig, mode="reduced")
    return lam, Ubig


def sign_fix_columns(U: torch.Tensor) -> torch.Tensor:
    m = torch.argmax(torch.abs(U), dim=0)
    s = torch.sign(U[m, torch.arange(U.shape[1], device=U.device)])
    s[s == 0] = 1.0
    return U * s

def resolve_rkhs_weights(
    count: int, requested: Optional[Sequence[float]]
) -> List[float]:
    """Resolve positive multi-RKHS weights and normalize them to sum to one."""
    if count <= 0:
        raise ValueError("At least one RKHS component is required.")
    if requested is None:
        return [1.0 / count] * count
    weights = [float(value) for value in requested]
    if len(weights) != count:
        raise ValueError(
            "--shooting_weights must have the same length as the resolved RKHS "
            "basis/radius lists."
        )
    if any(value <= 0.0 for value in weights):
        raise ValueError("All shooting weights must be strictly positive.")
    total = sum(weights)
    return [value / total for value in weights]

