"""Unchanged compact-support deformation kernels and sparse graph convention."""
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

try:
    from torch_cluster import radius_graph
except Exception as exc:
    radius_graph = None
    _TORCH_CLUSTER_IMPORT_ERROR = exc

def wendland_phi(q: torch.Tensor, kind: str = "c2") -> torch.Tensor:
    s = torch.clamp(1.0 - q, min=0.0)
    if kind == "c2":
        return s**4 * (1.0 + 4.0 * q)
    if kind == "c4":
        return s**6 * ((35.0 * q * q + 18.0 * q + 3.0) / 3.0)
    raise ValueError("Wendland kind must be 'c2' or 'c4'.")


def wendland_phi_dphi(q: torch.Tensor, kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return phi(q) and d phi / d q using the DSMN formulas."""
    s = torch.clamp(1.0 - q, min=0.0)
    if kind == "c2":
        phi = s**4 * (1.0 + 4.0 * q)
        dphi = -20.0 * q * s**3
        valid = (q >= 0.0) & (q <= 1.0)
        dphi = torch.where(valid, dphi, torch.zeros_like(dphi))
        return phi, dphi
    if kind == "c4":
        A = (35.0 * q * q + 18.0 * q + 3.0) / 3.0
        Ap = (70.0 * q + 18.0) / 3.0
        phi = s**6 * A
        dphi = s**5 * (-6.0 * A + s * Ap)
        valid = (q >= 0.0) & (q <= 1.0)
        dphi = torch.where(valid, dphi, torch.zeros_like(dphi))
        return phi, dphi
    raise ValueError("Wendland kind must be 'c2' or 'c4'.")


def _symmetrize_edges(edge_index: torch.Tensor) -> torch.Tensor:
    edge = torch.stack([edge_index[0], edge_index[1]], dim=1)
    reverse = torch.stack([edge_index[1], edge_index[0]], dim=1)
    merged = torch.unique(torch.cat([edge, reverse], dim=0), dim=0)
    return merged.t().contiguous()


def build_radius_graph(
    points: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> torch.Tensor:
    if radius_graph is None:
        raise RuntimeError(
            "torch_cluster is required for the sparse DSMN Hamiltonian. "
            "Install the torch_cluster build matching your PyTorch/CUDA version. "
            f"Original import error: {_TORCH_CLUSTER_IMPORT_ERROR}"
        )
    edges = radius_graph(
        points,
        r=float(radius),
        loop=True,
        max_num_neighbors=int(max_neighbors),
    )
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=points.device)
    return _symmetrize_edges(edges)

