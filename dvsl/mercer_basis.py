#!/usr/bin/env python3
"""Compute a unified Mercer basis for a weighted sum of Wendland kernels.

The constructed source Gram operator is

    K_mix(X, X) = sum_s w_s K_{R_s}(X, X),

where positive weights are normalized to sum to one.  A single top-k Mercer
basis of this mixed operator is saved as ``U`` and ``lam`` in an NPZ file.  The
output supports DF-PSM with matching radii/weights, and DTM for one radius.
The stored convention is unweighted Gram eigenvectors, not area quadrature.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyvista as pv
import torch
from .mercer_validation import validate_inputs, legacy_precision, enrich_metadata, validate_selected_eigenpairs, normalized_output_path

from . import mercer_support as core
from .mercer_support import (
    build_sparse_wendland_gram_cuda,
    randomized_top_eigs_sym,
    sign_fix_columns,
)


def normalize_weights(
    radii: Sequence[float], weights: Optional[Sequence[float]]
) -> List[float]:
    resolved_radii = [float(value) for value in radii]
    if not resolved_radii or any(value <= 0.0 for value in resolved_radii):
        raise ValueError("All shooting radii must be positive.")
    return core.resolve_rkhs_weights(len(resolved_radii), weights)


@torch.no_grad()
def _build_and_save_unified_basis(
    source_vtk: str,
    output_path: str,
    radii: Sequence[float],
    weights: Sequence[float],
    kernel: str,
    max_neighbors: int,
    basis_dim: int,
    oversample: int,
    power_iters: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    storage_dtype: str,
) -> Dict:
    """Build top eigenpairs of the normalized weighted sparse Gram sum."""
    if device.type != "cuda":
        raise ValueError("Unified mixed-basis construction currently requires CUDA.")
    source_mesh = pv.read(source_vtk).triangulate()
    points = torch.as_tensor(
        np.asarray(source_mesh.points, dtype=np.float64).copy(),
        dtype=dtype,
        device=device,
    )
    vertex_count = int(points.shape[0])
    if not 1 <= int(basis_dim) <= vertex_count:
        raise ValueError(
            f"--basis_dim must be in [1, {vertex_count}], got {basis_dim}."
        )

    grams: List[torch.Tensor] = []
    graph_statistics: List[Dict[str, int]] = []
    for index, radius in enumerate(radii):
        edge = core.bipartite_radius(
            points,
            points,
            r=float(radius),
            max_num_neighbors=int(max_neighbors),
        )
        counts = torch.bincount(edge[0], minlength=vertex_count)
        max_count = int(counts.max().item())
        if max_count >= int(max_neighbors):
            raise RuntimeError(
                f"Radius {radius:g} saturated --max_neighbors={max_neighbors}; "
                "increase the cap before constructing the basis."
            )
        print(
            f"Building mixed-basis Gram component {index}: radius={radius:g}, "
            f"edges={edge.shape[1]}, max_neighbors/query={max_count}"
        )
        grams.append(
            build_sparse_wendland_gram_cuda(
                points,
                R=float(radius),
                kind=kernel,
                max_neighbors=int(max_neighbors),
            )
        )
        graph_statistics.append(
            {"edges": int(edge.shape[1]), "max_neighbors_per_query": max_count}
        )

    def mixed_matmul(matrix: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (vertex_count, matrix.shape[1]),
            dtype=dtype,
            device=device,
        )
        for weight, gram in zip(weights, grams):
            result = result + float(weight) * torch.sparse.mm(gram, matrix)
        return result

    started = time.perf_counter()
    eigenvalues, basis = randomized_top_eigs_sym(
        mixed_matmul,
        vertex_count,
        int(basis_dim),
        oversample=int(oversample),
        iters=int(power_iters),
        dtype=dtype,
        device=str(device),
        seed=int(seed),
    )

    # Rayleigh-Ritz polish after the randomized subspace iteration.
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    projected = basis.T @ mixed_matmul(basis)
    projected = 0.5 * (projected + projected.T)
    values_small, vectors_small = torch.linalg.eigh(projected)
    order = torch.argsort(values_small, descending=True)
    eigenvalues = values_small[order]
    basis = basis @ vectors_small[:, order]
    basis = sign_fix_columns(basis)

    applied = mixed_matmul(basis)
    residual = torch.linalg.norm(applied - basis * eigenvalues.view(1, -1))
    residual = residual / torch.clamp(
        torch.linalg.norm(basis * eigenvalues.view(1, -1)),
        min=torch.finfo(dtype).eps,
    )
    orthogonality = torch.linalg.norm(
        basis.T @ basis
        - torch.eye(basis.shape[1], dtype=dtype, device=device)
    ) / math.sqrt(float(basis.shape[1]))

    metadata = {
        "src": os.path.abspath(source_vtk),
        "V": vertex_count,
        "kernel": kernel,
        "mixed_radii": [float(value) for value in radii],
        "mixed_weights": [float(value) for value in weights],
        "max_neighbors": int(max_neighbors),
        "k_eig": int(basis_dim),
        "oversample": int(oversample),
        "iters": int(power_iters),
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(seed),
        "graph_statistics": graph_statistics,
        "relative_eigen_residual": float(residual.cpu()),
        "relative_orthogonality_error": float(orthogonality.cpu()),
        "construction_elapsed_sec": time.perf_counter() - started,
        "basis_type": "unified_normalized_weighted_sum_kernel",
    }
    validate_selected_eigenpairs(eigenvalues, basis, storage_dtype)
    metadata = enrich_metadata(metadata, source_vtk, radii)
    output_path = normalized_output_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    numpy_dtype = np.float64 if storage_dtype == "float64" else np.float32
    np.savez_compressed(
        output_path,
        U=basis.detach().cpu().numpy().astype(numpy_dtype),
        lam=eigenvalues.detach().cpu().numpy().astype(numpy_dtype),
        meta=json.dumps(metadata),
    )
    print(
        f"Unified mixed basis saved: {output_path}\n"
        f"  relative eigen residual: {metadata['relative_eigen_residual']:.3e}\n"
        f"  relative orthogonality error: "
        f"{metadata['relative_orthogonality_error']:.3e}"
    )
    return metadata



@torch.no_grad()
def build_and_save_unified_basis(
    source_vtk: str, output_path: str, radii: Sequence[float], weights: Sequence[float],
    kernel: str, max_neighbors: int, basis_dim: int, oversample: int, power_iters: int,
    device: torch.device, dtype: torch.dtype, seed: int, storage_dtype: str,
) -> Dict:
    """Preserve the legacy numerical path and precision, restoring caller settings afterward."""
    validate_inputs(source_vtk, radii, weights, kernel, max_neighbors, basis_dim,
                    oversample, power_iters, device, dtype, storage_dtype)
    with legacy_precision():
        return _build_and_save_unified_basis(
            source_vtk, output_path, radii, weights, kernel, max_neighbors, basis_dim,
            oversample, power_iters, device, dtype, seed, storage_dtype)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a GPU Mercer eigenbasis of a normalized weighted sum of "
            "compact Wendland kernel Gram matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", "--source", required=True, help="Source/template VTK mesh.")
    parser.add_argument(
        "--radii",
        "--shooting_radii",
        dest="radii",
        nargs="+",
        type=float,
        required=True,
        help="One or more positive Wendland support radii.",
    )
    parser.add_argument(
        "--weights",
        "--shooting_weights",
        dest="weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional positive kernel weights. They are normalized to sum to "
            "one; equal weights are used when omitted."
        ),
    )
    parser.add_argument("--kernel", choices=["c2", "c4"], default="c2")
    parser.add_argument("--max_neighbors", type=int, default=512)
    parser.add_argument(
        "--norm",
        choices=["none"],
        default="none",
        help="Compatibility option; mixed Gram matrices are left unnormalized.",
    )
    parser.add_argument("--k_eig", "--basis_dim", "--rank", dest="k_eig", type=int, default=2048)
    parser.add_argument("--oversample", type=int, default=64)
    parser.add_argument(
        "--iters",
        type=int,
        default=8,
        help=(
            "Randomized subspace iterations. Eight is recommended for the "
            "2,048-mode mixed basis; two reproduces the legacy setting."
        ),
    )
    parser.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument(
        "--storage_dtype", choices=["float32", "float64"], default="float32"
    )
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", "--output", required=True, help="Output mixed-basis NPZ path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out = normalized_output_path(args.out)
    if args.max_neighbors <= 0:
        raise ValueError("--max_neighbors must be positive.")
    if args.k_eig <= 0:
        raise ValueError("--k_eig must be positive.")
    if args.oversample < 0:
        raise ValueError("--oversample cannot be negative.")
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")
    if core.bipartite_radius is None:
        raise RuntimeError("torch_cluster with CUDA support is required.")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Unified mixed-basis construction requires CUDA.")
    dtype = torch.float64 if args.dtype == "fp64" else torch.float32
    radii = [float(value) for value in args.radii]
    weights = normalize_weights(radii, args.weights)

    metadata = build_and_save_unified_basis(
        source_vtk=os.path.abspath(os.path.expanduser(args.src)),
        output_path=os.path.abspath(os.path.expanduser(args.out)),
        radii=radii,
        weights=weights,
        kernel=args.kernel,
        max_neighbors=args.max_neighbors,
        basis_dim=args.k_eig,
        oversample=args.oversample,
        power_iters=args.iters,
        device=device,
        dtype=dtype,
        seed=args.seed,
        storage_dtype=args.storage_dtype,
    )
    print("\nMixed Mercer basis construction finished.")
    print("  convention: unweighted Gram eigenvectors; U.T @ U = I; no area quadrature")
    if len(radii) > 1:
        print("  compatibility: DF-PSM supports this mixed basis; current DTM has a single-radius decoder only.")
    else:
        print("  compatibility: DF-PSM and single-radius DTM; GES does not consume a Mercer basis.")
    print(f"  output: {os.path.abspath(os.path.expanduser(args.out))}")
    print(f"  radii: {metadata['mixed_radii']}")
    print(f"  weights: {metadata['mixed_weights']}")
    print(f"  modes: {metadata['k_eig']}")
    print(f"  eigen residual: {metadata['relative_eigen_residual']:.6e}")
    print(
        "  orthogonality error: "
        f"{metadata['relative_orthogonality_error']:.6e}"
    )


def cli():
    import sys
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
