"""Validation and explicit compatibility metadata for the legacy Gram basis."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import torch


def normalized_output_path(path):
    value = os.path.abspath(os.path.expanduser(str(path)))
    # Match NumPy's existing automatic suffix behavior, but report the actual path.
    return value if value.endswith(".npz") else value + ".npz"


@contextmanager
def legacy_precision():
    """Apply the original imported helpers' TF32/high policy only during construction."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def validate_inputs(source, radii, weights, kernel, max_neighbors, rank,
                    oversample, power_iters, device, dtype, storage_dtype):
    if not radii or any(not math.isfinite(float(r)) or r <= 0 for r in radii):
        raise ValueError("All --radii values must be finite and positive.")
    if len(weights) != len(radii) or any(not math.isfinite(float(w)) or w <= 0 for w in weights):
        raise ValueError("Weights must match radii and be finite and strictly positive.")
    if kernel not in ("c2", "c4"):
        raise ValueError("Kernel must be c2 or c4.")
    if max_neighbors <= 0 or rank <= 0 or power_iters <= 0 or oversample < 0:
        raise ValueError("Require positive max_neighbors/rank/iters and nonnegative oversample.")
    if storage_dtype not in ("float32", "float64") or dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32/float64 storage and torch.float32/torch.float64 computation.")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Source/template file does not exist: {path}")
    try:
        mesh = pv.read(str(path)).triangulate()
        points = np.asarray(mesh.points)
    except Exception as exc:
        raise ValueError(f"Cannot read source/template vertices: {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Source must contain a nonempty, finite [N,3] point array.")
    if rank > len(points):
        raise ValueError(f"--k_eig must be in [1, {len(points)}], got {rank}.")
    from . import mercer_support
    if not mercer_support.HAS_CLUSTER or mercer_support.bipartite_radius is None:
        raise RuntimeError("torch_cluster with CUDA support is required for sparse basis construction.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Unified mixed-basis construction requires CUDA; no CPU construction path is implemented.")


def validate_selected_eigenpairs(eigenvalues, basis, storage_dtype):
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(basis).all()):
        raise ValueError("Computed eigenpairs are non-finite; no numerical clipping or replacement was applied.")
    dtype = np.float64 if storage_dtype == "float64" else np.float32
    saved = eigenvalues.detach().cpu().numpy().astype(dtype)
    if not np.isfinite(saved).all() or np.any(saved <= 0):
        raise ValueError("Selected eigenvalues must remain finite and positive in storage precision for matching loaders. Reduce rank or use higher precision; no eigenvalues were clipped.")


def enrich_metadata(metadata, source, radii):
    result = dict(metadata)
    digest = hashlib.sha256()
    with Path(source).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result.update(source_file_sha256=digest.hexdigest(),
                  quadrature="none_counting_measure",
                  eigenvector_normalization="U.T @ U = I (Euclidean)",
                  energy_coordinates="p = U beta; p.T K p = beta.T diag(lam) beta",
                  matcher_compatibility={"df_psm": "single_or_mixed_kernel",
                                         "dtm": "single_radius_only" if len(radii) == 1 else "unsupported_mixed_gradient_decoder",
                                         "ges": "no_basis_input"})
    if len(radii) == 1:
        result["R"] = float(radii[0])
    return result
