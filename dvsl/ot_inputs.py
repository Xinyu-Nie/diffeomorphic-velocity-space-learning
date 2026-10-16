"""Input resolution and validation for the legacy-compatible OT preprocessor."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def validate_parameters(args) -> None:
    if args.eps is not None and (not math.isfinite(args.eps) or args.eps <= 0):
        raise ValueError("--eps must be finite and positive (or choose --auto-eps).")
    if not math.isfinite(args.tol) or args.tol < 0:
        raise ValueError("--tol must be finite and nonnegative.")
    for name in ("max_iter", "topk", "workers"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")


def resolve_inputs(source, targets, output_dir, *, cwd=None):
    """Retain list order, whitespace/blank-line handling and CWD-relative paths."""
    working = Path.cwd() if cwd is None else Path(cwd)
    working = working.resolve()

    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else working / path).resolve()

    def require_file(path, label):
        path = resolve(path)
        if not path.is_file():
            raise ValueError(f"{label} is not an existing file: {path}")
        return path

    source_path = require_file(source, "Source mesh")
    targets_path = require_file(targets, "--targets input")
    if targets_path.suffix.lower() == ".vtk":
        target_paths = [str(targets_path)]
    else:
        try:
            lines = targets_path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise ValueError("--targets must be one .vtk file or a UTF-8 text list of mesh paths.") from exc
        target_paths = []
        for number, line in enumerate(lines, 1):
            entry = line.strip()
            if entry:
                try:
                    target = require_file(entry, f"Target on line {number} of {targets_path}")
                except (OSError, ValueError) as exc:
                    raise ValueError(f"{exc}. Relative list entries resolve from the working directory ({working}), not the list's folder.") from exc
                target_paths.append(str(target))
        if not target_paths:
            raise ValueError(f"No targets found in list: {targets_path}")

    # Legacy output names contain only target stems. Detect overwrite/worker races.
    stems = {}
    for target in target_paths:
        stem = Path(target).stem
        key = stem.casefold()
        if key in stems:
            raise ValueError(f"Targets '{stems[key]}' and '{target}' share output basename '{stem}'. Use unique target basenames.")
        stems[key] = target
    output = resolve(output_dir)
    if output.exists() and not output.is_dir():
        raise ValueError(f"--output-dir is an existing file: {output}")
    return source_path, target_paths, output


def validate_mesh_arrays(points, faces, path) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Mesh must contain a nonempty [N,3] point array: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Mesh contains non-finite coordinates: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Mesh must contain triangular faces: {path}")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError(f"Mesh has triangle indices outside its vertex array: {path}")
