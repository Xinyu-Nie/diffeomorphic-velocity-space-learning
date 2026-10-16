"""Input checks for the induced-alpha DF-PSM inverse-recovery diagnostic."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .surface_io import load_mercer_basis, load_mesh
from .surface_matching import validate_basis_metadata
from .surface_preprocessing import matching_topology


def validate_recovery_arguments(args):
    if args.inner_iters < 0:
        raise ValueError("--inner-iters must be nonnegative.")
    if not math.isfinite(args.inner_lr) or args.inner_lr <= 0:
        raise ValueError("--inner-lr must be finite and positive.")
    if args.recovery_intervals is not None and args.recovery_intervals <= 0:
        raise ValueError("--recovery-intervals must be positive.")
    for name in ("lambda_energy", "lambda_temporal", "lambda_jacobian", "lambda_edge", "jacobian_floor", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if args.scheme not in ("euler", "midpoint") or args.alpha_norm not in ("raw", "rkhs", "body"):
        raise ValueError("Unsupported recovery scheme or alpha norm.")


def resolve_match_directory(value, selected_stage=None):
    requested = Path(value).expanduser().resolve()
    if not requested.is_dir():
        raise ValueError(f"Match directory does not exist: {requested}")
    if (requested / "config.json").is_file():
        if selected_stage is not None:
            raise ValueError("--match-stage selects a two-stage workflow root; omit it for a leaf match directory.")
        return requested, requested
    if (requested / "two_stage_config.json").is_file():
        stage = selected_stage or "stage2"
        leaf = requested / ("stage1_ot" if stage == "stage1" else "stage2_target")
        if not (leaf / "config.json").is_file():
            raise ValueError(f"Selected two-stage match has no config.json: {leaf}")
        return requested, leaf
    raise ValueError("Expected a DF-PSM run with config.json or a packaged two-stage workflow root.")


def _saved_file(value, match_dir, label, override=None):
    if override is not None:
        candidates = [Path(override).expanduser().resolve()]
    else:
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, match_dir / path]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    flag = {"source": "--source", "target": "--target", "basis": "--basis"}[label]
    raise ValueError(f"Saved {label} file cannot be found ({value}); use {flag} for an explicitly relocated input.")


def _array(path, shape_tail, minimum_frames=1):
    if not path.is_file():
        raise ValueError(f"Required DF-PSM export is missing: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Cannot load numerical DF-PSM export {path}: {exc}") from exc
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != tuple(shape_tail) or array.shape[0] < minimum_frames:
        raise ValueError(f"{path.name} must have shape (T,{','.join(map(str, shape_tail))}) with T >= {minimum_frames}; got {array.shape}.")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{path.name} must contain real numerical data.")
    for start in range(0, array.shape[0], 16):
        if not np.isfinite(array[start:start+16]).all():
            raise ValueError(f"{path.name} contains non-finite data.")
    return array


def prepare_recovery_inputs(args):
    requested, match_dir = resolve_match_directory(args.match_dir, args.match_stage)
    try:
        config = json.loads((match_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read DF-PSM config.json: {exc}") from exc
    required = ("source_vtk", "target_vtk", "basis", "body_basis_dim_resolved", "control_basis_dim_resolved",
                "radii_resolved", "weights_resolved", "kernel", "max_neighbors", "n_steps")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"This diagnostic requires DF-PSM exports, not standalone DTM/GES output. Missing configuration fields: {', '.join(missing)}")
    config = dict(config)
    for name in ("body_basis_dim_resolved", "control_basis_dim_resolved", "max_neighbors", "n_steps"):
        value = config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Saved {name} must be a positive integer.")
    if "time_intervals" in config and (not isinstance(config["time_intervals"], int) or config["time_intervals"] <= 0):
        raise ValueError("Saved time_intervals must be positive when present.")
    radii, weights = config["radii_resolved"], config["weights_resolved"]
    if not isinstance(radii, list) or not isinstance(weights, list) or not radii or len(radii) != len(weights):
        raise ValueError("Saved radii_resolved/weights_resolved must be matching nonempty lists.")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in radii + weights):
        raise ValueError("Saved kernel radii and weights must be finite and positive.")
    if config["kernel"] not in ("c2", "c4"):
        raise ValueError("Saved kernel must be c2 or c4.")
    ridge = config.get("pullback_solve_ridge", 1e-6)
    if not math.isfinite(float(ridge)) or ridge < 0:
        raise ValueError("Saved pullback solve ridge must be finite and nonnegative.")
    for field, label in (("source_vtk", "source"), ("target_vtk", "target"), ("basis", "basis")):
        config[field] = str(_saved_file(config[field], match_dir, label, getattr(args, label)))
    output = Path(args.output_dir).expanduser().resolve()
    if output in (requested, match_dir) or (output / "config.json").is_file():
        raise ValueError("Recovery output must be separate from matching/workflow directories; matching config/summary files are protected.")
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
        raise ValueError(f"Recovery output must be new or empty unless --overwrite is explicit: {output}")

    source = load_mesh(config["source_vtk"], torch.device("cpu"), torch.float32)
    target = load_mesh(config["target_vtk"], torch.device("cpu"), torch.float32)
    D, R = config["body_basis_dim_resolved"], config["control_basis_dim_resolved"]
    U, _, metadata = load_mercer_basis(config["basis"], max(D, R), torch.device("cpu"), torch.float32)
    if U.shape[0] != source.points.shape[0] or U.shape[1] < max(D, R):
        raise ValueError("The fitted Mercer basis must have the original source vertex count and enough body/control modes.")
    validate_basis_metadata(metadata, radii, weights, config["kernel"], args.allow_basis_mismatch)
    alpha = _array(match_dir / "alpha_knots.npy", (D, 3), 2)
    _array(match_dir / "beta_knots.npy", (R, 3))
    q_path = match_dir / "trajectory_q_knots.npy"
    if not q_path.exists():
        q_path = match_dir / "trajectory_q.npy"
    v_path = match_dir / "trajectory_velocity_knots.npy"
    if not v_path.exists():
        v_path = match_dir / "trajectory_velocity.npy"
    q = _array(q_path, (source.points.shape[0], 3))
    _array(v_path, (source.points.shape[0], 3))
    if not np.array_equal(np.asarray(q[0], dtype=np.float32), source.points.numpy()):
        raise ValueError("Source geometry/order does not match the saved trajectory's initial source frame.")
    if config.get("loss") == "pointwise" and target.points.shape[0] != source.points.shape[0]:
        raise ValueError("A saved pointwise match requires equal source/target vertex counts.")

    # Where available, the saved deformed surface preserves the original connectivity.
    summary_file = match_dir / "summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        saved_mesh = summary.get("outputs", {}).get("deformed_vtk")
        if saved_mesh:
            for candidate in (Path(saved_mesh), match_dir / Path(saved_mesh).name):
                if candidate.is_file():
                    mesh = pv.read(candidate).triangulate()
                    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
                    if not matching_topology(source.faces.numpy(), faces):
                        raise ValueError("Source indexed topology does not match the saved matching endpoint connectivity.")
                    break
    try:
        requested_device = "cuda" if args.device == "auto" else args.device
        parsed_device = torch.device(requested_device)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Unsupported --device: {args.device}") from exc
    device = parsed_device if torch.cuda.is_available() else torch.device("cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Recovery supports cpu/cuda devices.")
    return dict(requested_match_dir=requested, match_dir=match_dir, output_dir=output, config=config, device=device,
                generated_alpha_samples=int(alpha.shape[0]), reference_q=q_path, reference_velocity=v_path)
