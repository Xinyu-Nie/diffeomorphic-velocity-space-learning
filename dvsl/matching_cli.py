"""Argument checks shared by the in-house surface-matching comparators."""
from __future__ import annotations

import math


def validate_comparator_arguments(args, algorithm: str) -> None:
    """Validate before mesh loading or output writes, retaining valid legacy defaults."""
    if algorithm not in ("ges", "dtm"):
        raise ValueError(f"Unknown comparator: {algorithm}")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name} must be finite.")
    positive = ["n_steps", "chamfer_chunk", "var_Ti", "var_Tj", "lbfgs_history_size", "trajectory_samples"]
    positive += ["max_neighbors"] if algorithm == "ges" else ["time_knots", "gradient_tile_q", "gradient_tile_s"]
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    for name in ("radius", "varifold_radius"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.adam_iters < 0 or args.lbfgs_iters < 0:
        raise ValueError("Optimizer iteration counts cannot be negative.")
    if args.adam_lr <= 0 or args.lbfgs_lr <= 0 or args.adam_weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative.")
    for name, value in vars(args).items():
        if (name.startswith("lambda_") or name in ("m0_reg", "m0_rkhs_scale", "w_data")) and value < 0:
            raise ValueError(f"--{name} cannot be negative.")
    uses_varifold = args.loss == "varifold" or getattr(args, "evaluate_varifold", False)
    if uses_varifold and args.varifold_backend == "stable" and args.var_amp:
        raise ValueError("Stable varifold requires full precision; omit --var_amp.")
    if args.strict_pointwise_topology and args.loss != "pointwise":
        raise ValueError("--strict_pointwise_topology only applies to pointwise attachment.")
    if algorithm == "ges":
        for name in ("shooting_radii", "shooting_weights"):
            values = getattr(args, name)
            if values is not None and (not values or any(not math.isfinite(v) for v in values)):
                raise ValueError(f"--{name} requires finite values.")
        radii = args.shooting_radii or [args.radius]
        weights = args.shooting_weights
        if any(r <= 0 for r in radii):
            raise ValueError("Kernel radii must be positive.")
        if weights is not None and (len(weights) != len(radii) or any(w < 0 for w in weights) or sum(weights) <= 0):
            raise ValueError("Kernel weights must match radii, be nonnegative and have a positive sum.")
        if args.divergence_factor is not None and args.divergence_factor <= 1:
            raise ValueError("--divergence_factor must be greater than 1.")
    else:
        if args.batch_size != 1:
            raise ValueError("Direct per-pair optimization requires --batch_size 1.")
        if args.loss == "varifold" and args.varifold_radius is None and args.radius is None:
            raise ValueError("DTM varifold requires --varifold_radius or --kernel_radius, as in the legacy adapter.")
