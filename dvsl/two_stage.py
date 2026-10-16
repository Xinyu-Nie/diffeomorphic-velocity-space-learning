"""Two-stage OT-guided orchestration around the unchanged single-stage solvers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyvista as pv

from .ot_inputs import validate_mesh_arrays


SPECS = {
    "df_psm": dict(module="surface_matching", parser="build_parser", method="DF-PSM",
                   state="beta_knots.npy", initial="initial_beta_knots.npy", init_option="init_beta", weight="lambda_rec"),
    "ges": dict(module="ges", parser="build_arg_parser", method="GES",
                state="m0_density.npy", initial="initial_m0_density.npy", init_option="init_m0", weight="w_data"),
    "dtm": dict(module="dtm", parser="build_arg_parser", method="DTM",
                state="alpha_knots.npy", initial="initial_alpha_knots.npy", init_option="init_alpha", weight="lambda_rec"),
}


def add_two_stage_arguments(parser):
    group = parser.add_argument_group("OT-guided two-stage matching")
    group.add_argument("--two-stage", action="store_true",
                       help="Pointwise fit to the OT surface, then warm-start matching to the real target.")
    inputs = group.add_mutually_exclusive_group()
    inputs.add_argument("--ot-target", help="MeshOTmap corresponding VTK or NPZ for this real target.")
    inputs.add_argument("--ot-dir", help="Find <real-target-stem>__ot_T_of_x.vtk or __ot.npz here.")
    group.add_argument("--allow-ot-name-mismatch", action="store_true",
                       help="Explicitly accept renamed OT input; all geometry/count checks still apply.")
    for stage in (1, 2):
        group.add_argument(f"--stage{stage}-adam-iters", type=int, default=None,
                           help="Stage-1 default: 500; stage-2 default: --adam_iters.")
        group.add_argument(f"--stage{stage}-lbfgs-iters", type=int, default=None,
                           help="Stage-1 default: 0; stage-2 default: --lbfgs_iters.")
        group.add_argument(f"--stage{stage}-adam-lr", type=float, default=None,
                           help="Override --adam_lr for this stage.")
        group.add_argument(f"--stage{stage}-attachment-weight", type=float, default=None,
                           help="Override this stage's attachment weight; other weights are shared.")
    return parser


def _two_stage_options_used(args):
    return args.ot_target or args.ot_dir or args.allow_ot_name_mismatch or any(
        getattr(args, f"stage{stage}_{suffix}") is not None
        for stage in (1, 2) for suffix in ("adam_iters", "lbfgs_iters", "adam_lr", "attachment_weight")
    )


def _existing_file(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not an existing file: {path}")
    return path


def _read_mesh(path, label):
    try:
        mesh = pv.read(str(path)).triangulate()
        packed = np.asarray(mesh.faces, dtype=np.int64)
        if packed.size % 4 or not packed.size:
            raise ValueError("no usable triangular faces")
        cells = packed.reshape(-1, 4)
        if not np.all(cells[:, 0] == 3):
            raise ValueError("mesh is not a triangular surface")
        validate_mesh_arrays(np.asarray(mesh.points), cells[:, 1:], path)
        return mesh
    except Exception as exc:
        raise ValueError(f"Cannot read {label} '{path}': {exc}") from exc


def _read_map(path, source_count, target_count):
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ("T_of_x", "Ns", "Nt"):
                if key not in data:
                    raise ValueError(f"missing MeshOTmap array '{key}'")
            points = np.asarray(data["T_of_x"])
            for key, expected in (("Ns", source_count), ("Nt", target_count)):
                value = np.asarray(data[key])
                if value.ndim != 0 or value.dtype.kind not in "iu" or int(value) != expected:
                    raise ValueError(f"{key} must equal {expected} for the supplied source/real-target pair")
            if points.shape != (source_count, 3) or points.dtype.kind not in "fiu" or not np.isfinite(points).all():
                raise ValueError(f"T_of_x must be finite with shape ({source_count}, 3)")
            return points.astype(np.float64, copy=True)
    except Exception as exc:
        raise ValueError(f"Invalid MeshOTmap NPZ '{path}': {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_two_stage(args, algorithm, module):
    """Check pairing/source/basis and construct both namespaces before any output writes."""
    if not (args.ot_target or args.ot_dir):
        raise ValueError("--two-stage requires --ot-target FILE or --ot-dir DIRECTORY; OT preprocessing is not run automatically.")
    for field in ("init_beta", "init_alpha", "init_m0"):
        if getattr(args, field, None):
            raise ValueError(f"--two-stage manages initialization; remove --{field}.")
    for field in ("resume_checkpoint", "ablation_zero_init"):
        if getattr(args, field, False):
            raise ValueError(f"--{field} is incompatible with the automatic two-stage warm start.")
    if getattr(args, "init_scale", 1.) != 1.:
        raise ValueError("Two-stage handoff requires --init_scale 1; initialization must not be rescaled.")
    source = _existing_file(args.pairs[0], "Original source")
    target = _existing_file(args.pairs[1], "Real target")
    source_mesh = _read_mesh(source, "original source")
    target_mesh = _read_mesh(target, "real target")
    source_count, target_count = source_mesh.n_points, target_mesh.n_points
    if args.loss == "pointwise" and source_count != target_count:
        raise ValueError("Stage-2 pointwise attachment requires equal original-source and real-target vertex counts.")
    if args.strict_pointwise_topology and args.loss != "pointwise":
        raise ValueError("--strict_pointwise_topology applies to stage 2 only when --metric pointwise; stage 1 always checks OT topology.")
    if args.strict_pointwise_topology and not np.array_equal(source_mesh.faces, target_mesh.faces):
        raise ValueError("Stage-2 --strict_pointwise_topology requires original-source/real-target connectivity to match.")

    if args.ot_dir:
        directory = Path(args.ot_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"--ot-dir is not an existing directory: {directory}")
        options = [directory / f"{target.stem}__ot_T_of_x.vtk", directory / f"{target.stem}__ot.npz"]
        ot_input = next((path for path in options if path.is_file()), None)
        if ot_input is None:
            raise ValueError(f"No MeshOTmap output for real target '{target.stem}' in {directory}; expected {options[0].name} or {options[1].name}.")
    else:
        ot_input = _existing_file(args.ot_target, "OT input")
    if ot_input.suffix.lower() not in (".vtk", ".npz"):
        raise ValueError("--ot-target must be a MeshOTmap corresponding .vtk or .npz file.")
    if ot_input in (source, target):
        raise ValueError("OT input must be distinct from the original source and the real target.")
    expected = f"{target.stem}__ot_T_of_x.vtk" if ot_input.suffix.lower() == ".vtk" else f"{target.stem}__ot.npz"
    if ot_input.name.casefold() != expected.casefold() and not args.allow_ot_name_mismatch:
        raise ValueError(f"OT input basename does not pair with real target '{target.name}'; expected '{expected}'. For a deliberately renamed map use --allow-ot-name-mismatch.")

    companion = None
    map_points = None
    if ot_input.suffix.lower() == ".npz":
        map_points = _read_map(ot_input, source_count, target_count)
    else:
        ot_mesh = _read_mesh(ot_input, "OT corresponding surface")
        if ot_mesh.n_points != source_count:
            raise ValueError(f"OT corresponding surface has {ot_mesh.n_points} vertices; original source has {source_count}.")
        if not np.array_equal(ot_mesh.faces, source_mesh.faces):
            raise ValueError("OT corresponding surface must preserve the original source's triangular connectivity and vertex order.")
        if ot_input.stem.lower().endswith("__ot_t_of_x"):
            candidate = ot_input.with_name(ot_input.stem[:-len("__ot_T_of_x")] + "__ot.npz")
            if candidate.is_file():
                companion = candidate
                saved_points = _read_map(companion, source_count, target_count)
                if not np.allclose(ot_mesh.points, saved_points, rtol=1e-6, atol=1e-7):
                    raise ValueError("OT VTK and its companion NPZ T_of_x disagree; use the correctly paired MeshOTmap outputs.")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Two-stage --output_dir must be new or empty; preserving existing work: {output}")
    basis = None
    if algorithm != "ges":
        basis = _existing_file(args.basis, "Template Mercer basis")
        import torch
        if algorithm == "df_psm":
            control_dim = args.body_basis_dim if args.control_basis_dim <= 0 else args.control_basis_dim
            loaded_dim = max(args.body_basis_dim, control_dim)
            U, _, metadata = module.load_mercer_basis(str(basis), loaded_dim, torch.device("cpu"), torch.float64)
            if U.shape[1] < loaded_dim:
                raise ValueError(f"Basis supplies {U.shape[1]} modes but {loaded_dim} are required.")
            radii, weights = module.resolve_kernel_mixture(args.radius, args.radii, args.weights)
            module.validate_basis_metadata(metadata, radii, weights, args.kernel, args.allow_basis_mismatch)
        else:
            U, _, metadata = module.load_mercer_basis(str(basis), args.basis_dim, torch.device("cpu"), torch.float64)
            kernel = metadata.get("kernel", "c2") if args.kernel == "auto" else args.kernel
            radius = metadata.get("R") if args.radius is None else args.radius
            if kernel not in ("c2", "c4") or radius is None or not math.isfinite(float(radius)) or float(radius) <= 0:
                raise ValueError("DTM basis/arguments must resolve a valid C2/C4 body kernel and positive radius.")
        if U.shape[0] != source_count:
            raise ValueError(f"Basis has {U.shape[0]} template vertices but original source has {source_count}.")

    stage1 = copy.deepcopy(args)
    stage2 = copy.deepcopy(args)
    spec = SPECS[algorithm]
    for number, namespace in ((1, stage1), (2, stage2)):
        namespace.pairs = [str(source), str(target)]
        namespace.output_dir = str(output / ("stage1_ot" if number == 1 else "stage2_target"))
        if basis is not None:
            namespace.basis = str(basis)
        for optimizer, fallback in (("adam", 500 if number == 1 else args.adam_iters),
                                    ("lbfgs", 0 if number == 1 else args.lbfgs_iters)):
            value = getattr(args, f"stage{number}_{optimizer}_iters")
            value = fallback if value is None else value
            if value < 0:
                raise ValueError(f"--stage{number}-{optimizer}-iters cannot be negative.")
            setattr(namespace, f"{optimizer}_iters", value)
        lr = getattr(args, f"stage{number}_adam_lr")
        if lr is not None:
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError(f"--stage{number}-adam-lr must be finite and positive.")
            namespace.adam_lr = lr
        weight = getattr(args, f"stage{number}_attachment_weight")
        if weight is not None:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"--stage{number}-attachment-weight must be finite and nonnegative.")
            setattr(namespace, spec["weight"], weight)
    stage1.loss = "pointwise"
    stage1.strict_pointwise_topology = True
    stage1.pairs[1] = str(ot_input if map_points is None else output / "inputs" / f"{target.stem}__ot_T_of_x.vtk")
    setattr(stage2, spec["init_option"], str(output / "stage1_ot" / spec["state"]))
    if algorithm == "ges":
        stage2.init_representation = "density"
    for namespace in (stage1, stage2):
        if algorithm == "df_psm":
            module.validate_arguments(namespace)
        else:
            from .matching_cli import validate_comparator_arguments
            validate_comparator_arguments(namespace, algorithm)
    tracked = {"source": source, "real_target": target, "ot_input": ot_input}
    if companion is not None:
        tracked["ot_companion"] = companion
    if basis is not None:
        tracked["basis"] = basis
    if algorithm == "ges" and args.mass_npy:
        tracked["mass_npy"] = _existing_file(args.mass_npy, "Source vertex masses")
    return dict(source_mesh=source_mesh, map_points=map_points, output=output,
                stage1=stage1, stage2=stage2, tracked=tracked,
                input_hashes={key: _sha256(path) for key, path in tracked.items()})


def namespace_to_argv(parser, namespace):
    """Serialize only original solver actions; workflow fields never enter solver config/checkpoints."""
    arguments = []
    for action in parser._actions:
        if action.dest == "help" or not action.option_strings:
            continue
        value = getattr(namespace, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                arguments.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                arguments.append(action.option_strings[0])
        elif value is not None:
            arguments.append(action.option_strings[0])
            if isinstance(value, (list, tuple)):
                arguments.extend(str(item) for item in value)
            else:
                arguments.append(str(value))
    return arguments


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def execute_two_stage(prepared, algorithm, parser):
    spec = SPECS[algorithm]
    root = prepared["output"]
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RuntimeError(f"Two-stage output directory changed after validation; preserving existing work: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if prepared["map_points"] is not None:
        path = Path(prepared["stage1"].pairs[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = prepared["source_mesh"].copy(deep=True)
        mesh.points = prepared["map_points"]
        mesh.save(path)
    command = {"df_psm": "match-df-psm", "dtm": "match-dtm", "ges": "match-ges"}[algorithm]
    commands = {
        f"stage{number}": [sys.executable, "-B", "-u", "-m", "dvsl.cli", command] + namespace_to_argv(parser, prepared[f"stage{number}"])
        for number in (1, 2)
    }
    config = {
        "algorithm": spec["method"], "original_source": prepared["stage1"].pairs[0],
        "real_target": prepared["stage2"].pairs[1], "ot_input": str(prepared["tracked"]["ot_input"]),
        "ot_surface": prepared["stage1"].pairs[1], "commands": commands,
        "input_sha256": prepared["input_hashes"],
        "optimizer_state": "Fresh optimizer and RNG initialization for each stage; only the saved optimized parameters are transferred.",
        "initialization": {"representation": "momentum_density" if algorithm == "ges" else ("beta" if algorithm == "df_psm" else "alpha"),
                           "state_path": getattr(prepared["stage2"], spec["init_option"]), "scale": 1.},
    }
    _write_json(root / "two_stage_config.json", config)
    report = {"status": "running", "algorithm": spec["method"], "config": str(root / "two_stage_config.json"), "stages": {}}
    _write_json(root / "two_stage_summary.json", report)
    try:
        for number in (1, 2):
            for key, path in prepared["tracked"].items():
                if _sha256(path) != prepared["input_hashes"][key]:
                    raise RuntimeError(f"Input '{key}' changed between preparation/stages; refusing an inconsistent source/basis/target handoff.")
            print(f"\n{spec['method']} two-stage matching: stage {number}/2", flush=True)
            result = subprocess.run(commands[f"stage{number}"], check=False,
                                    stdout=sys.stdout, stderr=sys.stderr,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise RuntimeError(f"Stage {number} exited with code {result.returncode}; prior stage outputs are preserved in {root}.")
            folder = Path(prepared[f"stage{number}"].output_dir)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status", "success") != "success":
                raise RuntimeError(f"Stage {number} reported status '{summary['status']}'; automatic continuation requires a successful stage.")
            report["stages"][f"stage{number}"] = {"summary": str(folder / "summary.json"),
                "metric": prepared[f"stage{number}"].loss, "adam_iters": prepared[f"stage{number}"].adam_iters,
                "lbfgs_iters": prepared[f"stage{number}"].lbfgs_iters, "best_objective": summary["best_objective"]}
            if number == 1:
                state = np.load(folder / spec["state"], allow_pickle=False)
                if not np.isfinite(state).all():
                    raise RuntimeError("Stage-1 optimized state contains non-finite values.")
            else:
                expected = np.load(Path(prepared["stage1"].output_dir) / spec["state"], allow_pickle=False)
                actual = np.load(folder / spec["initial"], allow_pickle=False)
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    raise RuntimeError("Stage-2 initialization differs from the saved stage-1 optimized state.")
                report.update(status="success", outputs=summary["outputs"], metrics=summary.get("metrics", {}),
                              warm_start_verified=True, original_source=config["original_source"])
            _write_json(root / "two_stage_summary.json", report)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        _write_json(root / "two_stage_summary.json", report)
        raise
    print(f"\nTwo-stage matching finished: {root / 'two_stage_summary.json'}", flush=True)
    return report


def run_cli(algorithm, argv=None):
    """Keep the existing single-stage main untouched; opt into orchestration explicitly."""
    spec = SPECS[algorithm]
    module = importlib.import_module(f"dvsl.{spec['module']}")
    base_parser = getattr(module, spec["parser"])()
    parser = add_two_stage_arguments(getattr(module, spec["parser"])())
    args = parser.parse_args(argv)
    if not args.two_stage:
        if _two_stage_options_used(args):
            parser.error("OT inputs and stage-specific options require --two-stage.")
        if argv is None:
            module.main()
        else:
            previous = sys.argv
            sys.argv = [previous[0], *argv]
            try:
                module.main()
            finally:
                sys.argv = previous
        return 0
    try:
        prepared = prepare_two_stage(args, algorithm, module)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        execute_two_stage(prepared, algorithm, base_parser)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Two-stage matching failed: {exc}", file=sys.stderr)
        return 1
    return 0
