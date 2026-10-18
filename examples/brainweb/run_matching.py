"""Run the four prepared BrainWeb examples through the installed DVSL commands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CASES = ("ico5-to-ico5", "ico6-to-ico6", "ico5-to-ico6", "ico6-to-ico5")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_arguments(args):
    """Return standard DVSL argv lists, with no surface preprocessing step."""
    source_level, target_level = args.case.split("-to-")
    source = HERE / "surfaces" / f"lh_inflated_{source_level}.vtk"
    target = HERE / "surfaces" / f"lh_pial_{target_level}.vtk"
    cap = 1024 if source_level == "ico5" else 4096
    rank = 1536 if source_level == "ico5" else 4096
    basis = (args.basis or args.basis_dir / f"lh_inflated_{source_level}_k4096.npz").resolve()
    output = (args.output_dir or HERE.parents[1] / "outputs" / "brainweb" / args.case).resolve()
    base = [sys.executable, "-B", "-u", "-m", "dvsl.cli"]
    basis_command = base + ["basis", "--src", str(source), "--radii", "0.19", "--weights", "1",
        "--kernel", "c2", "--max_neighbors", str(cap), "--k_eig", "4096", "--oversample", "64",
        "--iters", "8", "--dtype", "fp32", "--storage_dtype", "float32", "--device", "cuda",
        "--seed", "12345", "--out", str(basis)]
    matching = base + ["match-df-psm", "--pairs", str(source), str(target), "--basis", str(basis),
        "--body_basis_dim", str(rank), "--control_basis_dim", str(rank),
        "--radii", "0.19", "--weights", "1", "--kernel", "c2", "--max_neighbors", str(cap),
        "--time_intervals", "12", "--n_steps", "12", "--integrator", "midpoint",
        "--field_chunk_size", str(args.field_chunk_size), "--cuda_cache_trim_gib", "24",
        "--pullback_stride", "1", "--geometry_stride", "3", "--pullback_solve_ridge", "1e-6",
        "--jacobian_floor", "0.05", "--lambda_rec", "1000", "--lambda_smooth", "0.01",
        "--lambda_action", "0.0001", "--lambda_pullback", "3", "--lambda_alpha_energy", "0.5",
        "--lambda_alpha_time", "0.5", "--lambda_beta_time", "1.25", "--lambda_jacobian", "1",
        "--lambda_edge", "0.01", "--adam_iters", str(args.adam_iters), "--adam_lr", "0.002",
        "--adam_weight_decay", "0", "--lbfgs_iters", "0", "--grad_clip", "10",
        "--checkpoint_every", "25", "--log_every", "10", "--device", "cuda", "--dtype", "float32",
        "--seed", "1234", "--output_dir", str(output)]
    if source_level == target_level:
        matching += ["--metric", "pointwise", "--strict_pointwise_topology"]
    else:
        ot = HERE / "ot" / args.case.replace("-", "_") / f"lh_pial_{target_level}__ot_T_of_x.vtk"
        matching += ["--metric", "varifold", "--varifold_backend", "stable", "--varifold_radius", "0.16",
            "--varifold_kernel", "c2", "--two-stage", "--ot-target", str(ot),
            "--stage1-adam-iters", str(args.stage1_iters), "--stage2-adam-iters", str(args.adam_iters)]
    return basis, output, basis_command, matching


def run(command):
    print("\n" + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--basis", type=Path, help="Reuse an existing basis for this exact source.")
    parser.add_argument("--basis-dir", type=Path, default=HERE / "basis")
    parser.add_argument("--output-dir", type=Path, help="New/empty directory; default: outputs/brainweb/CASE.")
    parser.add_argument("--adam-iters", type=int, default=800, help="Single-stage or stage-2 updates.")
    parser.add_argument("--stage1-iters", type=int, default=500)
    parser.add_argument("--field-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--dry-run", action="store_true", help="Print standard CLI commands without computing or writing.")
    args = parser.parse_args(argv)
    if min(args.adam_iters, args.stage1_iters) < 0 or args.field_chunk_size <= 0:
        parser.error("Iteration counts must be nonnegative; field chunk size must be positive.")
    basis, output, basis_command, matching = command_arguments(args)
    if args.dry_run:
        print("# Build once if the basis is missing:\n" + shlex.join(basis_command))
        print("# Match:\n" + shlex.join(matching))
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be new or empty: {output}")
    manifest = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    for relative, entry in manifest["files"].items():
        path = HERE / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            parser.error(f"Missing or modified illustrative asset: {relative}")
    if args.basis and not basis.is_file():
        parser.error(f"Requested existing basis is missing: {basis}")
    if not basis.is_file():
        basis.parent.mkdir(parents=True, exist_ok=True)
        run(basis_command)
    import numpy as np
    with np.load(basis, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
    source_level = args.case.split("-to-")[0]
    expected = manifest["files"][f"surfaces/lh_inflated_{source_level}.vtk"]["sha256"]
    if meta.get("source_file_sha256") != expected:
        parser.error("Basis source hash does not match this example; build a basis from the supplied source.")
    run(matching)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
