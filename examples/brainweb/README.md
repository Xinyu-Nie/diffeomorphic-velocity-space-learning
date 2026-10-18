# Prepared BrainWeb cortical matching examples

These examples match processed inflated sources to processed left pial targets reconstructed with FreeSurfer from a synthetic BrainWeb T1 image. 
See [source attribution](NOTICE.md) and [asset provenance and SHA-256 hashes](provenance.json).

## Install and run

Clone/download this repository and install the package with its `environment.yml`.
If upgrading an existing installation, run `python -m pip install --no-deps .`
from the repository root. The example requires version 0.1.1 or later.

```bash
conda env create -f environment.yml
conda activate dvsl
python examples/brainweb/run_matching.py ico5-to-ico5
python examples/brainweb/run_matching.py ico6-to-ico6
python examples/brainweb/run_matching.py ico5-to-ico6
python examples/brainweb/run_matching.py ico6-to-ico5
```

Run the matching commands sequentially. They work in PowerShell and Bash as
written. No surface preprocessing, original T1 image, or OT computation
is required. All six VTK assets are already prepared and checked by SHA-256.
Examples and assets are included in the repository and source distribution;
they are not bundled in the installed wheel.

The first use of each source constructs a 4,096-mode Mercer basis on CUDA in
`examples/brainweb/basis/`. Later commands reuse it. This is a model basis
construction step, not mesh preprocessing. The basis files are large and are
intentionally not committed. Use `--basis-dir PATH` to select another cache or
`--basis FILE` to reuse a basis built from the exact supplied source. The helper
checks the basis's recorded source hash. Do not substitute an ADNI basis.

Outputs default to `outputs/brainweb/CASE/`. Use `--output-dir PATH` to select a
new/empty directory. The helper refuses to overwrite an existing run.

| Command case | Source → real target | Attachment | Adam updates |
|---|---|---|---|
| `ico5-to-ico5` | 10,242 → 10,242 vertices | Pointwise, strict identical connectivity | 800 |
| `ico6-to-ico6` | 40,962 → 40,962 vertices | Pointwise, strict identical connectivity | 800 |
| `ico5-to-ico6` | 10,242 → 40,962 vertices | Pointwise to OT; then varifold to real target | 500 + 800 |
| `ico6-to-ico5` | 40,962 → 10,242 vertices | Pointwise to OT; then varifold to real target | 500 + 800 |

The same-resolution vertex orders correspond. A cross-resolution OT surface has
the **source's** vertex order/connectivity and target barycentric positions; it
is an initialization target, not the final anatomical target. Stage 2 starts
again from the original inflated source, warm-started with the optimized
stage-1 beta coefficients and a fresh optimizer. The original pial surface is
used for its varifold attachment. 

Both source and target resolutions share the ico6-derived normalized frame.
The same physical transforms were applied at each resolution. Distances are in
normalized coordinates, not millimeters. Do not renormalize or smooth these files
before using the included OT maps or a cached basis.

## Settings and standard CLI commands

The helper only assembles standard `python -m dvsl.cli basis` and
`python -m dvsl.cli match-df-psm` commands. Inspect the complete commands, with no
computation or writes:

```bash
python examples/brainweb/run_matching.py ico5-to-ico5 --dry-run
python examples/brainweb/run_matching.py ico6-to-ico6 --dry-run
python examples/brainweb/run_matching.py ico5-to-ico6 --dry-run
python examples/brainweb/run_matching.py ico6-to-ico5 --dry-run
```

The printed matching arguments can also be passed to `dvsl-match-df-psm`.
All examples use Wendland C2, radius 0.19, 12 temporal intervals/13 coefficient
samples, 12 midpoint integration steps, learning rate 0.002, zero initialization,
and no L-BFGS. The source determines the body/control rank: 1,536 for ico5 and
4,096 for ico6. Neighbor caps are 1,024 and 4,096 respectively. Regularization
weights are explicit in the helper, rather than relying on generic CLI defaults.
Cross-resolution stage 2 uses full-face, separately area-normalized C2 varifold
attachment with radius 0.16 and the stable full-precision backend.

For a short functional check, override the budgets explicitly:

```bash
python examples/brainweb/run_matching.py ico5-to-ico5 --adam-iters 2 --output-dir outputs/brainweb/check-single
python examples/brainweb/run_matching.py ico5-to-ico6 --stage1-iters 1 --adam-iters 1 --output-dir outputs/brainweb/check-two-stage
```

These overrides do not reproduce the default 800-update examples. Both stages
export the best evaluated coefficients, fitted VTK, trajectory, alpha/beta paths,
optimization history and numerical/mesh-quality summaries. Cross-resolution
results are in `stage1_ot/` and `stage2_target/`, with a checked warm-start handoff
record in `two_stage_summary.json`.

## Memory and numerical scope

Use the validated CUDA environment with a GPU and sufficient system RAM.
The helper enables
`--field_chunk_size 2000000`, which chunks field edge tensors and recomputes them
during backward, and `--cuda_cache_trim_gib 24`, which releases inactive allocator
cache between Adam updates if reserved memory exceeds 24 GiB. The latter is a
cache threshold, **not** a memory limit or a minimum GPU requirement. This mode
preserves the graph union, support radius, ranks, attachment and objective;
floating-point accumulation order can introduce small numerical differences.
Other CLI users retain legacy evaluation by default (`--field_chunk_size 0`).

The previous full ico6→ico5 run with this memory strategy completed 500 + 800
updates with approximately 15.42 GiB peak allocated and 18.38 GiB peak. `--field-chunk-size` can reduce field block size further;
the basis, graph indices, trajectories and attachment still consume memory.

The processed pial targets retain small triangle intersections detected by the
existing quality checker (96 ico5 and 587 ico6 intersection pairs under its
default tolerance). They are supplied unchanged; they should not be described
as intersection-free ground truth. Both processed inflated sources have zero
detected intersection pairs and no degenerate faces. 



## Optional OT regeneration

The precomputed maps use area-weighted entropic Sinkhorn OT, squared Euclidean
cost, epsilon 0.01, at most 2,000 iterations, and stopping tolerance 1e-9.
Barycentric positions use the full transport plan; top-32 arrays are ancillary.
If you want to recompute them, write to separate output directories:

```bash
dvsl-ot --source examples/brainweb/surfaces/lh_inflated_ico5.vtk --targets examples/brainweb/surfaces/lh_pial_ico6.vtk --output-dir outputs/brainweb/ot_ico5_to_ico6 --eps 0.01 --max-iter 2000 --tol 1e-9 --topk 32 --workers 1
dvsl-ot --source examples/brainweb/surfaces/lh_inflated_ico6.vtk --targets examples/brainweb/surfaces/lh_pial_ico5.vtk --output-dir outputs/brainweb/ot_ico6_to_ico5 --eps 0.01 --max-iter 2000 --tol 1e-9 --topk 32 --workers 1
```

OT regeneration is CPU/RAM intensive and unnecessary for the included examples.
The helper deliberately uses the bundled maps and validates their hashes.
