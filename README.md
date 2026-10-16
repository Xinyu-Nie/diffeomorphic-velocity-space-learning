# Diffeomorphic Velocity-Space Learning

Python implementation of **Diffeomorphic Velocity-Space Learning for Surface Matching and Generation**: Dual-Frame Pullback Surface Matching (DF-PSM) and Calibrated Geometry-Refined Velocity Diffusion (CalVel-Diff).

DF-PSM optimizes Eulerian coefficient paths **beta**, integrates the ambient flow and its differential, and projects the realized pullback to obtain auxiliary **alpha**. CalVel-Diff learns optimized beta paths. The package also includes GES/DTM comparators, surface preprocessing, OT initialization, Mercer bases and alpha-recovery diagnostics.

## Installation

The validated configuration is Python **3.12**, PyTorch **2.7.0 / CUDA 12.8**, and a matching `torch-cluster` binary. Full-size GPU experiments and the Mercer constructor require CUDA. Small preprocessing, OT, matching and generation checks can use CPU where supported.

From this repository root, create a separate environment:

```bash
conda env create -f environment.yml
conda activate dvsl
dvsl --help
```

Alternatively, with Python 3.12 and your environment activated:

```bash
python -m pip install -r requirements-cu128.txt
python -m pip install .
dvsl-match-df-psm --help
```

For development use `python -m pip install -e .` after installing the binary dependencies. Install Excel support only if required: `python -m pip install '.[excel]'`.

`torch-cluster` must match the selected PyTorch and CUDA build. The pinned recipe uses only the needed cluster extension; it does not install all PyTorch Geometric extensions. See [installation](docs/installation.md) and [dependencies](docs/dependencies.md). Python versions other than 3.12 are currently outside the tested package range.

## Commands

All commands work after installation from any working directory. `python -m dvsl.cli COMMAND ...` provides the same workflows. Compatibility `.py` entry points are also provided in the source checkout.

| Command | Purpose |
|---|---|
| `dvsl-preprocess-corresponding` | Similarity alignment for trusted ordered vertices |
| `dvsl-preprocess-noncorresponding` | Independent normalization and smoothing |
| `dvsl-preprocess` | Explicit or topology/count-based routing |
| `dvsl-basis` | Unified Wendland Gram-Mercer basis |
| `dvsl-ot` | Sinkhorn barycentric OT initialization |
| `dvsl-match-df-psm` | DF-PSM, single or two stages |
| `dvsl-match-ges`, `dvsl-match-dtm` | In-house matching comparators |
| `dvsl-recover-alpha` | Approximate recovery from exported alpha |
| `dvsl-generate train`, `dvsl-generate sample` | Separate CalVel-Diff fitting and generation |

```bash
dvsl-match-df-psm --pairs template.vtk target.vtk --basis basis.npz --radii 0.19 --metric pointwise --output_dir runs/example
dvsl-generate train --help
dvsl-generate sample --help
```

Supply your own triangular meshes, exact basis assets and matching populations. Datasets, pretrained models, local reproduction outputs and validation harnesses are excluded. CLI defaults are generic; paper experiment budgets require explicit options.

## Workflow and numerical scope

See [surface matching and preparation](docs/workflows.md), [CalVel-Diff including hippocampal settings](docs/calvel-diff.md), and [numerical conventions and limitations](docs/numerical-conventions.md).

Component reproduction covered five cortical single-stage and five hippocampal two-stage fits at full budgets, ten alpha recoveries, fresh cortical/hippocampal training with 250/86 generated surfaces, one-fold benchmark comparisons excluding FID, two basis constructions, five OT maps, and preprocessing for ten subjects. These checks establish functionality in the validated environment, not full-cohort or ten-fold statistical reproduction. Recovery is approximate; positive sampled Jacobians do not guarantee an intersection-free discrete surface. Exact basis coordinates must be retained with existing beta paths and models.

## License

MIT; see [LICENSE](LICENSE).
