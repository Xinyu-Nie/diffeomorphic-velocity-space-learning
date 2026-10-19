# Diffeomorphic Velocity-Space Learning

Python implementation of **Diffeomorphic Velocity-Space Learning for Surface Matching and Generation**: Dual-Frame Pullback Surface Matching (DF-PSM) and Calibrated Geometry-Refined Velocity Diffusion (CalVel-Diff).

DF-PSM optimizes Eulerian coefficient paths and integrates the ambient flow and its differential. CalVel-Diff learns optimized paths. The package also includes GES/DTM comparators, surface preprocessing, OT initialization, Mercer bases, and alpha-recovery diagnostics.

## Installation

The validated configuration is Python **3.12**, PyTorch **2.7.0 / CUDA 12.8**, and a matching `torch-cluster` binary. Full-size GPU experiments and the Mercer constructor require CUDA. Small preprocessing, OT, matching, and generation checks can use CPU where supported.

From this repository root, create a separate environment:

```bash
conda env create -f environment.yml
conda activate dvsl
dvsl --help
```

Alternatively, with Python 3.12 and your environment activated:

```bash
python -m pip install -r requirements-cu128.txt
python -m pip install.
dvsl-match-df-psm --help
```

For development, use `python -m pip install -e .` after installing the binary dependencies. Install Excel support only if required: `python -m pip install '.[excel]'`.

`torch-cluster` must match the selected PyTorch and CUDA build. The pinned recipe uses only the needed cluster extension; it does not install all PyTorch Geometric extensions. See [installation](docs/installation.md) and [dependencies](docs/dependencies.md). We have not tested Python versions other than 3.12.

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

For a complete surface-matching example, use the [prepared BrainWeb ico5/ico6 surfaces](examples/brainweb/README.md). Four processed meshes and both cross-resolution OT surfaces are included; no surface preprocessing or OT computation is needed. The example helper builds and reuses a source Mercer basis, then runs the standard matching CLI:

```bash
python examples/brainweb/run_matching.py ico5-to-ico5
python examples/brainweb/run_matching.py ico6-to-ico6
python examples/brainweb/run_matching.py ico5-to-ico6
python examples/brainweb/run_matching.py ico6-to-ico5
```

The first two commands use pointwise attachment for 800 updates. The last two use 500 pointwise updates toward the supplied OT surface, followed by 800 varifold updates toward the real pial target. Run these commands from the repository root after installation; see the example guide for CUDA/memory requirements and expanded CLI commands.

## Data and Model Availability

Due to the data-use agreements and redistribution restrictions associated with the Alzheimer's Disease Neuroimaging Initiative (ADNI) and the Human Connectome Project (HCP), the training datasets and pretrained surface-generation models are not publicly distributed. However, all implementation code, including preprocessing, training, and inference scripts, is provided to support reproducibility.

To run the full surface generation workflows, users must provide their own triangular surface meshes. Researchers wishing to reproduce the experiments using ADNI or HCP data must obtain the necessary data access permissions from the respective projects.

## Workflow and numerical scope

See [surface matching and preparation](docs/workflows.md), [CalVel-Diff settings](docs/calvel-diff.md), and [numerical conventions and limitations](docs/numerical-conventions.md).

## License

MIT for the code; see [LICENSE](LICENSE). The derived BrainWeb example assets have separate [source attribution and data notes](examples/brainweb/NOTICE.md).
