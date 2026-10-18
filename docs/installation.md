# Installation

Run the commands from the repository root. `environment.yml` creates a new `dvsl` environment with Python 3.12 and installs the package using `requirements-cu128.txt`. It does not modify an existing research environment.

```bash
conda env create -f environment.yml
conda activate dvsl
python -c "import dvsl, torch; print(dvsl.__version__, torch.__version__, torch.cuda.is_available())"
dvsl --help
```

Without Conda, create and activate a Python 3.12 virtual environment first:

```bash
python -m venv .venv
```

PowerShell activation: `.venv\Scripts\Activate.ps1`. Linux activation: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements-cu128.txt
python -m pip install.
```

The pinned file records the eight direct dependency versions used during reproduction. Pip installs their necessary transitive dependencies too; this file is not a full transitive lock. A compatible NVIDIA driver is required for GPU execution. 

For another PyTorch/CUDA build, first install PyTorch using its [official version instructions](https://pytorch.org/get-started/previous-versions/), then install a matching `torch-cluster` wheel from the [PyG wheel index](https://data.pyg.org/whl/). See the [PyG binary compatibility guidance](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Do not mix wheels built against different Torch/CUDA versions. Package metadata currently restricts Torch to 2.7.x; compatibility outside this series has not been tested.

To install only the cluster extension for the validated combination:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch-cluster==1.6.3+pt27cu128 --only-binary=:all: --no-index --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
python -m pip install.
```

If no binary exists for your OS/Python, select a documented compatible combination or explicitly build the extension. The pinned recipe refuses a Torch/cluster source build. CPU mode is available for small supported workflows through `--device cpu`; the Mercer constructor remains CUDA-only.

Optional Excel export: `python -m pip install '.[excel]'`. CSV/JSON outputs do not need this extra. Build tooling (`setuptools`, `wheel`) is separate from runtime dependencies.
