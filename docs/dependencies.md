# Dependencies

The runtime list comes from imports in the production `dvsl` modules, including imports inside functions and installed workflow checks. It is not a full export of the research environment.

| Direct dependency | Tested version | Why it is needed |
|---|---|---|
| NumPy | 2.3.5 | Arrays, OT, geometry, PCA |
| SciPy | 1.17.0 | Sparse geometry, spatial queries, numerical routines |
| PyTorch | 2.7.0+cu128 | Differentiation, integration, optimization and diffusion |
| PyVista | 0.46.5 | Mesh I/O and geometry |
| VTK | 9.5.2 | Direct preprocessing I/O and triangle intersection helpers |
| torch-cluster | 1.6.3+pt27cu128 | Compact-support radius neighborhoods |
| torchdiffeq | 0.2.5 | GES Hamiltonian shooting / adjoint integration |
| threadpoolctl | 3.6.0 | PCA/codec BLAS thread limits |

Optional: `openpyxl` 3.1.5 for the existing Excel quality-report helper (`excel` extra). It is not required for CSV/JSON reports or the tested core workflows.

Pip will still install dependencies of these eight libraries, such as PyTorch's support libraries and PyVista's plotting/I/O dependencies. A library not directly imported by DVSL may therefore still be necessary transitively. Do not manually uninstall transitive dependencies from a shared environment.

`pyproject.toml` defines the installable runtime requirements. `requirements.txt` mirrors the compatible ranges; `requirements-cu128.txt` pins the validated direct versions and binary sources. `environment.yml` supplies Python and pip, then uses that pinned recipe. 
