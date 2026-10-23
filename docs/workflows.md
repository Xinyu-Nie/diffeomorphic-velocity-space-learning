# Surface preparation and matching

Use legacy VTK POLYDATA triangle meshes. Preprocessing preserves vertex order and connectivity and performs no remeshing or topology repair. Source/template geometry must remain consistent across basis construction, OT and fitting.

## Preprocessing

```bash
dvsl-preprocess-corresponding --source template.vtk --targets subject.vtk --output-dir prepared/corresponding
dvsl-preprocess-noncorresponding --source template.vtk --targets subject.vtk --output-dir prepared/independent --smooth-iterations 20 --smooth-lambda 0.2
```

Corresponding mode requires trusted ordered vertices. It area-centers and RMS-normalizes the target, then fits a proper similarity transform to the supplied source; no reflection, no additional smoothing by default. Independent mode area-centers each target and applies `2/largest_axis_extent`, then defaults to 20 uniform Laplacian iterations at 0.2. 
## Mercer basis

```bash
dvsl-basis --src template.vtk --radii 0.19 --kernel c2 --k_eig 4096 --max_neighbors 512 --out basis.npz
```

CUDA and matching `torch-cluster` are required. The example rank must fit your source vertex count. Rank is not silently reduced. Increase the neighbor cap when required; truncating kernel neighborhoods changes the operator. Multiple radii and positive normalized mixture weights are supported. Current DTM consumes one radius only; GES does not use a Mercer basis. Keep the exact basis used by existing coefficients and models.

## OT initialization

```bash
dvsl-ot --source template.vtk --targets subject.vtk --output-dir ot --eps 0.01 --max-iter 2000 --topk 32
```

This produces `<target_stem>__ot_T_of_x.vtk` and corresponding NPZ arrays. `--targets` accepts one VTK file or a text list. The archived hippocampal tests used 2,000 iterations; the CLI default is 1,000. The dense area-weighted Sinkhorn computation runs on CPU. The full-plan barycenter and renormalized top-k sparse transport weights are distinct exports. The absolute scaling-vector stopping criterion can remain unsatisfied despite small transport marginal errors.

## Matching

```bash
dvsl-match-df-psm --pairs template.vtk subject.vtk --basis basis.npz --radii 0.19 --metric pointwise --output_dir runs/single
dvsl-match-df-psm --pairs template.vtk subject.vtk --basis basis.npz --radii 1.0 --metric varifold --varifold_radius 0.16 --varifold_backend stable --two-stage --ot-dir ot --stage1-adam-iters 500 --stage2-adam-iters 2000 --output_dir runs/two_stage
```

Examples show workflow syntax; specify ranks, knots, integration settings, and all regularization weights. `--two-stage` fits the OT corresponding surface pointwise, then fits the real target, using the original source in both stages and a verified optimized-state handoff.

DF-PSM attachments: `pointwise`, `chamfer`, `surface_chamfer`, `varifold`. GES/DTM expose pointwise, vertex Chamfer, and varifold. `--metric` accepts the legacy `--loss` alias. All three have `--help` and two-stage support. Matching exports include controls, trajectories, configuration, history, summary, and deformed VTK. Use fresh output directories.

### GES smoothness normalization and archived weights

GES now uses the same coordinate-averaged displacement smoothness as DF-PSM
and DTM:

```text
d_i = q_i(1) - X_i
R_sm = sum_(i,j in E_F) ||d_i - d_j||^2 / (3 * |E_F|)
```

`E_F` contains the three edge occurrences of every triangle, including repeated
interior edges. There is no area, edge-length, or additional coordinate-scale
normalization. GES at revision `f348efdb369e1d341c183933f874c09da8966a70`
instead summed the three coordinate squares before averaging over edges,
giving `R_sm_old = 3 * R_sm`.

To reproduce an archived objective, use `lambda_smooth_new = 3 * lambda_smooth_old`.
This preserves the weighted loss and its gradient, up to floating-point rounding;
it does not imply that an unweighted smoothness log entry is unchanged.

| GES configuration | Archived weight | Coordinate-mean weight |
|---|---:|---:|
| Cortical comparison | 0.01 | 0.03 |
| Hippocampal Stage 1 | 1 | 3 |
| Hippocampal Stage 2 | 30 | 90 |
| CLI default | 25 | 75 |

The CLI default has been migrated. Explicit `--lambda_smooth` / `--smoothness_weight`
values are interpreted in the new convention, with no automatic conversion;
update archived commands and stage-specific overrides before rerunning them.
New `config.json` and `summary.json` files record
`smoothness_normalization: coordinate_mean_over_face_edge_occurrences`.
Their `smooth` history column and `final_smoothness` summary field use the new
unweighted normalization. This change does not rewrite archived runs or rerun
the manuscript experiments. Only the GES smoothness convention changes;
attachment and momentum penalties retain their existing definitions.

## Alpha recovery

```bash
dvsl-recover-alpha --match-dir runs/single --output-dir recovery/example --inner-iters 800 --inner-lr 0.003
```

Recovery accepts a DF-PSM leaf or two-stage root (Stage 2 by default), with exact source and basis assets. It fits beta to the saved induced alpha and advances geometry under regularization. It is an approximate inverse diagnostic; alpha agreement does not uniquely determine beta or recover the fitted surface. `--source`, `--target` and `--basis` can relocate unchanged assets.
