# Surface preparation and matching

Use legacy VTK POLYDATA triangle meshes. Preprocessing preserves vertex order and connectivity and performs no remeshing or topology repair. Source/template geometry must remain consistent across basis construction, OT and fitting.

## Preprocessing

```bash
dvsl-preprocess-corresponding --source template.vtk --targets subject.vtk --output-dir prepared/corresponding
dvsl-preprocess-noncorresponding --source template.vtk --targets subject.vtk --output-dir prepared/independent --smooth-iterations 20 --smooth-lambda 0.2
```

Corresponding mode requires trusted ordered vertices. It area-centers and RMS-normalizes the target, then fits a proper similarity transform to the supplied source; no reflection, no additional smoothing by default. Independent mode area-centers each target and applies `2/largest_axis_extent`, then defaults to 20 uniform Laplacian iterations at 0.2. Six-significant-digit ASCII export is retained; smoothing uses that rounded intermediate, without post-smoothing renormalization. Topology/count equality in auto mode is a heuristic, not proof of correspondence.

## Mercer basis

```bash
dvsl-basis --src template.vtk --radii 0.19 --kernel c2 --k_eig 4096 --max_neighbors 512 --out basis.npz
```

CUDA and matching `torch-cluster` are required. The example rank must fit your source vertex count. Rank is not silently reduced. Increase the neighbor cap when required; truncating kernel neighborhoods changes the operator. Multiple radii and positive normalized mixture weights are supported. Current DTM consumes one radius only; GES does not use a Mercer basis. Keep the exact basis used by existing coefficients and models.

## OT initialization

```bash
dvsl-ot --source template.vtk --targets subject.vtk --output-dir ot --eps 0.01 --max-iter 2000 --topk 32
```

This produces `<target_stem>__ot_T_of_x.vtk` and corresponding NPZ arrays. `--targets` accepts one VTK file or a text list. The archived hippocampal tests used 2,000 iterations; CLI default is 1,000. The dense area-weighted Sinkhorn computation runs on CPU. The full-plan barycenter and renormalized top-k sparse transport weights are distinct exports. The absolute scaling-vector stopping criterion can remain unsatisfied despite small transport marginal errors.

## Matching

```bash
dvsl-match-df-psm --pairs template.vtk subject.vtk --basis basis.npz --radii 0.19 --metric pointwise --output_dir runs/single
dvsl-match-df-psm --pairs template.vtk subject.vtk --basis basis.npz --radii 1.0 --metric varifold --varifold_radius 0.16 --varifold_backend stable --two-stage --ot-dir ot --stage1-adam-iters 500 --stage2-adam-iters 2000 --output_dir runs/two_stage
```

Examples show workflow syntax; specify ranks, knots, integration settings and all regularization weights for your experiment. Paper budgets are not generic CLI defaults. `--two-stage` fits the OT corresponding surface pointwise, then fits the real target, using the original source in both stages and a verified optimized-state handoff.

DF-PSM attachments: `pointwise`, `chamfer`, `surface_chamfer`, `varifold`. GES/DTM expose pointwise, vertex Chamfer and varifold. `--metric` accepts the legacy `--loss` alias. All three have `--help` and two-stage support. Matching exports include controls, trajectories, configuration, history, summary and deformed VTK. Use fresh output directories.

## Alpha recovery

```bash
dvsl-recover-alpha --match-dir runs/single --output-dir recovery/example --inner-iters 800 --inner-lr 0.003
```

Recovery accepts a DF-PSM leaf or two-stage root (Stage 2 by default), with exact source and basis assets. It fits beta to the saved induced alpha and advances geometry under regularization. It is an approximate inverse diagnostic; alpha agreement does not uniquely determine beta or recover the fitted surface. `--source`, `--target` and `--basis` can relocate unchanged assets.
