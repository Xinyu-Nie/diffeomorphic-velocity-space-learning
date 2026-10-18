# CalVel-Diff training and generation

CalVel-Diff learns directly optimized DF-PSM beta paths. It fits training-only energy-coordinate temporal/PCA transforms, one global latent scale, an EMA MLP with cosine diffusion/v-prediction, and independent frozen affine variance calibration. Sampling applies that calibration once, decodes beta, and optionally refines in full coefficient space. Alpha recovery is not part of this generative workflow.

Training and sampling use separate output directories. Both commands validate a plan without writing unless `--execute` is supplied. The same trained final-epoch EMA can generate multiple independently configured populations. `--resume` requires unchanged settings, input hashes, and relevant code hashes; changing a training budget requires a new directory.

## Input population

Use `--train-runs` / `--validation-runs` for matching collections, or CSV lists:

```csv
subject_id,match_dir
subject_001,../matching/subject_001
subject_002,../matching/subject_002
```

CSV paths resolve relative to the list file; row order is retained. Each run needs `beta_knots.npy` and its DF-PSM `config.json`. Two-stage roots select Stage 2 by default. Subjects must share exact source and basis contents, beta shape, kernel, and integration geometry. Duplicate IDs, paths and beta contents are rejected within/between splits. Standalone arrays, recovered beta, and GES/DTM exports are not accepted as direct DF-PSM populations.

## Cortex

The default locked geometry is beta `[13,1536,3]`, C2 radius 0.19, 12 midpoint steps, and neighbor limit 512. All temporal components are retained; PCA uses the SVD solver. q=2,048 requires a sufficiently large, numerically independent training population.

```bash
dvsl-generate train --train-list train.csv --validation-list validation.csv --decoder-profile locked --pca_latent_dim 2048 --mlpv_hidden_dim 2048 --mlpv_epochs 20000 --output-dir models/cortex --execute
dvsl-generate sample --model-dir models/cortex --output-dir generated/cortex --n_samples 250 --sample_seed 20261231 --arms full_beta --execute
```

## Hippocampus

The explicit profile requires beta `[21,512,3]`, C2 radius 1, 20 midpoint steps, and neighbor limit 2,562. It uses M=8 temporal components and q=640 float64 Gram PCA, with the archived scoring/sign convention. These options must be specified together; they are not silently inferred.

```bash
dvsl-generate train --train-list train.csv --validation-list validation.csv --decoder-profile hippocampus --temporal_dim 8 --pca_solver gram --pca_latent_dim 640 --mlpv_hidden_dim 2048 --mlpv_epochs 20000 --output-dir models/hippocampus --execute
dvsl-generate sample --model-dir models/hippocampus --output-dir generated/hippocampus --n_samples 86 --sample_seed 20260930 --arms full_beta --refinement_step 0.0005 --reference_dimension 640 --reference_scale 0.39562398195266724 --p_scale 0.03588920887391313 --loss_weights 1 0.1 0.05 0.001 0.001 --execute
```

The sample command records the original train-derived physical reference units. These constants describe the reproduced cohort and should not be assumed suitable for a different population. The hippocampal profile restores late knots 13/17/20, bands `[0,171)`, `[171,342)`, `[342,512)` and temporal risk sigma 2.5.
## Other matching geometry

`--decoder-profile match` explicitly adopts shared geometry from actual matching exports. Use q no larger than `min(training_count-1, 3*M*R)` and the measured numerical rank. `--temporal_dim 0` retains all temporal components. Small CPU checks can use `--device cpu`, smaller model/PCA settings, and shorter schedules.


