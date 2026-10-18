# Numerical conventions and limitations

- Mercer construction is an unweighted source-vertex Gram eigendecomposition: `K U ~= U diag(lam)`, `U.T U = I`. It does not introduce area quadrature or RKHS-unit eigenfunctions. Energy coordinates are `sqrt(lam) * beta`.
- Near-degenerate eigenspaces can rotate and a rank cutoff can split a spectral band. Close eigenvalues/operators do not imply identical beta coordinates. Retain the exact fitted basis with coefficients and models, or explicitly transport coordinates.
- Pointwise attachment is mean-squared coordinate residual over vertices and xyz. Vertex Chamfer is the sum of two mean-squared nearest-vertex distances. Area-sampled surface Chamfer is half the sum of mean squared nearest-sample distances.
- Varifold uses face areas, a compact-support positional kernel and squared normal dot products; areas are independently normalized by default. `tiled` and `stable` backends preserve their separate numerical reductions.
- Source, target and kernel radii must use consistent normalized units. 
- Positive sampled ambient Jacobian determinants, spherical topology, and watertightness do not guarantee an intersection-free discretized mesh or a continuous-time certificate. Inspect geometry as well as Jacobians.
- Alpha is auxiliary projected pullback data. Recovery is approximate, and beta is not uniquely determined by alpha. Generation uses directly optimized beta, not recovered alpha paths.
- Full refinement keeps feasible nonregressing candidates under explicit physical budgets. It does not repair meshes, reject subjects, or guarantee anatomical validity.


