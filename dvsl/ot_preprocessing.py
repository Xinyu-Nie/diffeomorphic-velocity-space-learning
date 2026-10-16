#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OT-guided mesh correspondence with the original dense Sinkhorn calculation.

Writes <target>__ot.npz (T_of_x, gamma_idx, gamma_w, Ns, Nt, eps, stats)
and <target>__ot_T_of_x.vtk with source connectivity. CLI defaults follow
the requested preprocessing recipe. Targets accept a UTF-8 list or one VTK.
"""

import argparse
import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from .ot_inputs import resolve_inputs, validate_parameters, validate_mesh_arrays
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------ I/O helpers ------------------------------

def read_vtk_mesh(path):
    """
    Returns (V, F) where
      V: (N,3) float64
      F: (M,3) int64
    Uses pyvista with triangulation (as in Unsupervised_LocalTransformer_fast_Inference.py).
    """
    # Use pyvista to robustly handle POLYDATA and triangulation
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("pyvista is required to read VTK meshes. Please install it: pip install pyvista")

    try:
        mesh = pv.read(path).triangulate()
        V = np.asarray(mesh.points, dtype=np.float64)
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        if faces.size == 0:
            raise ValueError("No triangle faces found in pyvista mesh after triangulate().")
        F = np.asarray(faces, dtype=np.int64)
        validate_mesh_arrays(V, F, path)
        return V, F
    except Exception as e:
        raise RuntimeError(f"PyVista failed to read VTK mesh '{path}': {e}")

def write_npz(out_path, **arrays):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **arrays)

# ------------------------------ geometry ---------------------------------

def vertex_areas(V, F):
    """
    Simple 1/3 triangle split per incident face (mixed area surrogate).
    V: (N,3), F: (M,3)
    Returns (N,) float64 strictly positive (unless degenerate faces).
    """
    v1 = V[F[:, 0], :]
    v2 = V[F[:, 1], :]
    v3 = V[F[:, 2], :]
    cross_prod = np.cross(v2 - v1, v3 - v1)
    faceA = 0.5 * np.linalg.norm(cross_prod, axis=1)  # (M,)
    N = V.shape[0]
    A = np.zeros(N, dtype=np.float64)
    np.add.at(A, F[:, 0], faceA / 3.0)
    np.add.at(A, F[:, 1], faceA / 3.0)
    np.add.at(A, F[:, 2], faceA / 3.0)
    # Avoid exact zeros
    A[A <= 0.0] = np.finfo(np.float64).eps
    return A

def squared_euclidean_cost(Vs, Vt):
    """
    Returns C[i,j] = ||Vs[i]-Vt[j]||^2
    """
    # Efficient expansion: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    xsq = np.sum(Vs * Vs, axis=1, keepdims=True)  # (Ns,1)
    ysq = np.sum(Vt * Vt, axis=1, keepdims=True).T  # (1,Nt)
    C = xsq + ysq - 2.0 * (Vs @ Vt.T)
    # Numerical guard: small negatives to zero
    np.maximum(C, 0.0, out=C)
    return C

# ------------------------------ sinkhorn ---------------------------------

def sinkhorn(ps, pt, C, eps, max_iter=10000, tol=1e-9, verbose=False, log_every=50):
    """
    Entropic-regularized Sinkhorn on full kernel K = exp(-C/eps).
    ps, pt: (Ns,), (Nt,), both sum to 1
    C: (Ns,Nt) ground cost (>=0)
    eps: entropic regularization (>0)
    Returns u (Ns,), v (Nt,), K (Ns,Nt) where Gamma = diag(u) K diag(v)
    """
    Ns, Nt = C.shape
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not np.isclose(ps.sum(), 1.0, atol=1e-6) or not np.isclose(pt.sum(), 1.0, atol=1e-6):
        raise ValueError("ps and pt must each sum to 1")

    if verbose:
        print(f"Sinkhorn: eps={eps:.3e}, size={Ns}x{Nt}")

    # Kernel
    K = np.exp(-C / eps)  # may underflow if eps << median(C)
    # Avoid exact zeros
    K[K < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny

    u = np.full(Ns, 1.0 / Ns, dtype=np.float64)
    v = np.full(Nt, 1.0 / Nt, dtype=np.float64)

    last_change = np.inf
    for it in range(1, int(max_iter) + 1):
        u_prev = u

        Kv = K @ v  # (Ns,)
        # Guard against zeros
        Kv[Kv <= 0.0] = np.finfo(np.float64).tiny
        u = ps / Kv

        KTu = K.T @ u  # (Nt,)
        KTu[KTu <= 0.0] = np.finfo(np.float64).tiny
        v = pt / KTu

        if it % log_every == 0 or it == 1:
            # Monitor max change in u (simple criterion)
            last_change = np.max(np.abs(u - u_prev))
            if verbose:
                print(f"\riter {it:6d}  max|du|={last_change:.2e}", end="")
            if last_change < tol:
                break

    if verbose:
        print()
    return u, v, K, it, last_change

# ------------------------------ top-K utils ------------------------------

def row_topk_indices_weights(alpha_row, K):
    """
    alpha_row: (Nt,) nonnegative
    Returns (idx,(K,), w,(K,)) where w sums to 1 over the selected indices.
    If Nt < K, fills up to Nt.
    """
    Nt = alpha_row.shape[0]
    k = min(K, Nt)
    # argpartition for top-k (largest)
    idx_part = np.argpartition(alpha_row, -k)[-k:]
    # sort descending
    idx_sorted = idx_part[np.argsort(alpha_row[idx_part])[::-1]]
    weights = alpha_row[idx_sorted]
    s = weights.sum()
    if s <= 0.0:
        # fallback: uniform over k indices
        weights = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        weights = weights / s
    return idx_sorted.astype(np.int64), weights.astype(np.float32)

# ------------------------------ pipeline ---------------------------------

def compute_ot_outputs(Vs, Fs, Vt, Ft, eps, max_iter, tol, topk, verbose, C_precomputed=None):
    """
    Core OT computation and outputs:
      - T_of_x: (Ns,3)
      - gamma_idx: (Ns, topk)
      - gamma_w: (Ns, topk)
    """
    Ns = Vs.shape[0]
    # masses
    ps = vertex_areas(Vs, Fs); ps = ps / ps.sum()
    pt = vertex_areas(Vt, Ft); pt = pt / pt.sum()

    # cost (allow reuse if already computed)
    C = C_precomputed if C_precomputed is not None else squared_euclidean_cost(Vs, Vt)

    # Sinkhorn
    t0 = time.time()
    u, v, Kmat, iters, last_change = sinkhorn(ps, pt, C, eps, max_iter=max_iter, tol=tol, verbose=verbose)
    elapsed = time.time() - t0

    # T_of_x = diag(u/ps) * (K * diag(v) * Vt)
    # Compute B = diag(v) * Vt  == v[:,None]*Vt
    B = v[:, None] * Vt  # (Nt,3)
    M = Kmat @ B         # (Ns,3)
    ratio = (u / ps)[:, None]  # (Ns,1)
    T_of_x = ratio * M         # (Ns,3)

    # Row-sparse γ via α_ij = u_i K_ij v_j / ps_i
    # For top-k we need only per-row α scores.
    gamma_idx = np.zeros((Ns, topk), dtype=np.int64)
    gamma_w   = np.zeros((Ns, topk), dtype=np.float32)

    # Precompute s_i = u_i / ps_i to avoid dividing in loop
    s_row = u / ps

    # We’ll iterate rows; for large Ns consider batching
    Nt = Vt.shape[0]
    for i in range(Ns):
        # α_i· = s_row[i] * (K[i,:] * v)
        alpha_row = s_row[i] * (Kmat[i, :] * v)  # (Nt,)
        idx, w = row_topk_indices_weights(alpha_row, topk)
        gamma_idx[i, :len(idx)] = idx
        gamma_w[i, :len(w)] = w
        # If K > Nt, trailing entries remain zeros (idx=0, w=0). That’s fine.

    stats = {
        "solver": "sinkhorn",
        "eps": float(eps),
        "iterations": int(iters),
        "last_du_change": float(last_change),
        "time_sec": float(elapsed),
        "Ns": int(Ns),
        "Nt": int(Vt.shape[0]),
        "topk": int(topk),
    }
    return T_of_x.astype(np.float32), gamma_idx, gamma_w, stats

# ------------------------------ CLI --------------------------------------

def output_paths_for_target(tpath, output_dir):
    base = os.path.splitext(os.path.basename(tpath))[0]
    out_npz = os.path.join(output_dir, f"{base}__ot.npz")
    out_vtk = os.path.join(output_dir, f"{base}__ot_T_of_x.vtk")
    return base, out_npz, out_vtk


def _process_single_target(tpath, source_path, output_dir, eps_opt, max_iter, tol, topk, verbose, skip_existing):
    # Compose output paths before expensive mesh loading / OT computation.
    base, out_path, out_vtk = output_paths_for_target(tpath, output_dir)
    if skip_existing and os.path.exists(out_path):
        if verbose:
            print(f"Skipping existing: {out_path}")
        return base

    # Read meshes
    if verbose:
        print(f"\nReading target: {tpath}")
    Vt, Ft = read_vtk_mesh(tpath)
    Vs, Fs = read_vtk_mesh(source_path)

    Nt = Vt.shape[0]
    if verbose:
        print(f"  Target verts: {Nt}, faces: {Ft.shape[0]}")

    # Build cost matrix and eps
    C = squared_euclidean_cost(Vs, Vt)
    eps_eff = eps_opt if eps_opt is not None else 0.01 * float(np.median(C))

    # Compute OT outputs (reuse C)
    T_of_x, gamma_idx, gamma_w, stats = compute_ot_outputs(
        Vs, Fs, Vt, Ft, eps=eps_eff,
        max_iter=max_iter,
        tol=tol,
        topk=topk,
        verbose=verbose,
        C_precomputed=C
    )

    # Save NPZ
    write_npz(out_path,
              T_of_x=T_of_x,
              gamma_idx=gamma_idx,
              gamma_w=gamma_w,
              Ns=np.int64(Vs.shape[0]),
              Nt=np.int64(Nt),
              eps=np.float64(eps_eff),
              stats=np.array([str(stats)], dtype=object))

    # Also write VTK for visualization
    try:
        import pyvista as pv
        src_mesh = pv.read(source_path).triangulate()
        src_mesh.points = T_of_x.astype(np.float64)
        src_mesh.save(out_vtk)
    except Exception as e:
        print(f"Warning: failed to write VTK for target '{tpath}': {e}", file=sys.stderr)

    if verbose:
        mean_disp = np.mean(np.linalg.norm(T_of_x - Vs, axis=1))
        print(f"Saved: {out_path}")
        print(f"VTK:   {out_vtk}")
        print(f"  mean |T(x)-x| = {mean_disp:.6f} (units of input coords)")

    return base

def build_parser():
    ap = argparse.ArgumentParser(description="Mesh OT correspondence (Sinkhorn, per-target .npz)")
    ap.add_argument("--source", required=True, help="Path to source .vtk")
    ap.add_argument("--targets", required=True, help="One .vtk file, or a UTF-8 text list of mesh paths (one per nonempty line; relative to working directory).")
    ap.add_argument("--output-dir", required=True, help="Directory to write per-target .npz outputs")
    epsilon = ap.add_mutually_exclusive_group()
    epsilon.add_argument("--eps", type=float, default=1e-2, help="Entropic regularization (default: 0.01).")
    epsilon.add_argument("--auto-eps", dest="eps", action="store_const", const=None,
                         help="Use the legacy 0.01 * median(cost) epsilon for each target.")
    ap.add_argument("--max-iter", type=int, default=1000, help="Max Sinkhorn iterations (default: 1000).")
    ap.add_argument("--tol", type=float, default=1e-9, help="Convergence tolerance on u change.")
    ap.add_argument("--topk", type=int, default=32, help="Top-K target indices per source row.")
    verbosity = ap.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", dest="verbose", action="store_true", help="Print progress (default).")
    verbosity.add_argument("--quiet", "--no-verbose", dest="verbose", action="store_false", help="Disable progress output.")
    ap.add_argument("--workers", type=int, default=1, help="Target worker processes; 1 runs directly without a pool (default: 1).")
    outputs = ap.add_mutually_exclusive_group()
    outputs.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                         help="Skip targets whose output NPZ already exists (default).")
    outputs.add_argument("--overwrite", "--no-skip-existing", dest="skip_existing", action="store_false",
                         help="Recompute and overwrite existing outputs.")
    ap.set_defaults(verbose=True, skip_existing=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        validate_parameters(args)
        source, target_paths, output = resolve_inputs(args.source, args.targets, args.output_dir)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    args.source = str(source)
    args.output_dir = str(output)

    # Read source
    if args.verbose:
        print(f"Reading source: {args.source}")
    Vs, Fs = read_vtk_mesh(args.source)
    Ns = Vs.shape[0]
    if args.verbose:
        print(f"  Source verts: {Ns}, faces: {Fs.shape[0]}")

    # Prepare source mesh only for sequential writing
    src_mesh = None
    if args.workers <= 1:
        try:
            import pyvista as pv
            src_mesh = pv.read(args.source).triangulate()
        except Exception as e:
            raise RuntimeError(f"Failed to prepare source VTK with pyvista: {e}")

    # target_paths were resolved and validated before output creation.
    os.makedirs(args.output_dir, exist_ok=True)

    if args.workers <= 1:
        for tpath in target_paths:
            base, out_path, out_vtk = output_paths_for_target(tpath, args.output_dir)
            if args.skip_existing and os.path.exists(out_path):
                if args.verbose:
                    print(f"Skipping existing: {out_path}")
                continue

            if args.verbose:
                print(f"\nReading target: {tpath}")
            Vt, Ft = read_vtk_mesh(tpath)
            Nt = Vt.shape[0]
            if args.verbose:
                print(f"  Target verts: {Nt}, faces: {Ft.shape[0]}")

            # Build cost matrix (to get default eps if needed)
            C = squared_euclidean_cost(Vs, Vt)
            eps = args.eps if args.eps is not None else 0.01 * float(np.median(C))

            # Compute OT outputs (reuse C)
            T_of_x, gamma_idx, gamma_w, stats = compute_ot_outputs(
                Vs, Fs, Vt, Ft, eps=eps,
                max_iter=args.max_iter,
                tol=args.tol,
                topk=args.topk,
                verbose=args.verbose,
                C_precomputed=C
            )

            # Compose out path from target basename
            write_npz(out_path,
                      T_of_x=T_of_x,
                      gamma_idx=gamma_idx,
                      gamma_w=gamma_w,
                      Ns=np.int64(Ns),
                      Nt=np.int64(Nt),
                      eps=np.float64(eps),
                      stats=np.array([str(stats)], dtype=object))

            # Also write (T_of_x, faces) as VTK to visualize OT map result
            try:
                src_mesh.points = T_of_x.astype(np.float64)
                src_mesh.save(out_vtk)
            except Exception as e:
                print(f"Warning: failed to write VTK for target '{tpath}': {e}", file=sys.stderr)
            if args.verbose:
                mean_disp = np.mean(np.linalg.norm(T_of_x - Vs, axis=1))
                print(f"Saved: {out_path}")
                print(f"VTK:   {out_vtk}")
                print(f"  mean |T(x)-x| = {mean_disp:.6f} (units of input coords)")
    else:
        # Parallel execution across targets
        if args.verbose:
            print(f"Running in parallel with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    _process_single_target,
                    tpath,
                    args.source,
                    args.output_dir,
                    args.eps,
                    args.max_iter,
                    args.tol,
                    args.topk,
                    args.verbose,
                    args.skip_existing,
                )
                for tpath in target_paths
            ]
            failures = []
            for fut in as_completed(futures):
                try:
                    _ = fut.result()
                except Exception as e:
                    print(f"Error in worker: {e}", file=sys.stderr)
                    failures.append(str(e))
        if failures:
            raise RuntimeError(f"{len(failures)} OT target(s) failed. First error: {failures[0]}")

def cli():
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
