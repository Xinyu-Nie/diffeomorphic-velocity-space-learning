"""Correspondence-based registration and independent normalization of triangle surfaces."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_vtk(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input does not exist: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if reader.GetErrorCode() or mesh.GetPoints() is None:
        raise ValueError(f"Cannot read legacy VTK POLYDATA: {path}")
    vertices = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    # Legacy POINTS float is read as float32 by VTK, whereas MATLAB fscanf
    # retains the decimal ASCII values as doubles. Preserve those values.
    with path.open('rb') as stream:
        header = [stream.readline() for _ in range(4)]
        if header[2].strip().upper() == b'ASCII':
            for line in stream:
                if line.strip().upper().startswith(b'POINTS '):
                    count = int(line.split()[1])
                    values = []
                    while len(values) < count * 3:
                        line = stream.readline()
                        if not line:
                            raise ValueError(f"Incomplete POINTS block: {path}")
                        values.extend(line.split())
                    vertices = np.asarray(values, dtype=np.float64).reshape(count, 3)
                    break
    polygons = mesh.GetPolys()
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    if not np.all(np.diff(offsets) == 3) or len(offsets) <= 1:
        raise ValueError(f"Expected a nonempty triangle mesh: {path}")
    faces = vtk_to_numpy(polygons.GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Invalid coordinates or face indices: {path}")
    return vertices, faces


def write_vtk(path, vertices, faces, precision=6):
    with Path(path).open('w', newline='\n') as stream:
        stream.write('# vtk DataFile Version 3.0\nLDDMM warped surface\nASCII\nDATASET POLYDATA\n')
        stream.write(f'POINTS {len(vertices)} float\n')
        np.savetxt(stream, vertices, fmt=f'%.{precision}g')
        stream.write(f'POLYGONS {len(faces)} {len(faces) * 4}\n')
        np.savetxt(stream, np.column_stack((np.full(len(faces), 3), faces)), fmt='%d')


def area_center(vertices, faces):
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    if areas.sum() <= 0:
        raise ValueError('Mesh has zero total triangle area')
    return vertices - areas @ triangles.mean(axis=1) / areas.sum()


def normalize_independently(vertices, faces):
    centered = area_center(vertices, faces)
    extent = np.ptp(centered, axis=0).max()
    if extent <= 0:
        raise ValueError('Mesh has zero spatial extent')
    return centered * (2.0 / extent)


def register_corresponding(source, target, faces):
    """Similarity Procrustes with scaling enabled and reflection disabled."""
    if source.shape != target.shape:
        raise ValueError('Corresponding preprocessing requires trusted vertex ordering and equal vertex counts')
    target = area_center(target, faces)
    radius = np.sqrt(np.mean(np.sum(target * target, axis=1)))
    if radius <= 0:
        raise ValueError('Target has zero RMS radius')
    target = target / radius
    sx, ty = source.mean(axis=0), target.mean(axis=0)
    x, y = source - sx, target - ty
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if min(nx, ny) <= 0:
        raise ValueError('Degenerate Procrustes input')
    u, singular, vt = np.linalg.svd((x / nx).T @ (y / ny))
    v = vt.T
    if np.linalg.det(v @ u.T) < 0:
        v[:, -1] *= -1
        singular[-1] *= -1
    return nx * singular.sum() * ((y / ny) @ (v @ u.T)) + sx


def mesh_smooth(vertices, faces, iterations=20, strength=0.2):
    """Uniform Laplacian smoothing; boundary vertices remain fixed."""
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = np.unique(undirected[counts == 1])
    rows = np.concatenate((undirected[:, 0], undirected[:, 1]))
    cols = np.concatenate((undirected[:, 1], undirected[:, 0]))
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vertices), len(vertices)))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    active = degree > 0
    active[boundary] = False
    result = vertices.copy()
    for _ in range(iterations):
        means = (adjacency @ result)[active] / degree[active, None]
        result[active] += strength * (means - result[active])
    return result


def matching_topology(source_faces, target_faces):
    # Face order and winding may differ; vertex indices must still correspond.
    def canonical(faces):
        f = np.sort(faces, axis=1)
        return f[np.lexsort((f[:, 2], f[:, 1], f[:, 0]))]
    return source_faces.shape == target_faces.shape and np.array_equal(canonical(source_faces), canonical(target_faces))


def quantize(vertices, precision):
    return np.asarray([[float(f'{x:.{precision}g}') for x in row] for row in vertices])


def canonical_mode(value):
    return {'cortical': 'corresponding', 'hippocampus': 'noncorresponding'}.get(value, value)


def build_parser(default_mode='auto'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--targets', nargs='+', type=Path)
    group.add_argument('--input-dir', type=Path)
    parser.add_argument('--pattern', default='*.vtk', help='Glob within --input-dir')
    parser.add_argument('--input-suffix', help='Remove this exact filename suffix to obtain the sample ID')
    parser.add_argument('--output-suffix', help='Override the base output suffix, e.g. _registered.vtk')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-output', type=Path, help='Optional normalized source (noncorresponding) or original source (corresponding)')
    parser.add_argument('--mode', type=canonical_mode, choices=['auto', 'corresponding', 'noncorresponding'], default=canonical_mode(default_mode))
    parser.add_argument('--correspondence', choices=['auto', 'yes', 'no'], default='auto',
                        help='Auto checks vertex counts and indexed triangle connectivity; yes asserts trusted ordered vertex correspondence')
    parser.add_argument('--smooth-iterations', type=int, help='Defaults: corresponding 0; noncorresponding 20; 0 disables smoothing')
    parser.add_argument('--smooth-lambda', type=float, default=0.2, help='Laplacian strength in [0,1] (default 0.2)')
    parser.add_argument('--precision', type=int, default=6, help='Significant digits; MATLAB write_vtk uses 6')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(default_mode='auto', argv=None):
    parser = build_parser(default_mode)
    args = parser.parse_args(argv)
    if args.smooth_iterations is not None and args.smooth_iterations < 0:
        parser.error('--smooth-iterations must be nonnegative')
    if not 0 <= args.smooth_lambda <= 1 or not 1 <= args.precision <= 17:
        parser.error('Require smooth-lambda in [0,1] and precision in [1,17]')
    if args.output_suffix and (not args.output_suffix.endswith('.vtk') or '/' in args.output_suffix or '\\' in args.output_suffix):
        parser.error('--output-suffix must end in .vtk and contain no directory separators')
    try:
        if args.mode == 'noncorresponding' and args.correspondence == 'yes':
            raise ValueError('--mode noncorresponding conflicts with --correspondence yes; select the corresponding workflow for trusted paired vertices')
        if args.input_dir is not None and not args.input_dir.is_dir():
            raise ValueError(f'Input directory does not exist: {args.input_dir}')
        source, sf = read_vtk(args.source)
        targets = args.targets if args.targets is not None else sorted(args.input_dir.glob(args.pattern))
        if not targets:
            raise ValueError('No target files found')
        plans, occupied = [], set()
        for target_path in targets:
            target, tf = read_vtk(target_path)
            corresponding = source.shape == target.shape and matching_topology(sf, tf)
            if args.correspondence != 'auto':
                corresponding = args.correspondence == 'yes'
            mode = args.mode if args.mode != 'auto' else ('corresponding' if corresponding else 'noncorresponding')
            if args.mode == 'auto' and args.correspondence == 'auto' and mode == 'corresponding':
                print(f'{target_path.name}: count/connectivity heuristic selected corresponding preprocessing; trusted vertex ordering is still required.')
            if mode == 'corresponding' and (source.shape != target.shape or args.correspondence == 'no'):
                raise ValueError(f'Invalid trusted correspondence: {target_path}')
            if args.input_suffix:
                if not target_path.name.endswith(args.input_suffix):
                    raise ValueError(f'Input does not end with {args.input_suffix}: {target_path}')
                subject = target_path.name[:-len(args.input_suffix)]
            else:
                subject = target_path.stem
            if not subject:
                raise ValueError('Empty sample ID')
            iterations = args.smooth_iterations if args.smooth_iterations is not None else (0 if mode == 'corresponding' else 20)
            base = args.output_dir / (subject + (args.output_suffix or ('_target.vtk' if mode == 'corresponding' else '_normalize.vtk')))
            outputs = [base]
            if iterations and args.smooth_lambda:
                outputs.append(args.output_dir / f'{subject}_sm{args.smooth_lambda:g}.vtk')
            for output in outputs:
                if output.resolve() in occupied:
                    raise ValueError(f'Duplicate output: {output}')
                occupied.add(output.resolve())
            plans.append((target_path, mode, iterations, outputs))
        if args.source_output:
            if len({plan[1] for plan in plans}) != 1:
                raise ValueError('--source-output requires all targets to use the same mode')
            if args.source_output.resolve() in occupied:
                raise ValueError('Source output collides with target output')
            occupied.add(args.source_output.resolve())
        inputs = {args.source.resolve(), *(p.resolve() for p in targets)}
        if occupied & inputs:
            raise ValueError('An output would overwrite an input mesh')
        if not args.overwrite:
            for output in occupied:
                if output.exists():
                    raise ValueError(f'Output exists; use --overwrite: {output}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target_path, mode, iterations, outputs in plans:
            target, tf = read_vtk(target_path)
            result = register_corresponding(source, target, tf) if mode == 'corresponding' else normalize_independently(target, tf)
            write_vtk(outputs[0], result, tf, args.precision)
            if len(outputs) > 1:
                # MATLAB's processing script reads the saved normalized file
                # before smoothing, so preserve that intermediate rounding.
                smoothed = mesh_smooth(quantize(result, args.precision), tf, iterations, args.smooth_lambda)
                write_vtk(outputs[1], smoothed, tf, args.precision)
            print(f'{target_path.name}: {mode}, vertices={len(target)}, smoothing={iterations} x {args.smooth_lambda:g} -> {outputs}')
        if args.source_output:
            result = source if plans[0][1] == 'corresponding' else normalize_independently(source, sf)
            args.source_output.parent.mkdir(parents=True, exist_ok=True)
            write_vtk(args.source_output, result, sf, args.precision)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
