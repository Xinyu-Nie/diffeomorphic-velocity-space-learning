"""Resolve direct DF-PSM beta populations without mixing subjects or stages."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .generation_io import sha
from .surface_io import load_mesh,load_mercer_basis
from .surface_matching import validate_basis_metadata

LOCKED_GEOMETRY=dict(shape=[13,1536,3],radii=[.19],weights=[1.],kernel='c2',n_steps=12,
                     integrator='midpoint',max_neighbors=512)
HIPPOCAMPUS_GEOMETRY=dict(shape=[21,512,3],radii=[1.],weights=[1.],kernel='c2',n_steps=20,
                         integrator='midpoint',max_neighbors=2562)


def resolve_file(value,parent):
    path=Path(value).expanduser()
    candidates=[path] if path.is_absolute() else [Path.cwd()/path,Path(parent)/path]
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    raise ValueError(f'Missing input file: {value}. Supply --source or --basis for relocated geometry assets.')


def discover_runs(path,stage='stage2'):
    """Stop recursion at each workflow/leaf, so its stages cannot be double counted."""
    path=Path(path).expanduser().resolve()
    if not path.is_dir():raise ValueError(f'Matching run/collection directory not found: {path}')
    if (path/'two_stage_config.json').is_file():
        leaf=path/('stage1_ot' if stage=='stage1' else 'stage2_target')
        if not (leaf/'config.json').is_file():raise ValueError(f'Missing selected {stage} export: {leaf}')
        return [(leaf,path.name,stage)]
    if (path/'config.json').is_file():return [(path,path.name,'leaf')]
    result=[]
    for child in sorted(path.iterdir(),key=lambda p:p.name.casefold()):
        if child.is_dir() and not child.name.startswith('.'):
            result.extend(discover_optional(child,stage))
    if not result:raise ValueError(f'No matching exports found in collection: {path}')
    return result


def discover_optional(path,stage):
    if (path/'config.json').is_file() or (path/'two_stage_config.json').is_file():return discover_runs(path,stage)
    result=[]
    for child in sorted(path.iterdir(),key=lambda p:p.name.casefold()):
        if child.is_dir() and not child.name.startswith('.'):result.extend(discover_optional(child,stage))
    return result


def collection(runs,list_path,stage):
    result=[]
    for path in runs or []:result.extend(discover_runs(path,stage))
    if list_path:
        path=Path(list_path).expanduser().resolve()
        if path.suffix.lower()=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
            if not rows or 'match_dir' not in rows[0]:raise ValueError('Collection CSV requires match_dir and optional subject_id columns.')
            for row in rows:
                value=Path(row['match_dir']).expanduser();value=value if value.is_absolute() else path.parent/value
                found=discover_runs(value,stage)
                subject=row.get('subject_id',row.get('subjID','')).strip()
                if subject and len(found)!=1:raise ValueError('A CSV row with a subject_id must resolve exactly one matching run.')
                result.extend([(leaf,subject or default,s) for leaf,default,s in found])
        else:
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip() and not line.lstrip().startswith('#'):
                    value=Path(line.strip()).expanduser();value=value if value.is_absolute() else path.parent/value
                    result.extend(discover_runs(value,stage))
    if not result:raise ValueError('Supply a nonempty matching-run collection for both train and validation.')
    return result


def inspect_populations(args):
    seen_ids=set();seen_paths=set();seen_beta=set();asset_cache={};baseline=None;populations={};manifests={}
    for split,runs,listing in [('train',args.train_runs,args.train_list),('validation',args.validation_runs,args.validation_list)]:
        rows=[];arrays=[]
        for leaf,subject,stage in collection(runs,listing,args.match_stage):
            key=str(leaf).casefold()
            if not subject or subject in seen_ids or key in seen_paths:raise ValueError(f'Duplicate subject or matching path within/between partitions: {subject}: {leaf}')
            seen_ids.add(subject);seen_paths.add(key)
            config_file=leaf/'config.json'
            c=json.loads(config_file.read_text(encoding='utf-8'))
            required=('source_vtk','basis','control_basis_dim_resolved','radii_resolved','weights_resolved','kernel','n_steps','max_neighbors','integrator')
            missing=[name for name in required if name not in c]
            if missing:raise ValueError(f'{leaf}: expected direct DF-PSM exports, not GES/DTM/recovered alpha. Missing: {missing}')
            beta_file=leaf/'beta_knots.npy'
            if not beta_file.is_file():raise ValueError(f'Direct beta_knots.npy is required: {leaf}; sequential recovered beta is not an input.')
            beta=np.load(beta_file,allow_pickle=False)
            if beta.dtype.kind not in 'fiu' or beta.ndim!=3 or beta.shape[-1]!=3 or beta.shape[0]<2 or not np.isfinite(beta).all():
                raise ValueError(f'Invalid direct beta path at {beta_file}: expected finite (T,R,3), T>=2.')
            digest=sha(beta_file)
            if digest in seen_beta:raise ValueError(f'Duplicate beta contents within/between partitions: {beta_file}')
            seen_beta.add(digest)
            R=c['control_basis_dim_resolved']
            if not isinstance(R,int) or isinstance(R,bool) or R<=0 or beta.shape[1]!=R:raise ValueError(f'Control rank does not match beta export: {leaf}')
            if 'time_intervals' in c and c['time_intervals']!=beta.shape[0]-1:raise ValueError(f'Saved alpha/control time grid disagrees with direct beta: {leaf}')
            for name in ('n_steps','max_neighbors'):
                if not isinstance(c[name],int) or isinstance(c[name],bool) or c[name]<=0:raise ValueError(f'{name} must be a positive integer: {leaf}')
            radii,weights=c['radii_resolved'],c['weights_resolved']
            if not isinstance(radii,list) or not isinstance(weights,list) or not radii or len(radii)!=len(weights):raise ValueError('Invalid matched kernel radii/weights.')
            if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in radii+weights):raise ValueError('Kernel radii/weights must be finite and positive.')
            if c['kernel'] not in ('c2','c4') or c['integrator'] not in ('euler','midpoint','rk4'):raise ValueError('Unsupported matched kernel/integrator.')
            source=resolve_file(args.source or c['source_vtk'],leaf)
            basis=resolve_file(args.basis or c['basis'],leaf)
            for asset in (source,basis):
                if asset not in asset_cache:asset_cache[asset]=sha(asset)
            if args.basis:
                try:original_basis=resolve_file(c['basis'],leaf)
                except ValueError:original_basis=None
                if original_basis is not None:
                    if original_basis not in asset_cache:asset_cache[original_basis]=sha(original_basis)
                    if asset_cache[original_basis]!=asset_cache[basis]:
                        raise ValueError('The --basis override differs from the accessible fitted basis; relocation must preserve exact eigenvector coordinates.')
            geometry=dict(shape=list(beta.shape),radii=radii,weights=weights,kernel=c['kernel'],n_steps=c['n_steps'],
                          integrator=c['integrator'],max_neighbors=c['max_neighbors'])
            signature=dict(geometry=geometry,source_sha256=asset_cache[source],basis_sha256=asset_cache[basis])
            if baseline is None:
                baseline=dict(signature=signature,source=str(source),basis=str(basis))
                mesh=load_mesh(str(source),torch.device('cpu'),torch.float32)
                U,lam,meta=load_mercer_basis(str(basis),R,torch.device('cpu'),torch.float32)
                if U.shape!=(len(mesh.points),R):raise ValueError('Source/basis dimensions do not match the fitted control rank.')
                # PCA uses float64 eigenvalues from the exact NPZ, as the locked fitter does.
                with np.load(basis,allow_pickle=False) as data:
                    raw_lam=np.asarray(data['lam'],dtype=np.float64).reshape(-1)
                    order=np.argsort(raw_lam)[::-1][:R];baseline['lam']=raw_lam[order]
                if (baseline['lam']<=1e-12).any():raise ValueError('Selected eigenvalues must be >1e-12 for the locked energy codec.')
                validate_basis_metadata(meta,radii,weights,c['kernel'],False)
                baseline['source_points']=mesh.points.numpy()
            elif signature!=baseline['signature']:
                raise ValueError(f'All training/validation runs must use identical template, exact basis, control grid and decoder settings: {leaf}')
            qfile=leaf/'trajectory_q.npy'
            if not qfile.exists():qfile=leaf/'trajectory_q_knots.npy'
            if qfile.exists():
                q=np.load(qfile,mmap_mode='r',allow_pickle=False)
                if not len(q) or not np.array_equal(np.asarray(q[0],dtype=np.float32),baseline['source_points']):
                    raise ValueError(f'Template geometry/order disagrees with saved initial matching frame: {leaf}')
            arrays.append(beta.astype(np.float32))
            rows.append(dict(subject_id=subject,match_dir=str(leaf),selected_stage=stage,beta_path=str(beta_file),
                             beta_sha256=digest,matching_config_sha256=sha(config_file),matching_dtype=c.get('dtype'),
                             metric=c.get('loss',c.get('metric'))))
        populations[split]=np.stack(arrays);manifests[split]=rows
    geometry=baseline['signature']['geometry']
    if args.decoder_profile=='hippocampus' and geometry!=HIPPOCAMPUS_GEOMETRY:
        raise ValueError(f'Exports differ from the hippocampal decoder: {geometry}')
    if args.decoder_profile=='locked' and geometry!=LOCKED_GEOMETRY:
        raise ValueError(f'Exports differ from locked cortex decoder {LOCKED_GEOMETRY}; got {geometry}. Use --decoder-profile match explicitly to retain these exports\' actual geometry instead of silently reinterpreting beta.')
    bound=min(len(populations['train'])-1,int(np.prod(geometry['shape'])))
    if args.pca_latent_dim>bound:
        raise ValueError(f'Requested PCA dimension {args.pca_latent_dim} exceeds centered rank bound {bound}; training count={len(populations["train"])}. No dimension was lowered. Increase independent training count or explicitly lower --pca_latent_dim.')
    return populations,manifests,baseline
