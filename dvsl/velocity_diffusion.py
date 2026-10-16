"""Train and sample Calibrated Geometry-Refined Velocity Diffusion (CalVel-Diff).

Training fits direct Eulerian beta paths using training-only energy PCA, global
scaling, EMA v-prediction, and independent frozen variance calibration. Sampling
loads those assets, decodes beta, applies the retained full-space refinement,
and integrates the same fitted Eulerian deformation geometry.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
import pyvista as pv
from .generation_io import sha,write_json,write_csv,immutable_json,protected_output,exclusive_run,digest_files
from .generation_inputs import inspect_populations,LOCKED_GEOMETRY,resolve_file
from .velocity_diffusion_network import VelocityPredictionMLP,sample_diffusion
from .velocity_diffusion_codec import fit_codec,encode_population,FrozenBetaCodec
from .velocity_diffusion_refinement import RefinementConfig,RiskMiner,integrate,full_correct
from .surface_io import load_mesh,load_mercer_basis
from .mercer_validation import legacy_precision

ARMS={'mlp':'beta-MLP-v','calibrated_mlp':'beta-MLP-v + VC','full_beta':'CalVel-Diff'}
FORMAT='dvsl_calvel_diff_v1'


def environment_info(device):
    import platform
    import scipy
    import threadpoolctl
    return dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
                pyvista=pv.__version__,scipy=scipy.__version__,threadpoolctl=threadpoolctl.__version__,
                cuda_runtime=torch.version.cuda,device=device,
                gpu=torch.cuda.get_device_name() if device=='cuda' and torch.cuda.is_available() else None)


def option(parser,name,**kwargs):
    aliases=['--'+name]
    if '_' in name:aliases.append('--'+name.replace('_','-'))
    parser.add_argument(*aliases,**kwargs)


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    modes=parser.add_subparsers(dest='mode',required=True)
    train=modes.add_parser('train',help='Fit PCA, EMA MLP-v, and frozen variance calibration.')
    sample=modes.add_parser('sample',help='Load frozen assets, sample beta, and generate surfaces.')
    for mode in (train,sample):
        mode.add_argument('--output-dir','--root',dest='output_dir',required=True,type=Path)
        mode.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='CPU is an explicit small-test option; the locked experiment used CUDA.')
        mode.add_argument('--source',type=Path,help='Explicit relocated template, with unchanged content.')
        mode.add_argument('--basis',type=Path,help='Explicit relocated exact fitted Mercer basis.')
        mode.add_argument('--execute',action='store_true',help='Execute this mode; default prints a validated plan without writing files.')
        mode.add_argument('--resume',action='store_true',help='Resume only an unchanged configuration and verified artifacts.')
    train.add_argument('--train-runs',nargs='+',type=Path)
    train.add_argument('--validation-runs',nargs='+',type=Path)
    train.add_argument('--train-list',type=Path,help='Text list or CSV with match_dir and optional subject_id.')
    train.add_argument('--validation-list',type=Path)
    train.add_argument('--match-stage',choices=['stage1','stage2'],default='stage2',help='Select one stage per two-stage root; explicit leaf paths are used as supplied.')
    train.add_argument('--decoder-profile',choices=['locked','match','hippocampus'],default='locked',help='locked requires cortex geometry; hippocampus requires the original 21-knot 512-mode decoder.')
    option(train,'temporal_dim',type=int,default=0,help='0 retains all saved temporal components; hippocampal protocol uses 8.')
    option(train,'pca_solver',choices=['svd','gram'],default='svd')
    for name,default in [('pca_latent_dim',2048),('mlpv_hidden_dim',2048),('mlpv_epochs',20000),('training_seed',1234),
                         ('calibration_count',4096),('calibration_seed',20261211),('calibration_block',256),
                         ('batch_size',128),('n_blocks',6),('time_emb_dim',128),('diffusion_steps',1000),
                         ('pca_threads',8),('save_every',1000),('val_every',100)]:option(train,name,type=int,default=default)
    for name,default in [('lr',1e-4),('weight_decay',1e-5),('grad_clip',1.),('ema',.999),('dropout',0.),('calibration_clip',6.)]:
        option(train,name,type=float,default=default)
    sample.add_argument('--model-dir',required=True,type=Path,help='Completed train-mode bundle; no refitting takes place.')
    for name,default in [('n_samples',250),('sample_seed',20261231),('sample_block',256),('refinement_updates',16),
                         ('contact_refresh',4),('reference_dimension',1536),('max_pairs',1024)]:option(sample,name,type=int,default=default)
    for name,default in [('x0_clip',6.),('refinement_step',.002),('reference_scale',.2991466820240021),
                         ('det_tolerance',1e-5),('rms_cap',.05),('clearance_margin',.02),('det_risk_threshold',.15),
                         ('p_scale',.012632200545880998),('nonregression_weight',1.)]:option(sample,name,type=float,default=default)
    option(sample,'band_caps',type=float,nargs=3,default=[.02,.05,.08])
    option(sample,'loss_weights',type=float,nargs=5,default=[.1,.1,.05,.001,.001])
    sample.add_argument('--arms',nargs='+',choices=list(ARMS),default=['full_beta'])
    option(sample,'write_vtk',action=argparse.BooleanOptionalAction,default=True)
    option(sample,'save_q_knots',action=argparse.BooleanOptionalAction,default=False)
    option(sample,'exact_si',action=argparse.BooleanOptionalAction,default=False)
    return parser


def validate_arguments(a):
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    positives=('pca_latent_dim','mlpv_hidden_dim','mlpv_epochs','calibration_count','calibration_block','batch_size',
               'n_blocks','time_emb_dim','diffusion_steps','pca_threads','save_every','val_every') if a.mode=='train' else (
               'n_samples','sample_block','contact_refresh','reference_dimension','max_pairs')
    for name in positives:
        if getattr(a,name)<1:raise ValueError(f'--{name} must be positive.')
    if a.mode=='train':
        if a.temporal_dim<0:raise ValueError('Temporal dimension must be nonnegative.')
        if a.decoder_profile=='hippocampus' and (a.temporal_dim,a.pca_latent_dim,a.pca_solver)!=(8,640,'gram'):
            raise ValueError('The hippocampal protocol requires temporal_dim=8, pca_latent_dim=640, pca_solver=gram.')
        if not 1<=a.pca_latent_dim<=2048:raise ValueError('--pca_latent_dim must lie in 1..2048; the retained validation noise uses a fixed 2048-coordinate prefix.')
        if a.calibration_count<2 or a.time_emb_dim<2 or a.diffusion_steps<2:raise ValueError('Calibration count, time embedding and diffusion steps must be at least two.')
        if not 0<=a.training_seed<2**32:raise ValueError('--training_seed must lie in 0..2^32-1.')
        if not math.isfinite(a.ema) or not 0<=a.ema<1 or not math.isfinite(a.dropout) or not 0<=a.dropout<1:raise ValueError('EMA and dropout must lie in [0,1).')
        positive_float=('lr','grad_clip');nonnegative=('weight_decay','calibration_clip')
    else:
        if a.refinement_updates<0:raise ValueError('--refinement_updates must be nonnegative.')
        if len(set(a.arms))!=len(a.arms):raise ValueError('Duplicate sampling arms.')
        positive_float=('refinement_step','reference_scale','rms_cap','clearance_margin','det_risk_threshold','p_scale')
        nonnegative=('x0_clip','det_tolerance','nonregression_weight')
        for value in a.band_caps+a.loss_weights:
            if not math.isfinite(value) or value<0:raise ValueError('Band caps and loss weights must be finite and nonnegative.')
    for name in positive_float:
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<=0:raise ValueError(f'--{name} must be finite and positive.')
    for name in nonnegative:
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<0:raise ValueError(f'--{name} must be finite and nonnegative.')
    seed=a.calibration_seed if a.mode=='train' else a.sample_seed
    if not 0<=seed<2**63:raise ValueError('Diffusion draw seeds must lie in 0..2^63-1.')
    if a.execute and a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; install/use the validated environment or explicitly select --device cpu for small checks.')


def settings(a):
    return json.loads(json.dumps({k:v for k,v in vars(a).items() if k not in ('execute','resume')},default=str))


def code_hashes():
    names=['velocity_diffusion.py','generation_inputs.py','generation_io.py','velocity_diffusion_network.py',
           'velocity_diffusion_codec.py','velocity_diffusion_refinement.py','velocity_diffusion_training.py',
           'surface_matching.py','kernels.py','surface_io.py']
    return {name:sha(Path(__file__).with_name(name)) for name in names}


def run_stage(root,name,fn,paths):
    marker=root/'.pipeline'/(name+'.done.json')
    if marker.exists():
        saved=json.loads(marker.read_text())
        if digest_files(root,paths)!=saved['outputs']:raise ValueError(f'Completed {name} output changed; restore it or use a new output directory.')
        print(f'[{name}] verified; skipping',flush=True);return
    print(f'[{name}] running',flush=True);fn()
    write_json(marker,dict(outputs=digest_files(root,paths)))


def load_network(root,bundle,device):
    ck=torch.load(root/'02_mlpv/final_ema.pt',map_location='cpu',weights_only=False)
    c=ck['config'];a=bundle['training_settings']
    expected=dict(latent_dim=a['pca_latent_dim'],hidden_dim=a['mlpv_hidden_dim'],n_blocks=a['n_blocks'],
                  time_emb_dim=a['time_emb_dim'],dropout=a['dropout'],T=a['diffusion_steps'],param='v',std_mode='global',
                  batch_size=a['batch_size'],lr=a['lr'],weight_decay=a['weight_decay'],grad_clip=a['grad_clip'],
                  ema_decay=a['ema'],seed=a['training_seed'],lr_schedule='cosine',loss_weight='none')
    for name,value in expected.items():
        if c.get(name)!=value:raise ValueError(f'Trained checkpoint {name} disagrees with bundle: {c.get(name)} vs {value}')
    if ck['epoch']!=a['mlpv_epochs']:raise ValueError('Sampling requires the final-epoch EMA checkpoint.')
    q=c['latent_dim'];mean=np.asarray(ck['y_mean']);std=np.asarray(ck['y_std'])
    if mean.shape!=(1,q) or std.shape!=(1,q) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid fitted latent normalization.')
    if not np.allclose(std,std.flat[0],rtol=1e-6,atol=0):raise ValueError('Expected one global training scale, not per-component whitening.')
    for name in ('betas','alphas','alpha_bar'):
        arr=np.asarray(ck[name])
        if arr.shape!=(c['T'],) or not np.isfinite(arr).all() or (arr<=0).any() or (arr>1).any():raise ValueError(f'Invalid saved diffusion schedule: {name}')
    net=VelocityPredictionMLP(q,c['hidden_dim'],c['time_emb_dim'],c['n_blocks'],c['dropout'])
    net.load_state_dict(ck['ema_state'],strict=True)
    return ck,net.to(device).eval().requires_grad_(False)


def draw(net,ck,count,seed,block,clip):
    return torch.cat([sample_diffusion(net,ck,min(block,count-i),seed+104729*(i//block),'ddpm',clip=clip).cpu()
                      for i in range(0,count,block)])


def tensor_context(data,device):
    def convert(value,dtype):
        if isinstance(value,list):return [convert(item,dtype) for item in value]
        return torch.as_tensor(np.array(value),device=device,dtype=dtype)
    return {key:convert(value,torch.long if key=='pairs' else torch.float32) for key,value in data.items()}


def fit_calibration(root,a):
    provisional=dict(training_settings=settings(a))
    ck,net=load_network(root,provisional,a.device)
    folder=root/'03_calibration';folder.mkdir(exist_ok=True)
    z=draw(net,ck,a.calibration_count,a.calibration_seed,a.calibration_block,a.calibration_clip).numpy()
    generated=(z*np.asarray(ck['y_std'])+np.asarray(ck['y_mean'])).astype(np.float64)
    train=np.load(root/'01_pca/train_y.npy').astype(np.float64)
    scale=train.std(0)/(generated.std(0)+1e-9);shift=train.mean(0)-scale*generated.mean(0)
    if not np.isfinite(scale).all() or not (scale>0).all() or not np.isfinite(shift).all():raise ValueError('Invalid calibration scale/shift; check latent rank and sampled variance.')
    stored_scale=scale.astype(np.float32);stored_shift=shift.astype(np.float32)
    if not np.isfinite(stored_scale).all() or not (stored_scale>0).all() or not np.isfinite(stored_shift).all():
        raise ValueError('Calibration cannot be represented by finite positive float32 scale/shift; no values were clipped.')
    np.savez(folder/'calibration.npz',scale=stored_scale,shift=stored_shift)
    np.save(folder/'calibration_raw_z.npy',z)
    write_json(folder/'calibration.json',dict(count=a.calibration_count,seed=a.calibration_seed,sample_block=a.calibration_block,
        x0_clip=a.calibration_clip,block_seed_stride=104729,training_targets_only=True,
        checkpoint_sha256=sha(root/'02_mlpv/final_ema.pt'),training_latents_sha256=sha(root/'01_pca/train_y.npy')))


def train_mode(a):
    populations,manifests,assets=inspect_populations(a)
    forbidden=[row['match_dir'] for rows in manifests.values() for row in rows]+[Path(assets['source']).parent,Path(assets['basis']).parent]
    root=protected_output(a.output_dir,'training_config.json',a.resume,forbidden)
    print(f'TRAIN: {len(manifests["train"])} train / {len(manifests["validation"])} validation; beta={assets["signature"]["geometry"]["shape"]}; q={a.pca_latent_dim}; profile={a.decoder_profile}',flush=True)
    if not a.execute:
        print('DRY RUN: prepare manifests -> training-only PCA/encoding -> EMA MLP-v -> independent variance calibration -> frozen bundle. Add --execute. No files changed.');return
    config=dict(settings=settings(a),inputs=manifests,geometry_assets={k:v for k,v in assets.items() if k not in ('lam','source_points')},code_hashes=code_hashes())
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        immutable_json(root/'training_config.json',config)
        def prepare():
            for split,rows in manifests.items():write_csv(root/f'00_data/{split}_manifest.csv',rows)
        run_stage(root,'prepare',prepare,['00_data/train_manifest.csv','00_data/validation_manifest.csv'])
        run_stage(root,'fit_pca',lambda:fit_codec(populations['train'],assets['lam'],root/'01_pca',a.pca_latent_dim,a.pca_threads,
            dict(training_manifest_sha256=sha(root/'00_data/train_manifest.csv')),a.temporal_dim,a.pca_solver),['01_pca/pca_model.npz','01_pca/codec'])
        def encode():
            report={}
            for split,population in populations.items():
                y,report[split]=encode_population(population,root/'01_pca',a.device,precise_temporal=a.decoder_profile=='hippocampus');np.save(root/f'01_pca/{split}_y.npy',y)
            write_json(root/'01_pca/reconstruction_metrics.json',report)
        run_stage(root,'encode',encode,['01_pca/train_y.npy','01_pca/validation_y.npy','01_pca/reconstruction_metrics.json'])
        def train():
            from .velocity_diffusion_training import train as train_model
            options=SimpleNamespace(**vars(a),q=a.pca_latent_dim,hidden_dim=a.mlpv_hidden_dim,epochs=a.mlpv_epochs,
                seed=a.training_seed,out=root/'02_mlpv',train_y=root/'01_pca/train_y.npy',val_y=root/'01_pca/validation_y.npy')
            train_model(options)
        run_stage(root,'train_mlpv',train,['02_mlpv/final_ema.pt','02_mlpv/completed.json','02_mlpv/metrics.jsonl','02_mlpv/resume.pt'])
        run_stage(root,'calibrate',lambda:fit_calibration(root,a),['03_calibration/calibration.npz','03_calibration/calibration.json','03_calibration/calibration_raw_z.npy'])
        def freeze():
            paths=['01_pca/pca_model.npz','01_pca/codec','02_mlpv/final_ema.pt','03_calibration/calibration.npz']
            bundle=dict(format=FORMAT,method='Calibrated Geometry-Refined Velocity Diffusion (CalVel-Diff)',
                model_identity='energy-weighted beta PCA + cosine VP latent v-prediction residual MLP; ancestral DDPM',
                training_settings=settings(a),geometry=assets['signature']['geometry'],decoder_profile=a.decoder_profile,
                source=assets['source'],basis=assets['basis'],source_sha256=assets['signature']['source_sha256'],basis_sha256=assets['signature']['basis_sha256'],
                artifact_hashes=digest_files(root,paths),checkpoint_rule='final-epoch EMA',environment=environment_info(a.device),
                calibration=json.loads((root/'03_calibration/calibration.json').read_text()),code_hashes=code_hashes(),
                sampling_defaults=dict(n_samples=250,sample_seed=20261231,sample_block=256,x0_clip=6.,arms=['full_beta'],
                    refinement_updates=16,refinement_step=.002,contact_refresh=4,reference_dimension=1536,
                    reference_scale=.2991466820240021,band_caps=[.02,.05,.08],rms_cap=.05,det_tolerance=1e-5))
            write_json(root/'model_bundle.json',bundle)
        run_stage(root,'freeze',freeze,['model_bundle.json'])
    print(f'TRAIN COMPLETE: {root/"model_bundle.json"}',flush=True)


def verify_bundle(a):
    root=a.model_dir
    if not (root/'model_bundle.json').is_file():raise ValueError('A completed train-mode model_bundle.json is required; sampling never fits missing assets.')
    bundle=json.loads((root/'model_bundle.json').read_text())
    if bundle.get('format')!=FORMAT:raise ValueError('Unsupported model bundle format; archived locked bundles require an explicit migration, not guessed defaults.')
    for name,digest in bundle['artifact_hashes'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Model artifact reference leaves the bundle directory.')
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'Frozen model artifact missing/changed: {path}; no refitting was performed.')
    source=resolve_file(a.source or bundle['source'],root);basis=resolve_file(a.basis or bundle['basis'],root)
    if sha(source)!=bundle['source_sha256'] or sha(basis)!=bundle['basis_sha256']:raise ValueError('Sampling requires the exact fitted source/basis content, including eigenvector coordinates.')
    cal=bundle['calibration']
    calibration_seeds={cal['seed']+104729*i for i in range((cal['count']+cal['sample_block']-1)//cal['sample_block'])}
    generation_seeds={a.sample_seed+104729*i for i in range((a.n_samples+a.sample_block-1)//a.sample_block)}
    if calibration_seeds&generation_seeds:raise ValueError('Calibration and sample RNG blocks must be disjoint; choose a different --sample-seed.')
    return bundle,source,basis


def refinement_config(a,geometry,hippocampus=False):
    R=geometry['shape'][1];steps=geometry['n_steps']
    late=(13,17,20) if hippocampus else (int(steps*2/3),int(steps*5/6),steps)
    if 'full_beta' in a.arms and geometry['shape'][0]!=steps+1:
        raise ValueError('The original full-beta outside-risk penalty requires beta knot count == integration steps + 1. No risk/controls were resampled. Use matching grids or explicitly choose --arms mlp/calibrated_mlp.')
    if 'full_beta' in a.arms and (R<3 or len(set(late))<3 or late[0]<1):raise ValueError('Full-beta refinement requires at least 3 control modes and three distinct positive late integration knots; increase matched integration steps or choose an unrefined arm explicitly.')
    one,two=(171,342) if hippocampus else (R//3,2*R//3)
    locked=all(geometry[name]==LOCKED_GEOMETRY[name] for name in LOCKED_GEOMETRY if name!='shape')
    return RefinementConfig(p_scale=a.p_scale,band_limits=tuple(a.band_caps),latent_rms_cap=a.rms_cap,
        clearance_margin=a.clearance_margin,det_risk_threshold=a.det_risk_threshold,max_pairs=a.max_pairs,
        reference_dimension=a.reference_dimension,reference_scale=a.reference_scale,loss_weights=tuple(a.loss_weights),
        nonregression_weight=a.nonregression_weight,late_knots=late,bands=(slice(0,one),slice(one,two),slice(two,R)),
        n_steps=steps,radii=tuple(geometry['radii']),weights=tuple(geometry['weights']),kernel=geometry['kernel'],
        max_neighbors=geometry['max_neighbors'],integrator=geometry['integrator'],locked_geometry=locked,
        hippocampus_geometry=hippocampus,temporal_risk_sigma=2.5 if hippocampus else 1.5)


def sample_mode(a):
    bundle,source_file,basis_file=verify_bundle(a)
    hippocampus=bundle.get('decoder_profile')=='hippocampus'
    cfg=refinement_config(a,bundle['geometry'],hippocampus)
    root=protected_output(a.output_dir,'sampling_config.json',a.resume,[a.model_dir,source_file.parent,basis_file.parent])
    source=load_mesh(str(source_file),torch.device(a.device if a.execute else 'cpu'),torch.float32)
    faces=source.faces
    if 'full_beta' in a.arms and len(faces)<25:raise ValueError('The retained contact miner requires at least 25 triangles; use a suitable surface or explicitly select --arms mlp/calibrated_mlp for a small decoder test.')
    print(f'SAMPLE: {a.n_samples} draws; arms={a.arms}; beta={bundle["geometry"]["shape"]}; frozen assets={a.model_dir}',flush=True)
    if a.x0_clip!=bundle['calibration']['x0_clip']:print('WARNING: sample clipping differs from the frozen calibration draw settings; the existing calibration will be reused unchanged.',flush=True)
    if not a.execute:
        print('DRY RUN: verify frozen PCA/EMA/normalization/calibration -> ancestral DDPM -> beta decode -> selected refinement -> Eulerian surface integration. Add --execute. No files changed.');return
    config=dict(settings=settings(a),model_bundle_sha256=sha(a.model_dir/'model_bundle.json'),
                source=str(source_file),basis=str(basis_file),geometry=bundle['geometry'],code_hashes=code_hashes(),environment=environment_info(a.device))
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        immutable_json(root/'sampling_config.json',config)
        ck,net=load_network(a.model_dir,bundle,a.device)
        def sample():
            folder=root/'04_samples';folder.mkdir(exist_ok=True)
            raw=draw(net,ck,a.n_samples,a.sample_seed,a.sample_block,a.x0_clip).numpy()
            y=raw*np.asarray(ck['y_std'])+np.asarray(ck['y_mean'])
            with np.load(a.model_dir/'03_calibration/calibration.npz',allow_pickle=False) as cal:
                yc=y*cal['scale']+cal['shift']
            zc=(yc-np.asarray(ck['y_mean']))/np.asarray(ck['y_std'])
            for name,value in dict(raw_z=raw,raw_y=y,calibrated_z=zc,calibrated_y=yc).items():
                if not np.isfinite(value).all():raise FloatingPointError(f'Nonfinite {name}')
                np.save(folder/(name+'.npy'),value.astype(np.float32))
            write_json(folder/'sampling.json',dict(count=a.n_samples,seed=a.sample_seed,block_size=a.sample_block,
                block_seed_stride=104729,calibration_refitted=False,checkpoint=str(a.model_dir/'02_mlpv/final_ema.pt')))
        run_stage(root,'sample',sample,['04_samples'])
        del net
        if hippocampus:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        U,lam,_=load_mercer_basis(str(basis_file),bundle['geometry']['shape'][1],torch.device('cpu'),torch.float32)
        codec=FrozenBetaCodec(a.model_dir,ck,U.numpy(),lam.numpy(),bundle['geometry']['shape']).to(a.device).requires_grad_(False)
        if len(source.points)!=len(codec.U):raise ValueError('Frozen source/basis vertex count mismatch.')
        miner=RiskMiner(source.points,faces,cfg) if 'full_beta' in a.arms else None
        options=SimpleNamespace(correction_steps=a.refinement_updates,refresh_every=a.contact_refresh,det_tolerance=a.det_tolerance)
        raw=np.load(root/'04_samples/raw_z.npy');cal=np.load(root/'04_samples/calibrated_z.npy')
        signature=sha(root/'sampling_config.json')+sha(root/'04_samples/raw_z.npy')+sha(root/'04_samples/calibrated_z.npy')
        rows=[]
        for i in range(a.n_samples):
            x=torch.as_tensor(cal[i:i+1],device=a.device);base_q=base_d=None
            for arm in a.arms:
                out=root/'05_generation'/arm;record=out/'records'/f'{i:04d}.json'
                if record.exists():
                    saved=json.loads(record.read_text())
                    if saved['generation_fingerprint']!=signature:raise ValueError(f'Stale generation record: {record}')
                    if not all((out/name).is_file() and sha(out/name)==digest for name,digest in saved['output_hashes'].items()):raise ValueError(f'Corrupted generated output: {record}')
                    rows.append(saved);continue
                if arm!='mlp' and base_q is None:
                    with torch.no_grad():base=codec.decode_z(x);base_q,base_d=integrate(source.points,base[0],codec.U,config=cfg)
                if arm=='full_beta':
                    with torch.no_grad():
                        context=tensor_context(miner.context(base_q,base_d),a.device)
                    beta,q,d,metrics,history=full_correct(x,context,codec,source.points,faces,cfg,miner,a.refinement_step,options,restricted=False)
                    metrics['refinement_applied']=True
                else:
                    xx=x if arm=='calibrated_mlp' else torch.as_tensor(raw[i:i+1],device=a.device)
                    with torch.no_grad():
                        beta=codec.decode_z(xx)
                        q,d=(base_q,base_d) if arm=='calibrated_mlp' else integrate(source.points,beta[0],codec.U,config=cfg)
                    metrics=dict(minimum_detJ=float(d.min()),refinement_applied=False);history=[]
                if not all(torch.isfinite(t).all() for t in (beta,q,d)):raise FloatingPointError(f'Nonfinite generated {arm} sample {i}')
                for sub in ('beta','records','surfaces','trajectories','history'):(out/sub).mkdir(parents=True,exist_ok=True)
                outputs=[];name=f'beta/{i:04d}_beta_knots.npy';np.save(out/name,beta.detach()[0].cpu().numpy());outputs.append(name)
                if a.write_vtk:
                    mesh=pv.PolyData(q[-1].detach().cpu().numpy().astype(np.float64),np.asarray(source.mesh.faces))
                    mesh.point_data['detJ_endpoint']=d[-1].detach().cpu().numpy();mesh.field_data['minimum_detJ_all']=np.array([float(d.min())])
                    name=f'surfaces/sample_{i:04d}.vtk';mesh.save(out/name);outputs.append(name)
                if a.save_q_knots:
                    name=f'trajectories/{i:04d}.npz';np.savez_compressed(out/name,q=q.detach().cpu().numpy(),detJ=d.detach().cpu().numpy());outputs.append(name)
                if history:
                    name=f'history/{i:04d}.json';write_json(out/name,history);outputs.append(name)
                if a.exact_si:
                    from .surface_quality import mesh_quality_metrics
                    metrics['endpoint_qc']=mesh_quality_metrics(q[-1].detach().cpu().numpy().astype(np.float64),faces.cpu().numpy(),compute_self_intersections=True)
                metrics.update(method=ARMS[arm],arm=arm,sample=i,generation_fingerprint=signature,output_hashes={name:sha(out/name) for name in outputs})
                write_json(record,metrics);rows.append(metrics)
                print(f'GENERATED {i+1}/{a.n_samples} {arm}; min detJ={float(d.min()):.6g}',flush=True)
        write_csv(root/'05_generation/per_sample_metrics.csv',rows)
        write_json(root/'05_generation/completed.json',dict(count_per_arm=a.n_samples,arms=a.arms,total=len(rows),
            rejection=False,refined_beta_projected_to_pca=False,mesh_repair=False,calibration_refitted=False))
    print(f'SAMPLE COMPLETE: {root/"05_generation"}',flush=True)


def main(argv=None):
    a=build_parser().parse_args(argv);validate_arguments(a)
    with legacy_precision():
        if a.mode=='train':train_mode(a)
        else:sample_mode(a)


def cli():
    try:main()
    except (ValueError,OSError,RuntimeError,KeyError,FloatingPointError) as exc:
        print(f'Error: {exc}',file=sys.stderr);raise SystemExit(1) from None
