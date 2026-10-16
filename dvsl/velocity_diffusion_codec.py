"""Training-only temporal/PCA transforms and the frozen affine beta decoder."""
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from .generation_io import sha, write_json

def precise_mm(a,b):
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return a@b
    finally:torch.backends.cuda.matmul.allow_tf32=previous

def weight_beta(alpha: np.ndarray, lam: np.ndarray) -> np.ndarray:
    # alpha: (S,T,N,3), lam: (N,)
    sqrt_lam = np.sqrt(np.maximum(lam, 1e-12))[None, None, :, None]
    return alpha * sqrt_lam

def unweight_beta(z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    inv_sqrt_lam = 1.0 / np.sqrt(np.maximum(lam, 1e-12))[None, None, :, None]
    return z * inv_sqrt_lam

def fit_temporal_basis(z: np.ndarray, temporal_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    z: (S,T,N,3) whitened coefficients.
    Returns:
      mean_z: (T,N,3)
      B: (T,M) shared temporal basis
      explained: (T,) explained variance ratio of temporal covariance eigenvalues
    """
    S, T, N, C = z.shape
    mean_z = z.mean(axis=0)  # (T,N,3)
    centered = z - mean_z[None, ...]
    seq = centered.transpose(0, 2, 3, 1).reshape(S * N * C, T)  # pooled temporal sequences
    cov = (seq.T @ seq) / max(1, seq.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    explained = evals / np.maximum(evals.sum(), 1e-12)
    M = min(max(1, temporal_dim), T)
    B = evecs[:, :M]
    # deterministic sign convention
    idx = np.argmax(np.abs(B), axis=0)
    sgn = np.sign(B[idx, np.arange(B.shape[1])])
    sgn[sgn == 0] = 1.0
    B = B * sgn[None, :]
    return mean_z, B, explained

def project_temporal(z: np.ndarray, mean_z: np.ndarray, B: np.ndarray) -> np.ndarray:
    # z: (S,T,N,3) -> C: (S,N,M,3)
    centered = z - mean_z[None, ...]
    tmp = centered.transpose(0, 2, 3, 1)  # (S,N,3,T)
    coeff = np.einsum('snct,tm->snmc', tmp, B)  # (S,N,M,3)
    return coeff

def reconstruct_temporal(C: np.ndarray, mean_z: np.ndarray, B: np.ndarray) -> np.ndarray:
    # C: (S,N,M,3) -> z: (S,T,N,3)
    tmp = np.einsum('snmc,tm->snct', C, B)  # (S,N,3,T)
    z = tmp.transpose(0, 3, 1, 2) + mean_z[None, ...]
    return z

def fit_sample_pca(C: np.ndarray, latent_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    C: (S,N,M,3)
    Returns:
      mean_C_flat: (D,)
      components: (q,D)
      y: (S,q)
      explained_ratio: (rank,)
    """
    S = C.shape[0]
    X = C.reshape(S, -1)
    mean_X = X.mean(axis=0)
    Xc = X - mean_X[None, :]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = Vt.shape[0]
    q = min(max(1, latent_dim), rank)
    components = Vt[:q]  # (q,D)
    y = Xc @ components.T
    denom = max(1, S - 1)
    evals = (s ** 2) / denom
    explained_ratio = evals / np.maximum(evals.sum(), 1e-12)
    return mean_X, components, y, explained_ratio

def decode_from_y(y: np.ndarray, mean_C_flat: np.ndarray, components: np.ndarray,
                  mean_z: np.ndarray, B: np.ndarray, lam: np.ndarray,
                  n_basis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y: (S,q) or (q,)
    Returns:
      C: (S,N,M,3)
      z: (S,T,N,3)
      alpha: (S,T,N,3)
    """
    if y.ndim == 1:
        y = y[None, :]
    X = mean_C_flat[None, :] + y @ components
    M = B.shape[1]
    S = X.shape[0]
    C = X.reshape(S, n_basis, M, 3)
    z = reconstruct_temporal(C, mean_z, B)
    alpha = unweight_beta(z, lam)
    return C, z, alpha


def fit_codec(train_beta,lam,out,q,threads=8,training_metadata=None,temporal_dim=0,pca_solver='svd'):
    """Fit once, retaining every temporal component; reject an infeasible q."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    S,T,R,_=train_beta.shape
    M=temporal_dim or T
    if not 1<=M<=T:raise ValueError('Temporal dimension must be within the saved knot count.')
    maximum=min(S-1,3*M*R)
    if q>maximum:
        raise ValueError(f'PCA latent dimension {q} exceeds the centered rank bound {maximum}: training count={S}, raw dimension={3*T*R}. Add independent training runs or explicitly lower --pca_latent_dim.')
    if len(lam)!=R or not np.isfinite(lam).all() or (lam<=1e-12).any():
        raise ValueError('The fitted control-mode Mercer eigenvalues must be finite and greater than 1e-12.')
    with threadpool_limits(limits=threads):
        z=weight_beta(train_beta.astype(np.float64),lam)
        mean,B,te=fit_temporal_basis(z,M)
        C=project_temporal(z,mean,B);del z
        if pca_solver=='gram':
            from scipy.linalg import eigh
            X=C.reshape(S,-1);meanC=X.mean(0);X=X-meanC
            values,vectors=eigh(X@X.T,check_finite=True,driver='evd')
            values,vectors=values[::-1],vectors[:,::-1]
            if values[q-1]<=values[0]*1e-12:raise ValueError('Requested PCA dimension exceeds numerical Gram rank.')
            V=(vectors[:,:q].T@X)/np.sqrt(values[:q,None])
            pivots=np.argmax(np.abs(V),axis=1);signs=np.sign(V[np.arange(q),pivots])
            V*=np.where(signs==0,1,signs)[:,None]
            values=np.maximum(values,0);explained=values/values.sum()
        elif pca_solver=='svd':meanC,V,y,explained=fit_sample_pca(C,q)
        else:raise ValueError('Unknown PCA solver')
        del C
    relative_singular=np.sqrt(np.maximum(explained,0))
    tolerance=max(S,3*T*R)*np.finfo(np.float64).eps*relative_singular.max(initial=0)
    # Gram roundoff can leave a tiny positive eigenvalue in the centered null
    # direction. Report a rank consistent with the exact centered rank bound.
    rank=min(maximum,int((relative_singular>tolerance).sum()))
    if q>rank:
        raise ValueError(f'Requested PCA dimension {q} exceeds measured centered training rank {rank}; no latent dimension was lowered. Add independent paths or explicitly lower --pca_latent_dim.')
    metadata=dict(training_metadata or {},training_count=S,fitting_partition='train only',temporal_dim=M,numerical_rank=rank,pca_solver=pca_solver)
    np.savez_compressed(out/'pca_model.npz',lam=lam,mean_z=mean,temporal_basis=B,
        temporal_explained=te,meanC=meanC,Vt=V,cumvar=np.cumsum(explained),meta=json.dumps(metadata))
    compile_codec(out/'pca_model.npz',out/'codec',q,threads)


def compile_codec(pca,out,q,threads=8):
    """Same mode/time/xyz permutation and affine map as the locked compiler."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with np.load(pca,allow_pickle=False) as z:
        V,Cmean,B,lam=z['Vt'],z['meanC'],z['temporal_basis'],z['lam']
        T,M=B.shape;R=len(lam)
        if M>T or q>len(V):raise ValueError('Invalid temporal basis or requested fitted PCA rank.')
        np.testing.assert_allclose(B.T@B,np.eye(M),atol=1e-6)
        decoder=np.empty((q,3*T*R),np.float32)
        for i in range(0,q,64):
            c=V[i:min(i+64,q)].astype(np.float64).reshape(-1,R,M,3)
            decoder[i:i+len(c)]=(np.einsum('snmc,tm->stnc',c,B)/np.sqrt(lam)[None,None,:,None]).reshape(len(c),-1)
        mean=((np.einsum('nmc,tm->tnc',Cmean.reshape(R,M,3),B)+z['mean_z'])/np.sqrt(lam)[None,:,None]).reshape(-1).astype(np.float32)
        weight=np.broadcast_to(lam[None,:,None],(T,R,3)).reshape(-1).astype(np.float32)
    np.save(out/'decoder.npy',decoder);np.save(out/'mean_beta.npy',mean);np.save(out/'metric_weight.npy',weight)
    with threadpool_limits(limits=threads):
        Q=decoder*np.sqrt(weight)[None]
        error=float(np.max(np.abs(Q@Q.T-np.eye(q))))
    if error>2e-4:raise ValueError(f'Nonorthogonal PCA decoder: {error}')
    write_json(out/'codec.json',dict(pca_sha256=sha(pca),q=q,shape=[T,R,3],
        fitting_metric='sqrt(lambda)*beta',orthogonality_max_error=error))


def encode_population(beta,folder,device,precise_temporal=False):
    folder=Path(folder)
    if precise_temporal:
        with np.load(folder/'pca_model.npz',allow_pickle=False) as p:
            lam,mean,B,cm,V=p['lam'],p['mean_z'],p['temporal_basis'],p['meanC'],p['Vt']
        values=[];err=energy=0.
        for i in range(0,len(beta),64):
            raw=beta[i:i+64].astype(np.float64);z=weight_beta(raw,lam)
            C=project_temporal(z,mean,B)
            y=(C.reshape(len(raw),-1)-cm)@V.T
            _,reconstructed,_=decode_from_y(y,cm,V,mean,B,lam,len(lam))
            err+=float(np.square(z-reconstructed).sum());energy+=float(np.square(z-mean).sum())
            values.append(y.astype(np.float32))
        return np.concatenate(values),dict(count=len(beta),centered_energy_relative_mse=err/max(energy,1e-30))
    decoder=torch.as_tensor(np.load(folder/'codec/decoder.npy'),device=device)
    mean=torch.as_tensor(np.load(folder/'codec/mean_beta.npy'),device=device)
    weight=torch.as_tensor(np.load(folder/'codec/metric_weight.npy'),device=device)
    values=[];err=energy=0.
    with torch.no_grad():
        for i in range(0,len(beta),64):
            b=torch.as_tensor(beta[i:i+64].copy(),device=device).flatten(1)
            y=precise_mm((b-mean)*weight,decoder.T)
            rec=precise_mm(y,decoder)+mean
            err+=float(((b-rec).square()*weight).sum());energy+=float(((b-mean).square()*weight).sum())
            values.append(y.cpu().numpy())
    return np.concatenate(values),dict(count=len(beta),centered_energy_relative_mse=err/max(energy,1e-30))


class FrozenBetaCodec(nn.Module):
    def __init__(self,root,checkpoint,U,lam,shape):
        super().__init__();folder=Path(root)/'01_pca/codec'
        fields=dict(components=np.load(folder/'decoder.npy'),mean_flat=np.load(folder/'mean_beta.npy'),
            metric_weight=np.load(folder/'metric_weight.npy'),U=U,lam=lam,sqrt_lam=np.sqrt(lam),
            y_mean=np.asarray(checkpoint['y_mean']).reshape(1,-1),y_std=np.asarray(checkpoint['y_std']).reshape(1,-1))
        for key,value in fields.items():self.register_buffer(key,torch.as_tensor(np.array(value),dtype=torch.float32))
        self.latent_dim=checkpoint['config']['latent_dim'];self.time_samples,self.basis_dim,_=shape

    def decode_z(self,z):
        y=z.float()*self.y_std+self.y_mean
        return (precise_mm(y,self.components)+self.mean_flat).reshape(-1,self.time_samples,self.basis_dim,3)

    def encode_beta(self,beta):
        y=precise_mm((beta.flatten(1)-self.mean_flat)*self.metric_weight,self.components.T)
        return (y-self.y_mean)/self.y_std

    def encode_beta_delta(self,beta):return precise_mm(beta.flatten(1)*self.metric_weight,self.components.T)/self.y_std
    def p_from_beta(self,beta):return torch.einsum('vr,btrc->btvc',self.U,beta)
    def beta_from_p(self,p):return torch.einsum('vr,btvc->btrc',self.U,p)

