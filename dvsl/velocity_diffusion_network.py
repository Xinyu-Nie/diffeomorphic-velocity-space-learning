"""Residual MLP for cosine VP diffusion with v-prediction (not flow matching)."""
import math
from typing import Tuple
import numpy as np
import torch
from torch import nn

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = int(emb_dim)
        if self.emb_dim < 2:
            raise ValueError("emb_dim must be >= 2")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        half = self.emb_dim // 2
        device = t.device
        dtype = t.dtype
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < self.emb_dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], self.emb_dim - emb.shape[1]), device=device, dtype=dtype)], dim=-1)
        return emb

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class VelocityPredictionMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, time_emb_dim: int = 64,
                 n_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = self.latent_dim + time_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(int(n_blocks))
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        temb = self.time_embed(t)
        h = torch.cat([x, temb], dim=-1)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h)
        return self.output(h)

def compute_standardization(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)

def cosine_schedule(T: int, device, s: float = 0.008):
    steps = torch.arange(T + 1, dtype=torch.float64, device=device)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    ab_full = f / f[0]  # length T+1, ab_full[0]=1 (t=0, clean)
    betas = 1.0 - (ab_full[1:] / ab_full[:-1])
    betas = torch.clamp(betas, max=0.999).to(torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)  # length T, alpha_bar[t] = ab at step t+1 (1-indexed)
    return betas, alphas, alpha_bar

@torch.no_grad()
def sample_diffusion(mlp,ck,n,seed,sampler,clip=6.,branch=None,codec=None,strength=1.,branch_batch=4,branch_count=None):
    """Both solvers use identical clipping; matched seed => matched initial noise.

    Generate the entire noise tensor independently of branch microbatching.
    All step-index tables and ancestral posterior variance come from checkpoint.
    """
    device=next(mlp.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn(n,ck['config']['latent_dim'],device=device,generator=generator)
    ab=torch.as_tensor(ck['alpha_bar'],device=device,dtype=torch.float32)
    al=torch.as_tensor(ck['alphas'],device=device,dtype=torch.float32)
    be=torch.as_tensor(ck['betas'],device=device,dtype=torch.float32)
    times=list(range(len(ab)-1,-1,-1)) if sampler=='ddpm' else torch.linspace(len(ab)-1,0,100).round().long().tolist()
    for j,t in enumerate(times):
        norm=torch.full((n,),(t+.5)/len(ab),device=device)
        v=mlp(x,norm);a=ab[t].sqrt();b=(1-ab[t]).sqrt()
        if branch is not None and t<=500 and strength:
            clean=(a*x-b*v).detach()
            limit=n if branch_count is None else min(n,branch_count)
            for i in range(0,limit,branch_batch):
                stop=min(i+branch_batch,limit)
                dv,_=branch(x[i:stop],norm[i:stop],codec,clean[i:stop],None)
                v[i:stop]+=strength*dv
        eps=b*x+a*v
        clean=(x-b*eps)/a
        if clip>0:
            clean=clean.clamp(-clip,clip);eps=(x-a*clean)/b
        if sampler=='ddpm':
            prev=ab[t-1] if t else ab.new_tensor(1.)
            mu=(x-be[t]/b*eps)/al[t].sqrt()
            x=mu+((1-prev)/(1-ab[t])*be[t]).sqrt()*torch.randn(x.shape,device=device,generator=generator) if t else mu
        elif j+1<len(times):
            prev=ab[times[j+1]]
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        else:x=clean
    assert torch.isfinite(x).all()
    return x

