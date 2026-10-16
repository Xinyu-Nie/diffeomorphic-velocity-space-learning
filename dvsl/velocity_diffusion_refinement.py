"""Frozen-calibration full-coefficient projected geometric refinement.

Only kernels called by the locked pipeline are included; graph residual models,
archived data paths, ablation runners and evaluation jobs are not imported.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy.spatial import cKDTree
from scipy import sparse
from .surface_matching import interpolate_knots
from .kernels import build_radius_graph, wendland_phi_dphi
from .velocity_diffusion_codec import precise_mm

@dataclass
class RefinementConfig:
    p_scale: float = .012632200545880998
    band_limits: tuple = (.02,.05,.08)
    latent_rms_cap: float = .05
    preserve_radial_energy: bool = True
    clearance_margin: float = .02
    det_risk_threshold: float = .15
    max_pairs: int = 1024
    include_near_contacts: bool = True
    reference_dimension: int = 1536
    reference_scale: float = .2991466820240021
    loss_weights: tuple = (.1,.1,.05,.001,.001)
    nonregression_weight: float = 1.
    late_knots: tuple = (8,10,12)
    bands: tuple = (slice(0,512),slice(512,1024),slice(1024,1536))
    n_steps: int = 12
    radii: tuple = (.19,)
    weights: tuple = (1.,)
    kernel: str = 'c2'
    max_neighbors: int = 512
    integrator: str = 'midpoint'
    locked_geometry: bool = True
    hippocampus_geometry: bool = False
    temporal_risk_sigma: float = 1.5

def integrate_locked(source, beta, U, *, gradients=False,n_steps=12,radius=.19,max_neighbors=512):
    """Identical 12-step C2 midpoint q/J flow, with per-step checkpointing."""
    q=source; J=torch.eye(3,device=q.device,dtype=q.dtype).expand(len(q),3,3).clone()
    qs=[q]; Js=[J]; dt=1./n_steps
    def field(q,b,edges):
        # Identical native field algebra, but neighbor identities are fixed
        # for this forward/backward pair (the original graph is detached).
        p=U@b; row,col=edges
        difference=q[row]-q[col]; distance=difference.norm(dim=-1)
        phi,dphi=wendland_phi_dphi(distance/radius,'c2'); pe=p[col]
        velocity=torch.zeros_like(q); velocity.index_add_(0,row,phi[:,None]*pe)
        grad=(dphi/(radius*distance.clamp_min(torch.finfo(q.dtype).eps)))[:,None]*difference
        D=q.new_zeros((len(q),3,3));D.index_add_(0,row,pe[:,:,None]*grad[:,None,:])
        return velocity,D
    def make_advance():
        # A separate closure per midpoint step must survive until backward.
        edges={}
        def advance(q,J,b0,bm):
            if 'start' not in edges:edges['start']=build_radius_graph(q.detach(),radius,max_neighbors)
            v,D=field(q,b0,edges['start'])
            qm=q+.5*dt*v;Jm=J+.5*dt*(D@J)
            if 'mid' not in edges:edges['mid']=build_radius_graph(qm.detach(),radius,max_neighbors)
            v,D=field(qm,bm,edges['mid'])
            return q+dt*v,J+dt*(D@Jm)
        return advance
    for k in range(n_steps):
        b0=interpolate_knots(beta,k*dt); bm=interpolate_knots(beta,k*dt+.5*dt)
        advance=make_advance()
        q,J=checkpoint(advance,q,J,b0,bm,use_reentrant=False) if gradients else advance(q,J,b0,bm)
        qs.append(q); Js.append(J)
    return torch.stack(qs),torch.linalg.det(torch.stack(Js))

def integrate(source, beta, U, *, gradients=False, config=None):
    if config is not None and config.hippocampus_geometry:
        return integrate_locked(source,beta,U,gradients=gradients,n_steps=20,radius=1.,max_neighbors=2562)
    if config is None or config.locked_geometry:
        return integrate_locked(source,beta,U,gradients=gradients)
    from .surface_matching import integrate_eulerian_path
    flow=integrate_eulerian_path(source,beta,U,config.n_steps,config.radii,
        config.weights,config.kernel,config.max_neighbors,config.integrator)
    return flow.q,torch.linalg.det(flow.J)

def weighted_quantile(values, weights, levels):
    order=values.argsort(); v=values[order]; w=weights[order]
    c=(w.cumsum(0)-.5*w)/w.sum().clamp_min(1e-12)
    p=values.new_tensor(levels); hi=torch.searchsorted(c.detach(),p).clamp(1,len(c)-1)
    lo=hi-1; a=((p-c[lo])/(c[hi]-c[lo]).clamp_min(1e-12)).clamp(0,1)
    return v[lo]+a*(v[hi]-v[lo])

def morphology_proxy(q, faces):
    """Same differentiable proxy on original training and generated surfaces."""
    t=q[faces]; e1=t[:,1]-t[:,0]; e2=t[:,2]-t[:,0]
    cross=torch.cross(e1,e2,dim=-1); area2=cross.norm(dim=-1).clamp_min(1e-10)
    area=.5*area2.sum(); volume=(t[:,0]*torch.cross(t[:,1],t[:,2],dim=-1)).sum().abs()/6
    va=q.new_zeros(len(q)); angles=q.new_zeros(len(q)); lap=torch.zeros_like(q)
    for i in range(3):
        a=t[:,(i+1)%3]-t[:,i]; b=t[:,(i+2)%3]-t[:,i]
        cos=(a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)
        angles.index_add_(0,faces[:,i],torch.acos(cos.clamp(-1+1e-7,1-1e-7)))
        va.index_add_(0,faces[:,i],area2/6)
        cot=(a*b).sum(-1)/area2
        u=faces[:,(i+1)%3]; v=faces[:,(i+2)%3]; d=(q[u]-q[v])*cot[:,None]
        lap.index_add_(0,u,d); lap.index_add_(0,v,-d)
    h=.25*lap.norm(dim=-1)/va.clamp_min(1e-10)
    g=(2*np.pi-angles)/va.clamp_min(1e-10)
    return torch.cat((torch.stack((area,volume)),weighted_quantile(h,va,[.5,.9]),
                      weighted_quantile(g,va,[.05,.95])))

def separating_gap(q, faces, pairs, scale):
    """Signed SAT gap, including in-plane axes for the coplanar special case."""
    a=q[faces[pairs[:,0]]]; b=q[faces[pairs[:,1]]]
    ea=torch.roll(a,-1,1)-a; eb=torch.roll(b,-1,1)-b
    na=torch.cross(ea[:,0],ea[:,1],dim=-1); nb=torch.cross(eb[:,0],eb[:,1],dim=-1)
    axes=torch.cat((na[:,None],nb[:,None],
        torch.cross(ea[:,:,None].expand(-1,3,3,-1),eb[:,None].expand(-1,3,3,-1),dim=-1).flatten(1,2),
        torch.cross(na[:,None].expand_as(ea),ea,dim=-1),
        torch.cross(nb[:,None].expand_as(eb),eb,dim=-1)),dim=1).detach()
    norm=axes.norm(dim=-1); unit=axes/norm[:,:,None].clamp_min(1e-12)
    pa=torch.einsum('pac,pvc->pav',unit,a); pb=torch.einsum('pac,pvc->pav',unit,b)
    gap=torch.maximum(pb.min(-1).values-pa.max(-1).values,pa.min(-1).values-pb.max(-1).values)
    gap=gap.masked_fill(norm<1e-12,-1e6).max(-1).values
    return gap/scale.clamp_min(1e-8)

class RiskMiner:
    """Subject-specific late broad phase with topology-dependent margins.

    Shared-vertex triangles are excluded. Non-shared triangles within three
    face hops receive penetration-only supervision, NOT wall repulsion.
    No observed validation/test SI map enters candidate mining.
    """
    def __init__(self, source, faces, config):
        self.source=source; self.faces=faces; self.config=config
        f=faces.cpu().numpy(); nf=len(f); nv=len(source)
        inc=sparse.csr_matrix((np.ones(f.size,dtype=np.int32),
             (np.repeat(np.arange(nf),3),f.ravel())),shape=(nf,nv))
        adj=inc@inc.T; adj.data[:]=1
        reach=adj@adj@adj; reach.data[:]=1
        row,col=reach.nonzero(); self.excluded=np.sort(row.astype(np.int64)*nf+col)
        self.nf=nf; self.faces_np=f
        tri=source[faces]; center=tri.mean(1)
        self.radii=(tri-center[:,None]).norm(dim=-1).max(-1).values
        # Source intrinsic adjacency used only to spread risk one vertex ring.
        e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])); e=np.concatenate((e,e[:,::-1]))
        self.edges=torch.as_tensor(e,device=source.device)

    @torch.no_grad()
    def context(self, q, det):
        risks=[]; pairs_out=[]; scales=[]; gaps=[]; margins=[]
        for k in self.config.late_knots:
            centers=q[k,self.faces].mean(1).cpu().numpy()
            ids=cKDTree(centers).query(centers,k=25,workers=4)[1][:,1:]
            a=np.repeat(np.arange(self.nf),ids.shape[1]); b=ids.ravel()
            codes=np.unique(np.minimum(a,b).astype(np.int64)*self.nf+np.maximum(a,b))
            pos=np.searchsorted(self.excluded,codes)
            allowed=(pos==len(self.excluded))|(self.excluded[np.minimum(pos,len(self.excluded)-1)]!=codes)
            if not self.config.include_near_contacts:codes=codes[allowed];allowed=np.ones(len(codes),dtype=bool)
            pair=torch.as_tensor(np.stack((codes//self.nf,codes%self.nf),-1),device=q.device)
            # Exact-SI convention: faces sharing a vertex are not counted.
            shared=(self.faces[pair[:,0]][:,:,None]==self.faces[pair[:,1]][:,None,:]).any((1,2))
            margin=torch.as_tensor(np.where(allowed,self.config.clearance_margin,0.),device=q.device,dtype=q.dtype)
            pair=pair[~shared];margin=margin[~shared]
            if not len(pair): raise RuntimeError('No nonadjacent collision candidates')
            scale=self.radii[pair].sum(1)
            gap=separating_gap(q[k],self.faces,pair,scale)
            take=(gap-margin).argsort()[:self.config.max_pairs]; pair=pair[take];scale=scale[take];gap=gap[take];margin=margin[take]
            risk=F.relu((self.config.det_risk_threshold-det[k])/self.config.det_risk_threshold).clamp_max(1)
            pr=F.relu((margin-gap)/self.config.clearance_margin).clamp_max(1)
            pr=torch.maximum(pr,.25*(gap<0).to(pr.dtype))
            vertices=self.faces[pair].flatten(); values=pr[:,None,None].expand(-1,2,3).flatten()
            risk.scatter_reduce_(0,vertices,values,reduce='amax',include_self=True)
            spread=risk.clone();spread.scatter_reduce_(0,self.edges[:,0],risk[self.edges[:,1]],reduce='amax',include_self=True)
            risks.append(spread);pairs_out.append(pair);scales.append(scale);gaps.append(gap);margins.append(margin)
        temporal=q.new_zeros((self.config.n_steps+1,len(self.source)))
        for k,r in zip(self.config.late_knots,risks):
            support=torch.exp(-.5*((torch.arange(self.config.n_steps+1,device=q.device)-k)/self.config.temporal_risk_sigma)**2)[:,None]
            temporal=torch.maximum(temporal,support*r)
        def candidates(values):
            # Candidate counts may differ below the configured cap. Keep every
            # selected entry; do not pad, truncate, or change the contact loss.
            if len({tuple(v.shape) for v in values})==1:return torch.stack(values).cpu().numpy()
            return [v.cpu().numpy() for v in values]
        return dict(q=q.cpu().numpy(),det=det.cpu().numpy(),risk=temporal.cpu().numpy(),
            pairs=candidates(pairs_out),pair_scale=candidates(scales),gap=candidates(gaps),
            pair_margin=candidates(margins),proxy=morphology_proxy(q[-1],self.faces).cpu().numpy())

def geometry_loss(q,det,context,faces,cfg):
    clearance=[]; determinant=[]
    for j,k in enumerate(cfg.late_knots):
        gap=separating_gap(q[k],faces,context['pairs'][j],context['pair_scale'][j])
        # Correct pre-existing clearance failures; do not separate healthy walls.
        # Healthy walls may move closer while remaining safely separated.
        # Matching their entire old gap would needlessly freeze local anatomy.
        target=context.get('pair_margin',gap.new_full((3,len(gap)),cfg.clearance_margin))[j]
        deficit=F.relu((target-gap)/cfg.clearance_margin)
        clearance.append(F.smooth_l1_loss(deficit,torch.zeros_like(deficit),reduction='none').topk(min(128,len(gap))).values.mean())
        baseline=context['det'][k]
        target=torch.maximum(.98*baseline,torch.minimum(baseline*1.05,baseline.new_full(baseline.shape,.10)))
        d=(F.relu(target-det[k])/target.clamp_min(.03)).square()
        determinant.append(d.topk(max(1,len(d)//100)).values.mean())
    trust=(q[-1]-context['q'][-1]).square().mean()/.0001
    return torch.stack(clearance).mean(),torch.stack(determinant).mean(),trust

ORIGINAL_GEOMETRY=geometry_loss

def active_contact_penalty(deficit):
    """Do not dilute sparse violations with safe pairs; floor at four pairs."""
    active=(deficit.detach()>0).to(deficit.dtype)
    values=F.smooth_l1_loss(deficit,torch.zeros_like(deficit),reduction='none')
    return (values*active).sum()/active.sum().clamp_min(4.)

def active_geometry(q,det,context,faces,cfg):
    _,dj,trust=ORIGINAL_GEOMETRY(q,det,context,faces,cfg)
    clear=[]
    for j,k in enumerate(cfg.late_knots):
        gap=separating_gap(q[k],faces,context['pairs'][j],context['pair_scale'][j])
        deficit=F.relu((context['pair_margin'][j]-gap)/cfg.clearance_margin)
        clear.append(active_contact_penalty(deficit))
    return torch.stack(clear).mean(),dj,trust

def minreg(det,baseline):
    return ((baseline[1:].min(1).values-det[1:].min(1).values).clamp_min(0)/.03).square().mean()

def feasible(det,baseline,tol):
    return bool(torch.isfinite(det).all() and (det[1:].min(1).values>=baseline[1:].min(1).values-tol).all())

def refreshed(c,new,faces):
    """Accumulate candidates, so refresh never forgets earlier dangerous pairs."""
    out=dict(c);out['pairs']=[];out['pair_scale']=[];out['pair_margin']=[]
    nf=len(faces)
    for j in range(len(c["pairs"])):
        pair=torch.cat((c['pairs'][j],torch.as_tensor(new['pairs'][j],device=faces.device)))
        scale=torch.cat((c['pair_scale'][j],torch.as_tensor(new['pair_scale'][j],device=faces.device)))
        margin=torch.cat((c['pair_margin'][j],torch.as_tensor(new['pair_margin'][j],device=faces.device)))
        code=pair[:,0]*nf+pair[:,1];unique,inv=torch.unique(code,return_inverse=True)
        s=scale.new_zeros(len(unique));s.scatter_reduce_(0,inv,scale,reduce='amax',include_self=True)
        m=margin.new_zeros(len(unique));m.scatter_reduce_(0,inv,margin,reduce='amax',include_self=True)
        out['pairs'].append(torch.stack((unique//nf,unique%nf),1));out['pair_scale'].append(s);out['pair_margin'].append(m)
    return out

def losses(q,d,dz,b,c,faces,cfg):
    clear,dj,trust=active_geometry(q,d,c,faces,cfg)
    cost,outside,ratio=b.losses(dz)
    # An exact-penalty-style term remains sensitive to tiny min-J regressions;
    # a squared penalty alone becomes too weak near the hard feasibility gate.
    nr=minreg(d,c['det'])+(c['det'][1:].min(1).values-d[1:].min(1).values).clamp_min(0).max()/.03
    wc,wj,wq,wb,wo=cfg.loss_weights
    loss=wc*clear+wj*dj+wq*trust+wb*cost+wo*outside+cfg.nonregression_weight*nr
    return loss,dict(objective=float(loss.detach()),contact=float(clear.detach()),minreg=float(nr.detach()),
        det_tail=float(dj.detach()),trust=float(trust.detach()),latent_rms=float(dz.detach().square().mean().sqrt()),
        band_rms=ratio.detach().cpu().tolist()[0])

class EnergyBudget:
    """w = sqrt(lambda) delta_beta / global_y_std; ||w|| / sqrt(d_z).

Using d_z, not the raw dimension, makes total energy identical to the latent
budget. The PCA-restricted control uses the same full-coordinate preconditioner
as the unrestricted arm, isolating access to directions outside the PCA span.
"""
    def __init__(self,z,codec,cfg,risk,restricted=False):
        self.codec=codec;self.cfg=cfg;self.risk=risk;self.restricted=restricted
        self.scale=codec.y_std.flatten()[0]
        assert torch.allclose(codec.y_std,self.scale.expand_as(codec.y_std),rtol=1e-6,atol=0),'This energy match requires global scaling'
        self.rank=codec.latent_dim
        self.root=codec.sqrt_lam[None,None,:,None]
        self.Q=codec.components*codec.metric_weight.sqrt()[None]
        with torch.no_grad():
            self.base=codec.decode_z(z);self.shape=self.base.shape
            centered=self.base-codec.decode_z(torch.zeros_like(z))
            fields=[];den=[]
            for band in cfg.bands:
                field=torch.zeros_like(centered)
                field[:,:,band]=centered[:,:,band]*self.root[:,:,band]
                fields.append(field.flatten(1))
                den.append(field.square().flatten(1).sum(1).clamp_min(1e-12))
            self.den=torch.stack(den,1)
            G=torch.stack(fields,1)
            if restricted:
                G=self.subspace(G.flatten(0,1)).reshape_as(G)
            self.G=G/G.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            self.gram=self.G@self.G.transpose(1,2)+torch.eye(3,device=z.device)[None]*1e-6
        self.rank=cfg.reference_dimension
        self.scale=self.scale.new_tensor(float(cfg.reference_scale))

    def subspace(self,w):
        return precise_mm(precise_mm(w,self.Q.T),self.Q)

    def tangent(self,w):
        if self.restricted:w=self.subspace(w)
        coef=torch.linalg.solve(self.gram,self.G@w[:,:,None])
        return w-(self.G.transpose(1,2)@coef).squeeze(-1)

    def rms(self,w):return (w.square().sum(1)/self.rank).sqrt()

    def decoded(self,w):return w.reshape(self.shape)*self.scale/self.root

    def ratios(self,db):
        energy=torch.stack([(db[:,:,b].square()*self.root[:,:,b].square()).sum((1,2,3)) for b in self.cfg.bands],1)
        return (energy/self.den+1e-20).sqrt()

    @torch.no_grad()
    def project(self,w):
        w=self.tangent(w)
        ratio=self.ratios(self.decoded(w))
        cap=(w.new_tensor(self.cfg.band_limits)[None]/ratio.clamp_min(1e-10)).min(1).values.clamp_max(1)
        cap=torch.minimum(cap,self.cfg.latent_rms_cap/self.rms(w).clamp_min(1e-10)).clamp_max(1)
        return w*cap[:,None]

    def losses(self,w):
        db=self.decoded(w);dp=self.codec.p_from_beta(db)
        # Avoid sqrt(0) followed by squaring: its backward produces 0/0.
        budget=w.square().sum(1).mean()/self.rank/self.cfg.latent_rms_cap**2
        outside=(dp.square()*(1-self.risk[None,:,:,None])).mean()/self.cfg.p_scale**2
        return budget,outside,self.ratios(db)

def full_losses(q,d,w,b,c,faces,cfg):
    loss,m=losses(q,d,w,b,c,faces,cfg)
    # Replace the old dimension-dependent display metric, not the objective.
    m['latent_rms']=float(b.rms(w.detach()).mean())
    m['equivalent_latent_rms']=m['latent_rms']
    return loss,m

def full_correct(x,c,codec,source,faces,cfg,miner,step_size,a,restricted=False):
    b=EnergyBudget(x,codec,cfg,c['risk'],restricted)
    zero=torch.zeros_like(b.base.flatten(1));best=zero.clone();current=zero.clone();variance=zero.clone()
    ctx=c;bestq=c['q'];bestd=c['det'];selected=0;history=[];start=time.monotonic()
    with torch.no_grad():best_score=float(full_losses(bestq,bestd,best,b,ctx,faces,cfg)[0])
    for iteration in range(a.correction_steps+1):
        grad=iteration<a.correction_steps
        w=current.detach().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            q,d=integrate(source,(b.base+b.decoded(w))[0],codec.U,gradients=grad,config=cfg)
            if iteration==0:
                assert float((q.detach()-c['q']).abs().max())<2e-5,'Baseline cache mismatch'
            if iteration>0 and iteration%a.refresh_every==0:
                ctx=refreshed(ctx,miner.context(q.detach(),d.detach()),faces)
                with torch.no_grad():best_score=float(full_losses(bestq,bestd,best,b,ctx,faces,cfg)[0])
            loss,m=full_losses(q,d,w,b,ctx,faces,cfg)
        safe=feasible(d.detach(),c['det'],a.det_tolerance)
        if not np.isfinite(m['objective']):raise FloatingPointError('Nonfinite full-beta objective')
        history.append(dict(iteration=iteration,safe=safe,**m))
        if safe and m['objective']<best_score:
            best_score=m['objective'];best=w.detach().clone();bestq=q.detach().clone();bestd=d.detach().clone();selected=iteration
        if grad:
            g=torch.autograd.grad(loss,w)[0]
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite full-beta gradient')
            with torch.no_grad():
                g=b.tangent(g);variance=.9*variance+.1*g.square()
                direction=-b.tangent(g/(variance.sqrt()+1e-8))
                norm=b.rms(direction)
                if float(norm.max())<1e-10:break
                assert float((g*direction).sum())<=1e-6,'Not a projected descent direction'
                current=b.project(w.detach()+step_size*direction/norm[:,None].clamp_min(1e-10))
        del q,d,loss
    with torch.no_grad():
        ctx=refreshed(ctx,miner.context(bestq,bestd),faces)
        score,_=full_losses(bestq,bestd,best,b,ctx,faces,cfg)
        base_score,base_m=full_losses(c['q'],c['det'],zero,b,ctx,faces,cfg)
        if score>base_score or not bool(best.any()):best=zero;bestq=c['q'];bestd=c['det'];selected=0
        _,m=full_losses(bestq,bestd,best,b,ctx,faces,cfg)
        assert feasible(bestd,c['det'],a.det_tolerance)
        assert bool((b.ratios(b.decoded(best))<=x.new_tensor(cfg.band_limits)+1e-6).all())
        assert float(b.rms(best).max())<=cfg.latent_rms_cap+1e-6
        off=best-b.subspace(best)
        fraction=float(off.square().sum()/best.square().sum().clamp_min(1e-20))
        if restricted:assert fraction<1e-5,'PCA-restricted control left the PCA span'
    m.update(step_size=step_size,selected_iteration=selected,minimum_detJ=float(bestd.min()),
        baseline_objective=base_m['objective'],baseline_contact=base_m['contact'],
        endpoint_rmse=float((bestq[-1]-c['q'][-1]).square().mean().sqrt()),
        max_knot_min_regression=float((c['det'][1:].min(1).values-bestd[1:].min(1).values).max()),
        correction_off_pca_energy_fraction=fraction,restricted_to_pca=restricted,
        safe=True,seconds=time.monotonic()-start)
    return b.base+b.decoded(best),bestq,bestd,m,history

