"""Original EMA v-prediction AdamW training, with explicit mode settings."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .velocity_diffusion_network import VelocityPredictionMLP, compute_standardization, set_seed, cosine_schedule
from .generation_io import write_json as json_write, immutable_json, torch_save, sha

def train(a):
    torch.set_num_threads(4); set_seed(a.seed)
    device=torch.device(a.device); a.out.mkdir(parents=True,exist_ok=True)
    train_y=np.load(a.train_y).astype(np.float32); val_y=np.load(a.val_y).astype(np.float32)
    assert train_y.shape[1]==val_y.shape[1]==a.q and len(train_y)>a.q
    assert np.isfinite(train_y).all() and np.isfinite(val_y).all()
    mean,std=compute_standardization(train_y)
    scale=float(np.sqrt(np.mean(std.astype(np.float64)**2))); std=np.full_like(std,scale)
    x=(train_y-mean)/std; xv=torch.as_tensor((val_y-mean)/std,device=device)
    cfg=dict(latent_dim=a.q,hidden_dim=a.hidden_dim,time_emb_dim=a.time_emb_dim,n_blocks=a.n_blocks,dropout=a.dropout,
             batch_size=a.batch_size,epochs=a.epochs,lr=a.lr,weight_decay=a.weight_decay,grad_clip=a.grad_clip,ema_decay=a.ema,
             seed=a.seed,T=a.diffusion_steps,std_mode='global',param='v',loss_weight='none',
             lr_schedule='cosine',val_every=a.val_every,validation_weights='EMA',val_replicates=4,
             save_every=a.save_every,optimizer='AdamW foreach',precision='float32 with TF32 matmul',device=str(device))
    spec=dict(config=cfg,train_y=str(a.train_y.resolve()),val_y=str(a.val_y.resolve()),
              train_sha256=sha(a.train_y),val_sha256=sha(a.val_y),source_sha256=sha(Path(__file__)),
              candidate_rule=['best_ema_validation','final_ema'])
    immutable_json(a.out/'protocol.json',spec)
    if (a.out/'completed.json').exists(): return
    model=VelocityPredictionMLP(a.q,a.hidden_dim,a.time_emb_dim,a.n_blocks,a.dropout).to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay,foreach=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    be,al,ab=cosine_schedule(a.diffusion_steps,device); root=ab.sqrt(); noise=(1-ab).sqrt()
    # Separate generator: fixed validation noise never changes training RNG.
    vg=torch.Generator(device=device).manual_seed(a.seed+100003)
    vt=torch.randint(0,a.diffusion_steps,(4,len(xv)),device=device,generator=vg)
    # Prefix coordinates use identical validation noise across q dimensions.
    ve=torch.randn((4,len(xv),2048),device=device,generator=vg)[:,:,:a.q]
    history=[]; best=float('inf'); best_epoch=0; start_epoch=1
    resume=a.out/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['config']==cfg
        model.load_state_dict(saved['model_state']);ema.load_state_dict(saved['ema_state'])
        opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng']) if device.type=='cuda' else None
        np.random.set_state(saved['numpy_rng'])
        history=saved['history']; best=saved['best_val_loss'];best_epoch=saved['best_epoch'];start_epoch=saved['epoch']+1
        print('RESUME',start_epoch,'learning rate',opt.param_groups[0]['lr'],flush=True)
        del saved
    # Canonical history drops any uncheckpointed tail after an interruption.
    metrics=a.out/'metrics.jsonl'
    metrics.write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    dl=DataLoader(TensorDataset(torch.as_tensor(x)),batch_size=a.batch_size,shuffle=True,drop_last=False,num_workers=0)
    @torch.no_grad()
    def validation(net):
        net.eval(); values=[]
        for r in range(4):
            aa=root[vt[r]][:,None];bb=noise[vt[r]][:,None]
            pred=net(aa*xv+bb*ve[r],(vt[r].float()+.5)/a.diffusion_steps)
            values.append(float((pred-(aa*ve[r]-bb*xv)).square().mean()))
        return float(np.mean(values))
    def payload(epoch,full=False):
        state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
        d=dict(model_state=state,ema_state=state,config=cfg,y_mean=mean,y_std=std,
               betas=be.cpu().numpy(),alphas=al.cpu().numpy(),alpha_bar=ab.cpu().numpy(),
               epoch=epoch,best_epoch=best_epoch,best_val_loss=best,
               train_y_path=str(a.train_y),val_y_path=str(a.val_y),selection_metric='fixed-noise EMA validation v-MSE',
               history=history,resumable=full)
        if full:
            d.update(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     optimizer=opt.state_dict(),scheduler=sched.state_dict(),cpu_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None,numpy_rng=np.random.get_state())
        return d
    print('TRAIN CONFIG',json.dumps(cfg),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    ep_params=list(ema.parameters());raw_params=list(model.parameters());start=time.monotonic()
    for epoch in range(start_epoch,a.epochs+1):
        model.train();total=0.;seen=0;max_grad=0.;epoch_start=time.monotonic()
        for (xb,) in dl:
            xb=xb.to(device);n=len(xb);t=torch.randint(0,a.diffusion_steps,(n,),device=device);eps=torch.randn_like(xb)
            aa=root[t][:,None];bb=noise[t][:,None]
            loss=(model(aa*xb+bb*eps,(t.float()+.5)/a.diffusion_steps)-(aa*eps-bb*xb)).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip,error_if_nonfinite=True)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            opt.step()
            with torch.no_grad():
                torch._foreach_mul_(ep_params,a.ema);torch._foreach_add_(ep_params,raw_params,alpha=1.-a.ema)
            total+=float(loss)*n;seen+=n;max_grad=max(max_grad,float(gn))
        sched.step()
        record=dict(epoch=epoch,train_v_mse=total/seen,lr=opt.param_groups[0]['lr'],max_grad_norm=max_grad,
                    epoch_seconds=time.monotonic()-epoch_start,wall_seconds=time.monotonic()-start)
        is_val=epoch==1 or epoch%a.val_every==0 or epoch==a.epochs
        new_best=False
        if is_val:
            record['ema_val_v_mse']=validation(ema);record['raw_val_v_mse']=validation(model)
            if record['ema_val_v_mse']<best:
                best=record['ema_val_v_mse'];best_epoch=epoch;new_best=True
        history.append(record)
        with metrics.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        if new_best:torch_save(a.out/'best_ema_validation.pt',payload(epoch))
        if epoch==1 or epoch%a.save_every==0 or epoch==a.epochs:
            torch_save(a.out/f'epoch_{epoch:05d}_ema.pt',payload(epoch))
        if is_val or epoch%a.save_every==0:
            torch_save(resume,payload(epoch,full=True))
            json_write(a.out/'status.json',dict(best_epoch=best_epoch,best_ema_val=best,**record))
        if epoch==1 or epoch%25==0 or epoch==a.epochs:print('EPOCH',json.dumps(record),flush=True)
    torch_save(a.out/'final_ema.pt',payload(a.epochs))
    json_write(a.out/'completed.json',dict(completed=True,epochs=a.epochs,best_epoch=best_epoch,
        best_ema_validation=best,candidates=['best_ema_validation.pt','final_ema.pt']))

