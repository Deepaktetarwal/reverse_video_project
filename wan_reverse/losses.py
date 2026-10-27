import torch

def interpolate(z,noise,sigma):
    return (1-sigma)*z+sigma*noise

def transport_target(teacher_forward,z_forward,z_reverse):
    # For xr(s)=xf(s)+(1-s)*(zr-zf), d xr/ds = d xf/ds + zf-zr.
    return teacher_forward.float()+z_forward.float()-z_reverse.float()

def losses(pred,target,kd,zr,xr,sigma,c):
    pred=pred.float(); target=target.float(); kd=kd.float()
    flow=(pred-target).square().mean()
    distill=(pred-kd).square().mean()
    x0=xr.float()-sigma*pred
    if x0.shape[2]>1:
        temporal=(torch.diff(x0,dim=2)-torch.diff(zr.float(),dim=2)).square().mean()
    else: temporal=flow*0
    total=c['flow_weight']*flow+c['kd_weight']*distill+c['temporal_weight']*temporal
    return total,{'flow':flow.detach(),'kd':distill.detach(),'temporal':temporal.detach()}
