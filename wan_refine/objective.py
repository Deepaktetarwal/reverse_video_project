"""Proposed endpoint-order hinge and normalized latent-motion reconstruction; not DPO."""
import torch

def objective(pred,zr,zf,noise,sigma,c):
    pred=pred.float();zr=zr.float();zf=zf.float();noise=noise.float()
    target=noise-zr
    flow=(pred-target).square().mean()
    xr=(1-sigma)*zr+sigma*noise
    x0=xr-sigma*pred
    reverse_error=(x0-zr).square().mean()
    forward_error=(x0-zf).square().mean()
    separation=(zr-zf).square().mean().detach()
    # Exact zr has gap +1; exact zf has gap -1 (outside the numerical floor).
    gap=(forward_error-reverse_error)/separation.clamp_min(c['order_floor'])
    active=float(c['order_sigma_min']<=sigma<=c['order_sigma_max'] and float(separation)>=c['min_pair_separation'])
    order=torch.relu(c['order_margin']*separation+reverse_error-forward_error)/separation.clamp_min(c['order_floor'])*active
    dr=torch.diff(zr,dim=2);dh=torch.diff(x0,dim=2)
    motion=(dh-dr).square().mean()/dr.square().mean().detach().clamp_min(c['motion_floor'])
    loss=flow+c['motion_weight']*motion+c['order_weight']*order
    return loss,{'flow':flow,'motion':motion,'order':order,'order_gap':gap,'order_active':flow.new_tensor(active),'reverse_x0_mse':reverse_error,'forward_x0_mse':forward_error}
