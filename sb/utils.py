import torch
import numpy as np
import ot
import math

def compute_emd2(ref_data, pred_data, p=2):
    M = torch.cdist(pred_data, ref_data, p=p)
    a, b = ot.unif(pred_data.size()[0]), ot.unif(ref_data.size()[0])
    loss = ot.emd2(a, b, M.cpu().detach().numpy(),  numItermax=1000000)
    return loss

def marginal_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :]
        pred_traj = pred_traj[:, eval_idx, :]
        int_time = int_time[eval_idx]
    
    if pred_traj.ndim == 3:
        data_size, t_size, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            ref_dist = ref_traj[:, j]
            pred_dist = pred_traj[:, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            res[f't={int_time[j].item()}'] = { 'mean' : loss }
        
        return res

    elif pred_traj.ndim == 4:
        data_size, t_size, num_repeat, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            losses = []
            for i in range(num_repeat):
                ref_dist = ref_traj[:, j]
                pred_dist = pred_traj[:, j, i]
                M = torch.cdist(ref_dist, pred_dist, p=p)
                a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
                loss = ot.emd2(a, b, M.cpu().detach().numpy())
                losses.append(loss)
            
            res[f't={int_time[j].item()}'] = { 'mean' : np.mean(losses), 'std' : np.std(losses) }
        return res
    
def conditional_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :, :]
        pred_traj = pred_traj[:, eval_idx, :, :]
        int_time = int_time[eval_idx]

    data_size, t_size, num_repeat, dim = ref_traj.size()
    res = {}
    for j in range(1, t_size):
        losses = []
        for i in range(data_size):
            ref_dist = ref_traj[i, j]
            pred_dist = pred_traj[i, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            losses.append(loss)
        res[f't={int_time[j].item()}'] = sum(losses) / data_size
    return res

