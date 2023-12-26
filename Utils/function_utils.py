import torch
import numpy as np

from typing import Callable


def expectation_maximization(flows_loads, links_loads, rm, num_epoch: int):
    b, device = flows_loads.shape[0], flows_loads.device
    flows_loads, links_loads = flows_loads.unsqueeze(1), links_loads.unsqueeze(1)
    flows_loads_final = torch.zeros_like(flows_loads, device=device)

    for i in range(b):
        flows_loads_i = em_iteration(flows_loads[i], links_loads[i], rm, num_epoch)
        flows_loads_final[i] = flows_loads_i

    return flows_loads_final.squeeze(1)

def em_iteration(x, y, rm, num_epoch):
    device, length = x.device, x.shape[0]
    idxes = torch.arange(0, length).to(device)
    rm = rm.to(device)
    rm, x_final = rm.to(device), x.clone()
    x_known = x.clone()
    loss_min = torch.empty(length,).to(device)
    loss_min[:] = np.Inf

    for _ in range(num_epoch):
        a = x / rm.sum(dim=1)
        b = rm / (x @ rm).clamp_min_(1e-6).unsqueeze(1)

        c = y.unsqueeze(1) @ b.transpose(1, 2)
        x = torch.mul(a, c.squeeze(1))

        loss = torch.abs(x @ rm - y)
        loss = loss.sum(dim=1) + torch.abs(x - x_known).sum(dim=1)

        select = (loss < loss_min).reshape(loss.shape)
        idx = idxes[select]

        if len(idx) != 0:
            loss_min[idx] = loss[idx]
            x_final[idx, :] = x[idx, :]

    return x_final

def cycle(dl):
    while True:
        for data in dl:
            yield data

def f_except(f: Callable, x: torch.Tensor, *dim, **kwargs):
    """ Apply f on all dimensions except those specified in dim """
    result = x
    dimensions = [d for d in range(x.dim()) if d not in dim]

    if not dimensions:
        raise ValueError(f"Cannot exclude dims {dim} from x with shape {x.shape}: No dimensions left.")

    return f(result, dim=dimensions, **kwargs)

def sum_except(x: torch.Tensor, *dim):
    """ Sum all dimensions of x except the ones specified in dim """
    return f_except(torch.sum, x, *dim)

def sum_except_batch(x):
    """ Sum all dimensions of x except the first (batch) dimension """
    return sum_except(x, 0)