import torch
from typing import Callable


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


def rbf_kernel(X, Y, sigma):
    dist = torch.cdist(X, Y, p=2)
    return torch.exp(-dist ** 2 / (2 * sigma ** 2))


def MMD(X, Y, sigma):
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    N_X = X.shape[0]
    N_Y = Y.shape[0]

    K = rbf_kernel(X, X, sigma)     # N_X * N_X
    L = rbf_kernel(Y, Y, sigma)     # N_Y * N_Y
    KL = rbf_kernel(X, Y, sigma)    # N_X * N_Y

    c_K = 1 / (N_X ** 2)
    c_L = 1 / (N_Y ** 2)
    c_KL = 2 / (N_X * N_Y)

    mmd_XY = c_K * torch.sum(K) + c_L * torch.sum(L) - c_KL * torch.sum(KL)
    return torch.sqrt(mmd_XY)