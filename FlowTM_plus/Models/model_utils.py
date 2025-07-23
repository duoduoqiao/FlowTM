import torch
from torch import nn
from Utils.function_utils import sum_except_batch


class InvLayer(nn.Module):

    def __init__(self, size_1, size_2, clamp=1., hidden=100):
        super(InvLayer, self).__init__()
        self.clamp = clamp
        self.size_1, self.size_2 = size_1, size_2

        self.weight_1 = nn.Sequential(
                            nn.Linear(self.size_2, hidden),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(hidden, self.size_1)
                        )
        self.weight_2 = nn.Sequential(
                            nn.Linear(self.size_1, hidden),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(hidden, self.size_2)
                        )
        self.weight_3 = nn.Sequential(
                            nn.Linear(self.size_1, hidden),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(hidden, self.size_2)
                        )

    def forward(self, x, rev=False, cal_jacobian=False):
        x1, x2 = (x.narrow(1, 0, self.size_1), x.narrow(1, self.size_1, self.size_2))
        if rev:
            y1 = x1 + self.weight_1(x2)
            self.s = self.clamp * (torch.sigmoid(self.weight_2(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.weight_3(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.weight_2(x1)) * 2 - 1)
            y2 = (x2 - self.weight_3(x1)).div(torch.exp(self.s))
            y1 = x1 - self.weight_1(y2)
        output = torch.cat((y1, y2), 1)

        if cal_jacobian:
            jacobian = self._jacobian(output.shape[0], rev)
            return output, jacobian
        else:
            return output
    
    def _jacobian(self, size, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)
        return jac / size


class Shuffle(nn.Module):

    def __init__(self, flow_size):
        super(Shuffle, self).__init__()
        self.flow_size = flow_size
        
        self.perm = torch.randperm(self.flow_size)
        self.perm_inv = torch.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(torch.LongTensor(self.perm_inv), requires_grad=False)

    def forward(self, x, rev=False, cal_jacobian=False):
        if rev:
            x = x[:, self.perm_inv]
        else:
            x = x[:, self.perm]

        if cal_jacobian:
            jacobian = self._jacobian(rev)
            return x, jacobian
        else:
            return x
    
    def _jacobian(self, rev=False):
        return 0.


class InvertibleSigmoid(nn.Module):

    def __init__(self):
        super(InvertibleSigmoid, self).__init__()
    
    def forward(self, x, rev=False, cal_jacobian=False):
        if not rev:
            output = self.rev_sigmoid(x)
        else:
            output = self.sigmoid(x)

        if cal_jacobian:
            _input = output if rev else x
            jacobian = self._jacobian(_input, rev)
            return output, jacobian
        else:
            return output
    
    @staticmethod
    def rev_sigmoid(x, eps=1e-6):
        return torch.special.logit(x, eps)
    
    @staticmethod
    def sigmoid(x):
        return torch.special.expit(x)
    
    def _jacobian(self, _input, rev=False):
        logJ = torch.log(1 / ((1 + torch.exp(_input)) * (1 + torch.exp(-_input))))
        logdet_J = logJ.sum()
        if not rev:
            return logdet_J
        else:
            return -logdet_J


class ActNorm(nn.Module):
    
    def __init__(self, dim):
        super(ActNorm, self).__init__()
        self.register_buffer("is_initialized", torch.tensor(False))
        self.log_scale = nn.Parameter(torch.empty(1, dim))
        self.loc = nn.Parameter(torch.empty(1, dim))

    @property
    def scale(self):
        return torch.exp(self.log_scale)
    
    def _jacobian(self, rev=False):
        if not rev:
            logdet_J = -sum_except_batch(self.log_scale)[0]
        else:
            logdet_J = sum_except_batch(self.log_scale)[0]
        return logdet_J

    def initialize(self, batch):
        self.is_initialized.data = torch.tensor(True)
        std = torch.std(batch, dim=0, keepdim=True)
        std = torch.where(std>0, std, 1.)
        self.log_scale.data = torch.log(std)
        self.loc.data = torch.mean(batch, dim=0, keepdim=True)
    
    def forward(self, x, rev=False, cal_jacobian=False):
        if not self.is_initialized:
            self.initialize(x)
        if rev:
            x = self.scale * x + self.loc
        else:
            x = (x - self.loc) / self.scale

        if cal_jacobian:
            jacobian = self._jacobian(rev)
            return x, jacobian
        else:
            return x
