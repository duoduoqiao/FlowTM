import torch

from torch import nn
from Models.model_utils import InvLayer, ActNorm, Shuffle


class FlowNet(nn.Module):

    def __init__(self, block_num, size_1, size_2, shuffle=False):
        super(FlowNet, self).__init__()
        assert size_2 > size_1, 'Link size should be lower than hidden_size.'
        self.size_1, self.size_2 = size_1, size_2
        self.mapping = InvNet(block_num, size_1, size_2, shuffle)

    def forward(self, x, z=None, rev=False, cal_jacobian=False):
        if rev:
            if cal_jacobian:
                x, jacobian = self.mapping(x, z, rev, cal_jacobian)
            else:
                x = self.mapping(x, z, rev)
            if cal_jacobian:
                return x, jacobian
            else:
                return x
        else:
            if cal_jacobian:
                x, z, jacobian = self.mapping(x, rev, cal_jacobian)
            else:
                x, z = self.mapping(x)
            if cal_jacobian:
                return x, z, jacobian
            else:
                return x, z


class InvNet(nn.Module):
    
    def __init__(self, block_num, size_1, size_2, shuffle=False):
        super(InvNet, self).__init__()
        self.size_1, self.size_2 = size_1, size_2
        layers = []
        layer = ActNorm(size_2)
        layers.append(layer)
        
        for idx in range(block_num):
            layer = InvLayer(size_1, size_2-size_1)
            layers.append(layer)
            if shuffle:
                layer = Shuffle(size_2)
                layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, z=None, rev=False, cal_jacobian=False):
        jacobian = 0
        i=0
        if rev:
            x = torch.concat([x, z], dim=1)
            for layer in reversed(self.layers):
                if cal_jacobian:
                    x, logjacdet = layer.forward(x, rev, cal_jacobian)
                    jacobian += logjacdet
                else:
                    x = layer.forward(x, rev)
                
            if cal_jacobian:
                return x, jacobian
            else:
                return x
        else:
            for layer in self.layers:
                if cal_jacobian:
                    x, logjacdet = layer.forward(x, rev, cal_jacobian)
                    jacobian += logjacdet
                else:
                    x = layer.forward(x, rev)
                    
            x, z = x[:, :self.size_1], x[:, self.size_1:]
            if cal_jacobian:
                return x, z, jacobian
            else:
                return x, z
