import math
from torch import nn


class Inspector(nn.Module):
    def __init__(self, dataset):
        super(Inspector, self).__init__()
        if dataset == 'abilene':
            dim = 12
            self.conv_init = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
            self.conv_final = nn.Identity()
        elif dataset == 'geant':
            dim = 24
            self.conv_init = nn.Conv2d(1, 4, kernel_size=4, stride=1, padding=2)
            self.conv_final = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=2)
        else:
            dim = 12
            self.conv_init = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
            self.conv_final = nn.Identity()

        self.layers = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([8, int(dim/2), int(dim/2)]),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, int(dim/4), int(dim/4)]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, int(dim/4), int(dim/4)]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([8, int(dim/2), int(dim/2)]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([4, int(dim), int(dim)]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        assert x.dim() == 2, 'x.shape must be (B, N)'
        b, d = x.shape[0], int(math.sqrt(x.shape[-1]))
        x = x.reshape(b, 1, d, d)  # (batch_size, channels, height, width)
        x = self.conv_init(x)
        x = self.layers(x)
        x = self.conv_final(x)
        x = self.act(x)
        return x.reshape(b, d**2)


class Inspector3D(nn.Module):
    def __init__(self, dataset):
        super(Inspector3D, self).__init__()
        if dataset == 'abilene':
            dim = 12
            self.conv_init = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1)
            self.conv_final = nn.Identity()
        elif dataset == 'geant':
            dim = 24
            self.conv_init = nn.Conv3d(1, 4, kernel_size=(1, 4, 4), stride=1, padding=(0, 2, 2))
            self.conv_final = nn.ConvTranspose3d(1, 1, kernel_size=(1, 4, 4), stride=1, padding=(0, 2, 2))
        else:
            dim = 12
            self.conv_init = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1)
            self.conv_final = nn.Identity()
        window = 12

        self.layers = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([8, int(window/2), int(dim/2), int(dim/2)]),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, int(window/4), int(dim/4), int(dim/4)]),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, int(window/4), int(dim/4), int(dim/4)]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([8, int(window/2), int(dim/2), int(dim/2)]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([4, int(window), int(dim), int(dim)]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4, 1, kernel_size=3, stride=1, padding=1),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        assert x.dim() == 3, 'x.shape must be (B, W, N)'
        b, w, d = x.shape[0], x.shape[1], int(math.sqrt(x.shape[-1]))
        x = x.reshape(b, 1, w, d, d)  # (batch_size, channels, window_size, height, width)
        x = self.conv_init(x)
        x = self.layers(x)
        x = self.conv_final(x)
        x = self.act(x)
        return x.reshape(b*w, d**2)