from torch import nn


class AutoEncoder(nn.Module):
    
    def __init__(self, feature_dim, hidden_dim, dropout=0.):
        super(AutoEncoder, self).__init__()
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)