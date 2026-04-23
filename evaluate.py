import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(

            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(256, embedding_dim)

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x, return_features=False):

        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        embedding = self.fc(x)

        if return_features:
            return F.normalize(embedding, dim=1)

        embedding = self.projector(embedding)
        embedding = F.normalize(embedding, dim=1)

        return embedding

# Projection head (NEW)
        embedding = self.projector(embedding)

# Normalize
        embedding = F.normalize(embedding, dim=1)

        return embedding