import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_embedding: int, d_hidden: int):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.network = nn.Sequential(
            nn.Linear(d_embedding, d_hidden, False),
            nn.GELU(),
            nn.Linear(d_hidden, d_embedding, False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
