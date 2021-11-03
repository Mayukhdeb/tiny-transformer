import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, layer: nn.Module, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.dropout(self.layer(x)))