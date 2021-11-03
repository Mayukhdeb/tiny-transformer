import torch
import torch.nn as nn

from .residual import Residual
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardLayer

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        num_features: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()

        assert (num_features % num_heads == 0), f'num_features: ({num_features}) should be divisible by num_heads ({num_heads})'
        num_features_per_head = out_num_features = num_features // num_heads
       
        multi_head_attention_layer = MultiHeadAttention(
                                        num_heads = num_heads, 
                                        num_features = num_features, 
                                        seq_length = num_features_per_head, 
                                        out_num_features = out_num_features
                                    )
        self.attention = Residual(
            multi_head_attention_layer,
            num_features=num_features,
            dropout=dropout,
        )

        self.feed_forward = Residual(
            FeedForwardLayer(dim_input= num_features, dim_feedforward = dim_feedforward),
            num_features=num_features,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        x = self.attention(x,x,x)
        y = self.feed_forward(x)
        return y