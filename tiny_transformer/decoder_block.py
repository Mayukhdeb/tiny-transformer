import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .residual import Residual
from .feed_forward import FeedForwardLayer


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_features: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert (
            num_features % num_heads == 0
        ), f"num_features: ({num_features}) should be divisible by num_heads ({num_heads})"
        num_features_per_head = out_num_features = num_features // num_heads

        multi_head_attention_layer_1 = MultiHeadAttention(
            num_heads=num_heads,
            num_features=num_features,
            seq_length=num_features_per_head,
            out_num_features=out_num_features,
            masked=True,
        )
        multi_head_attention_layer_2 = MultiHeadAttention(
            num_heads=num_heads,
            num_features=num_features,
            seq_length=num_features_per_head,
            out_num_features=out_num_features,
            masked=False,
        )
        self.attention_1 = Residual(
            multi_head_attention_layer_1,
            num_features=num_features,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            multi_head_attention_layer_2,
            num_features=num_features,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            FeedForwardLayer(dim_input=num_features, dim_feedforward=dim_feedforward),
            num_features=num_features,
            dropout=dropout,
        )

    def forward(self, target: torch.Tensor, encoded: torch.Tensor = None):
        y = self.attention_1(target, target, target)

        if encoded != None:
            y = self.attention_2(encoded, encoded, y)
        return self.feed_forward(y)
