import torch
import torch.nn as nn
from .encoder_block import EncoderBlock
from .position_encoding import get_position_encoding
from .decoder_block import DecoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        num_features: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device="cuda",
    ):
        """
        Encodes the input embedding. Is made up of multiple or a single encoder block.

        Example:
        ```
        e = TransformerEncoder(
            num_features= 512,
            num_heads= 2,
            dim_feedforward = 1024,
            dropout= 0.1,
            device= 'cpu'
        )

        x = torch.randn(1, 10, 512)
        y = e(x)
        print(y.shape) ## should be (1, 10, 512)
        ```

        Args:
            num_layers (int, optional): Defaults to 6.
            num_features (int, optional): Defaults to 512.
            num_heads (int, optional): Defaults to 8.
            dim_feedforward (int, optional): Defaults to 2048.
            dropout (float, optional): Defaults to 0.1.
            device (str, optional): Defaults to 'cuda'.
        """
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_features, num_heads, dim_feedforward, dropout).to(
                    device=self.device
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        seq_len, num_features = x.size(1), x.size(2)
        x += get_position_encoding(
            seq_len=seq_len, num_features=num_features, device=self.device
        )
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        num_features: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    num_features=num_features,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                ).to(self.device)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(num_features, num_features).to(self.device)

    def forward(self, target, encoded=None):
        seq_len, num_features = target.size(1), target.size(2)
        target += get_position_encoding(
            seq_len=seq_len, num_features=num_features, device=self.device
        )
        for layer in self.layers:
            target = layer(target, encoded)

        return torch.softmax(self.linear(target), dim=-1)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_features: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            num_features=num_features,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=self.device,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            num_features=num_features,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=self.device,
        )

    def forward(self, src, tgt):
        return self.decoder(tgt, self.encoder(src))
