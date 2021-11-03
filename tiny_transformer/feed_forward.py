import torch
import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, dim_input: int = 512, dim_feedforward: int = 2048):
        """Simple feed forward module containing 2 linear layers and a relu in between.

        As mentioned in the paper:
        'Each of the layers in outiny_r encoder and decoder contains a fully connected feed-forward network, 
        consists of two linear transformations with a ReLU activation in between."

        Args:
            dim_input (int, optional): Defaults to 512.
            dim_feedforward (int, optional): Defaults to 2048.
        """
        super().__init__()
        self.layers =  nn.Sequential(
            nn.Linear(dim_input, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_input),
        )

    def forward(self, x: torch.Tensor): 
        return self.layers(x)
