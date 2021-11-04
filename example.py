import torch
from tiny_transformer import TinyTransformer

t = TinyTransformer(
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    num_features = 512, 
    num_heads = 2, 
    dim_feedforward = 1024, 
    dropout = 0.1, 
    device= 'cpu'
)

x = torch.randn(1, 22, 512)
y = torch.randn(1, 22, 512)

out = t(x, y)
print(out.shape) ## should be (1, 22, 512)

