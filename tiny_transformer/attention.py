import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import generate_square_subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            assert (attn.shape[-2:] == mask.shape[-2:])
            attn = attn * mask

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_features, seq_length, out_num_features, dropout=0.1, masked = False):
        super().__init__()

        self.num_heads = num_heads
        self.seq_length = seq_length
        self.out_num_features = out_num_features


        # note that there are no biases
        self.w_qs = nn.Linear(num_features, num_heads * seq_length, bias=False)
        self.w_ks = nn.Linear(num_features, num_heads * seq_length, bias=False)
        self.w_vs = nn.Linear(num_features, num_heads * out_num_features, bias=False)
        self.fc = nn.Linear(num_heads * out_num_features, num_features, bias=False)

        self.attention = ScaledDotProductAttention(temperature = seq_length ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features, eps=1e-6)
        self.masked = masked

    def forward(self, q, k, v):

        seq_length, out_num_features, num_heads = self.seq_length, self.out_num_features, self.num_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(batch_size, len_q, num_heads, seq_length)
        k = self.w_ks(k).view(batch_size, len_k, num_heads, seq_length)
        v = self.w_vs(v).view(batch_size, len_v, num_heads, out_num_features)

        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.masked == True:
            mask= generate_square_subsequent_mask(size = len_k)
        else:
            mask = None
    
        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q

