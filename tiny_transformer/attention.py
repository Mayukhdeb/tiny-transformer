import torch
import torch.nn as nn
import torch.nn.functional as f

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """Super simple implementation of the attention mechanism. 
    Q, K, and V are batches of matrices.

        attention =  softmax(q.dot(k.T) / q.size(-1) ** 0.5)
        output = attention.dot(v)

    q, k, v are all of shape: each with shape (batch_size, seq_length, num_features)

    example usage:

    ```
    q = torch.randn(1, 20, 100)
    k = torch.randn(1, 20, 100)
    v = torch.randn(1, 20, 100)

    y = scaled_dot_product_attention(
        query = q,
        key = k,
        value = v
    )

    print(y.shape) ## should be (1, 20, 100)
    ```

    Returns:
        torch.tensor or shape: (batch_size, seq_length, num_features)
    """

    ## we use bmm instead of torch.mm because we're gonna expect batches
    q_dot_k = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    attention = f.softmax(q_dot_k / scale, dim=-1)
    return attention.bmm(value)


class AttentionHead(nn.Module):
    """Takes an input x and runs the attention mechanism over it.
    It generates q,k and v from x with 3 different linear layers.

    Example:
    ```
    ah = AttentionHead(num_features = 100, seq_length = 20, out_num_features = 6)
    x = torch.randn(1, 20, 100)  ## batch size, seq length, num features
    print(ah(x).shape) ## torch.Size([1, 20, 6]) i.e batch_size, seq length, out_seq_length
    ```

    Args:
        num_features (int): number of input features
        seq_length (int): length of sequence
        out_num_features (int): number of features on output
    """
    def __init__(self, num_features: int, seq_length: int, out_num_features: int):
        
        super().__init__()
        self.to_query = nn.Linear(num_features, seq_length)
        self.to_key = nn.Linear(num_features, seq_length)
        self.to_value = nn.Linear(num_features, out_num_features)

    def forward(self, x: torch.Tensor):
        return scaled_dot_product_attention(
            query = self.to_query(x), 
            key = self.to_key(x), 
            value = self.to_value(x)
        )
