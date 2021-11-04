import torch
import torch.nn as nn
import torch.nn.functional as f

from .mask import generate_square_subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor , mask = None):
        """Super simple implementation of the attention mechanism. 
        Q, K, and V are batches of matrices.

            attention =  softmax(q.dot(k.T) / q.size(-1) ** 0.5)
            if mask is not none: attention *= mask
            output = attention.dot(v)

        q, k, v are all of shape: each with shape (batch_size, seq_length, num_features)

        example usage:

        ```
        q = torch.randn(1, 20, 100)
        k = torch.randn(1, 20, 100)
        v = torch.randn(1, 20, 100)
        m = torch.randn(1, 20, 20)  ## not a real mask, but has valid shape

        y = ScaledDotProductAttention()(
            query = q,
            key = k,
            value = v,
            mask = m
        )

        print(y.shape) ## should be (1, 20, 100)
        ```

        Returns:
            torch.Tensor or shape: (batch_size, seq_length, num_features)
        """

        ## we use bmm instead of torch.mm because we're gonna expect batches
        q_dot_k = query.bmm(key.transpose(1, 2))
        
        if mask is not None:
            assert(mask.shape == q_dot_k.shape), f'Invalid mask shape {mask.shape} for q_dot_k of shape {q_dot_k.shape}'
            q_dot_k = q_dot_k * mask

        if self.scale is None:
            self.scale = query.size(-1) ** 0.5

        attention = f.softmax(q_dot_k / self.scale, dim=-1)
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
    def __init__(self, num_features: int, seq_length: int, out_num_features: int, masked: bool = False):
        
        super().__init__()
        self.to_query = nn.Linear(num_features, seq_length)
        self.to_key = nn.Linear(num_features, seq_length)
        self.to_value = nn.Linear(num_features, out_num_features)

        self.core_attention_mechanism = ScaledDotProductAttention()

        if masked == True:
            self.mask = generate_square_subsequent_mask(size = seq_length).unsqueeze(0)
        else:
            self.mask = None

    def forward(self, query, key, value):
        """
        Note: query, key, and value are generally the same input vector, 
        except for when you're feeding the output of the transformer's encoder into the decoder.

        In that case:
        - query = output of encoder
        - key = output of encoder
        - value = output of masked attention layer in decoder

        Returns:
            torch.Tensor
        """

        return self.core_attention_mechanism(
            query = self.to_query(query), 
            key = self.to_key(key), 
            value = self.to_value(value),
            mask = self.mask
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, num_features: int, seq_length: int, out_num_features: int, masked: bool = False):
        """
        each head can attend to different parts of the input sequence, independent of the others. 
        Increasing the number of attention heads allows us to “pay attention” to 
        more parts of the sequence at once, which makes the model more powerful.

        Args:
            num_heads (int): [description]
            num_features (int): [description]
            seq_length (int): [description]
            out_num_features (int): [description]
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(num_features, seq_length, out_num_features, masked = masked) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * out_num_features, num_features)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """forward pass on a multi attention head

                   x
                  /|\      
            head_1...head_n
                   |
         (concat along dim: -1)
                   |
             (linear layer)
                   |
                (output)

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            torch.Tensor
        """
        list_of_attention_outputs = [h(query = query, key = key, value = value) for h in self.heads]
        concatenated_outs = torch.cat(list_of_attention_outputs, dim=-1)
        return self.linear(concatenated_outs)
