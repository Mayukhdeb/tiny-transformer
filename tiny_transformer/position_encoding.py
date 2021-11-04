import torch 

def get_position_encoding(seq_len: int, num_features: int, device: str):
    """
    We have to provide positional information to the model,
    so that it knows about the relative position of data points in the input sequences.
    
    According to the authors:
    "We chose the sinusoidal version because it may allow the model to extrapolate to 
    sequence lengths longer than the ones encountered during training."

    According to Samuel: 
    "Relative position between words are more important than their absolute position. 
    This is where sinusoidal positional encodings help."

    example:
    ```
    pos_enc = position_encoding(seq_len = 20, num_features = 100, device = 'cpu')
    print(pos_enc.shape) ## shape: (1, 20, 100)
    ```

    Args:
        seq_len (int): length of sequence
        num_features (int): number of features per thingy in sequence
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.tensor
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(num_features, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // num_features)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

