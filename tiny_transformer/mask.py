import torch

def generate_square_subsequent_mask(size: int):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.

    Example:
    ```
    print(generate_square_subsequent_mask(size = 3))
    ```

    output:
    ```
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    ```
    """
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)