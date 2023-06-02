"""
@author : Tien Nguyen
@date   : 2023-05-24
"""
import torch

def concat_tensors(
        tensors: list,
        device: str
    ) -> torch.Tensor:
    """
    @desc:
        + The model's outputs have 
                        the shape [num_class, batch_size, max_output_shape]
        + The model's outputs are Python tuples (tuples of tensors)
        + There are five classes in PAR2023, so the num_class == 5
        + The number of dimensions of the output's scores for each class
                                                        have difference shape
        + The two first tensors have the shape of [16, 12]
        + The three last have the shape of [16, 3] or [16, 2] 
        + Example:
            ([[0.0, 0.4, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.4, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]],
              [[0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.6, 0.0],
               [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.7, 0.0],
               [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0]],
              [[0.0, 0.6, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0]],
              [[0.0, 0.3, 0.0],
               [0.0, 0.3, 0.0],
               [0.0, 0.1, 0.0]],
              [[0.0, 0.4, 0.0],
               [0.0, 0.6, 0.0],
               [0.0, 0.3, 0.0]],)
    """
    if len(tensors[0].shape) == 1:
        return torch.stack(tensors)

    tensors_ = []
    max_dim = max([tensor.shape[-1] for tensor in tensors])
    new_shape = (tensors[0].shape[0], max_dim)
    for tensor in tensors:
        tensor_ = torch.zeros(new_shape, device=device)
        tensor_[:, : tensor.shape[-1]] = tensor
        tensors_.append(tensor_)
    return torch.stack(tensors_)
