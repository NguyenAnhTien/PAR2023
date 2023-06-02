"""
@author : Tien Nguyen
@date   : 2023-05-23
"""
import torch

import utils

def cal_acc(
        preds: torch.Tensor, 
        labels: tuple,
        device: str
    ) -> torch.Tensor:
    """
    @desc:
        - labels is a Python tuple which has 5 tensors coressponding to the
            5 classes of the PAR2023 dataset
        - each tensor has the shape of [BATCH SIZE]
        - transform the labels to a tensor which has the shape of [5, 16]
        - then, transpose the tensor to the shape of [16, 5]
        - compare the preds and labels tensors together
    """
    import ipdb
    ipdb.set_trace()
    labels = utils.concat_tensors(labels, device)
    preds = torch.transpose(preds, 1, 0)
    elementwise_comparison = torch.eq(labels, preds)
    matching_rows = torch.sum(elementwise_comparison, dim=1)
    accs = matching_rows / labels.shape[1]
    mA = torch.mean(accs)
    return mA
