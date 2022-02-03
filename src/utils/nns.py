from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def batchify(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
