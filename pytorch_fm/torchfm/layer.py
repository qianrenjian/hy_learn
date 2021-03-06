# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2020/10/22 19:11
# -*- python3.6
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim, )))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
