# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2020/10/22 19:12
# -*- python3.6
import torch
import torch.nn as nn
from torchfm.layer import FeaturesLinear

class LogisticRegressionModel(nn.Module):
    """
    实现逻辑回归
    """
    def __init__(self, field_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)


    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: array
        """
        return torch.sigmoid(self.linear(x).squeeze(1))
