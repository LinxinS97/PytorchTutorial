# torch中有很多预定义的网络结构，可以直接用torch进行调用
# 这里以resnet18为例

import torch
import torch.nn as nn
from torchvision import models


class resnet18(nn.Module):
    def __init__(self, num_class: int):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)  # 定义resnet18，使用预训练的参数
        self.num_features = self.model.fc.in_features  # 定义model的feature
        self.model.fc = nn.Linear(self.num_features, num_class)  # 定义model的分类器

    def forward(self, x):
        out = self.model(x)
        return out


def torch_resnet18(num_class: int):
    return resnet18(num_class=num_class)
