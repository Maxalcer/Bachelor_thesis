import torch
import torch.nn as nn

input = torch.randn(1, 1, 10, 10)
avgpool = nn.AdaptiveAvgPool2d(10)
print(input == avgpool(input))