import torch

# torch.nn: 专门为神经网络设计的模块化接口
# nn构建与autograd之上，可以用来定义和运行神经网络
#   nn.Parameter: 用于定义参数
#   nn.Linear & nn.Conv2d: torch内置“模型”，包含了完整的forward和backward等nn.Module定义的方法
#   nn.functional: 定义了许多函数，例如conv2d等（需要注意的是这里的c是小写的，表示这只是一个函数，并不是模型）
#   nn.Module: torch的模型接口，定义了一个模型需要包含的所有方法，是所有其他模型的父类
#   nn.Sequential


