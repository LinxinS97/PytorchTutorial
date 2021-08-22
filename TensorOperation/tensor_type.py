import torch

a = torch.Tensor([[1, 2], [3, 4]])
print('Tensor with data: ', a)
print(a.type())

a = torch.Tensor(2, 3)
print('Tensor with shape: ', a)
print(a.type())

a = torch.ones(2, 2)
print('ones matrix: ', a)
print(a.type())

a = torch.eye(2, 2)
print('diag matrix:', a)
print(a.type())

a = torch.zeros(2, 2)
print('zero matrix', a)
print(a.type())

b = torch.Tensor(2, 3)
b = torch.zeros_like(b)
b = torch.ones_like(b)
print('zero matrix made from ones_like: ', b)
print(b.type())

'''random'''
a = torch.rand(2, 2)
print(a)
print(a.type())

# normal distribution
# mean with 5 different std, which could made 5 different normal distribution
# sampled from 5 different normal distribution as return value

# a = torch.normal(mean=0.0, std=torch.rand(5))
a = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(a)
print(a.type())

# uniform distribution
a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())

'''sequences'''
a = torch.arange(0, 10, 2)
print(a)
print(a.type())

# generate a n number sequence with same space
# a = torch.linspace(2, 10, 3)
a = torch.linspace(2, 10, 4)
print(a)
print(a.type())

# generate shuffle number sequence
a = torch.randperm(10)
print(a)
print(a.type())

###################################################
import numpy as np
a = np.array([[2, 3], [2, 3]])
print(a)

