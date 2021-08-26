import torch

a = torch.rand(2, 2) * 10
print(a)
# we can use clamp(a, b) to limit the value of tensor(or gradient) from a to b
a = a.clamp(2, 5)
print(a)
