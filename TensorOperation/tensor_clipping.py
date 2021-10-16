import torch

a = torch.rand(2, 2) * 10
print(a)
# we can use clamp(a, b) to limit the value of tensor(or gradient) from a to b
a = a.clamp(2, 5)
print(a)

# torch.chunk: 按照某个维度平均分块（最后一个可能会小于平均值）
a = torch.rand((3, 4))
out = torch.chunk(a, 2, dim=1)
print(a)
print(out)

# torch.split：按照某个维度依照第二个参数给出的list或者int进行tensor分割
a = torch.rand((10, 4))
print(a)
out = torch.split(a, [1, 3, 6], dim=0)
print(out)
