import torch

# methods of tensors searching
# torch.where(condition, x, y): find tensors from x and y subjected to conditions and return a new tensor
# torch.gather(input, dim, index, out=None): return tensors which indices = index at dimension = dim
# torch.index_select(input, dim, index, out=None): return tensors in specific indices
# torch.masked_select(input, mask, out=None): return a VECTOR selected by mask
# torch.take(input, indices): look input as 1D tensor then output the values in indices
# torch.nonzero(input, out=None): output non-zero values' indices

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a, b)

# torch.where
out = torch.where(a > 0.5, a, b)
print(out)

# torch.index_select
# take the tensor at 0, 3, 2 from dimension 0
out = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print(out)

# torch.gather
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                           [0, 1, 2, 2],
                                           [0, 1, 3, 3]]))
print(out, out.shape)

# torch.mask
a = torch.linspace(1, 16, 16).view(4, 4)
mask = torch.gt(a, 8)
print(a, mask)
out = torch.masked_select(a, mask)
print(out)

# torch.take
a = torch.linspace(1, 16, 16).view(4, 4)
out = torch.take(a, index=torch.tensor([0, 15, 13, 10]))
print(out)

# torch.nonzero
a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
out = torch.nonzero(a)
print(out)

