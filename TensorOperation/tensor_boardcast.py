import torch

# broadcast means that 2 tensors with different shape can be operate by some function
# there are 2 conditions need to be met in broadcast:
# 1. each tensor has at least 1 dimension
# 2. right alignment
# for example, torch.rand(2, 1, 1) + torch.rand(3)
# First, pytorch will complete the dimension of torch.rand(3)'s shape from (3) to (1, 1, 3)
# Then, right alignment means that pytorch will compare the dimension of 2 tensors, in this case,
# (2, 1, 1) and
# (1, 1, 3)
# pytorch will compare any non-one dimension, and the non-one dimension must be the same
# which means that the third dimension of torch.rand(3) can be only 1 or 3

# a = torch.rand(2, 3)
# a = torch.rand(2, 2) # it will throw out an exception because 2 is not equals to 3
a = torch.rand(2, 1)

b = torch.rand(3)
c = a + b
# a, 2*3
# b, 1*3
# c, 2*3
print(a, b)
print(c)
print(c.shape)

