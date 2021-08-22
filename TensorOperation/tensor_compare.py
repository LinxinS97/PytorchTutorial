import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)
print()

# compare each element
print(torch.eq(a, b))
# compare shape and element
print(torch.equal(a, b))
# return True if a greater then b
print(torch.ge(a, b))
# return True if a less equal to b
print(torch.le(a, b))
# return True if a less (and not equal) to b
print(torch.lt(a, b))
# return True if a not equal to b
print(torch.ne(a, b))

# sort
a = torch.tensor([1, 4, 4, 3, 5])
print(torch.sort(a, dim=0, descending=True))  # descending sort

# top-k
a = torch.tensor([[2, 4, 3, 1, 5],
                  [2, 3, 5, 1, 4]])
print(a.shape)
# return #k of top value at dimension 0
print(torch.topk(a, k=1, dim=0))
# return kth minimum value
print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=1))

#
a = torch.rand(2, 3)
print(a)
print(torch.isfinite(a))
print(torch.isfinite(a/0))
print(torch.isinf(a/0))
print(torch.isnan(a))

import numpy as np
a = torch.tensor([1, 2, np.nan])
print(torch.isnan(a))
