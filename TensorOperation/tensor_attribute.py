# For every tensors, they have 3 attributes: torch.dtype, torch.device, torch.layout
# torch.device shows where torch.Tensor object stores in (CPU or CUDA0/1/2, which is GPU0/1/2)
# torch.layout shows torch.Tensor's layout in memory

import torch

# we can put all the tensors on CPU or GPU(CUDA) use device()
# put the tensor on CPU when pre-processing and on GPU when calculating and backward propagation
dev_cpu = torch.device('cpu')
dev_gpu = torch.device('cuda:0')
a = torch.tensor([2, 2], dtype=torch.float32, device=dev_gpu)
print(a)

# for sparse tensor, use torch.sparse_coo_tensor can reduce the cost of memory
# coo means the format of non-zero elements axis
# indices indicated the axis of non-zero elements
# values indicated the value of them
# a sparse matrix, for example:
# [[0 0 3 0 0],
#  [4 0 5 0 0],
#  [0 0 0 0 0]]
# the axis of non-zero elements is: (0, 2), (1, 0) and (1, 2)
# and the values of them is: 3, 4 and 5, so we can define indices and values as follows:
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32, device=dev_gpu)
x = torch.sparse_coo_tensor(indices, values, (4, 4))  # .to_dense() # switch to dense tensor
print(x)
