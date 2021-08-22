import torch

# we can define random seed by torch.manual_seed(seed),
# if we defined random seed, the number will be the same every time
# we can also define distribution that random number subjected to
torch.manual_seed(1)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)

print(torch.normal(mean, std))
