import torch

# mean, sum, prod, max, min, argmax(return index of max value),
# argmin(return index of min value value) et, al.

# std, var, median mode, histc, bincount for statistic value

a = torch.rand(2, 2)

print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
# the product of each value in dimension 0
print(torch.prod(a, dim=0))

print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=0))

print(torch.std(a))
print(torch.var(a))
print(torch.median(a))
print(torch.mode(a))

# histogram
a = torch.rand(2, 2) * 10
print(a)
print(torch.histc(a, 6, 0, 0))

# bincount, which can be used to calculate the counts of specific class
a = torch.randint(0, 10, [10])
print(a)
print(torch.bincount(a))


