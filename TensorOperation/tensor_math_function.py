import torch

# trigonometric functions:
# abs, acos, asin, atan, atan2, cos, cosh, sin, sinh, tan, tanh
# *all these functions support in place method
a = torch.rand(2, 3)
print(a)
a = torch.cos(a)
print(a)

# other math functions like:
# abs erf erfinv, sigmoid, neg, reciprocal, rsqrt, sign, lerp, addcdiv
# addcmul, cumprod, cumsum et al.
# abs, sigmoid and sign is usually used

# norm
# 0 norm: count of zero values
# 1 norm: sum of absolute value
# 2 norm: squared sum of each value then root ==> torch.norm()
# p norm: p powered sum of each value then p root ==> torch.dist(input, other, p=...) or torch.norm(a, p=...)
# et, al.
a = torch.rand(2, 1)
b = torch.rand(2, 1)
print(a, b)
print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))
print(torch.norm(a))
print(torch.norm(a, p=0))
# Frobenius Norm
print(torch.norm(a, p='fro'))

# matrix decomposition
# LU, QR, EVD, SVD

a = torch.rand(4, 4)
print(torch.svd(a))