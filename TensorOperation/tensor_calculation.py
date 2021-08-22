import torch
# tensors 4-way calculation: add subtraction multiple and divide
# the method with "_" (for example, add_) means in place operation

# init
a = torch.rand(2, 3)
b = torch.rand(2, 3)

print('a=', a)
print('b=', b)

# add
print('a + b:')
print(a + b)
print(a.add(b))
print(torch.add(a, b))
print(a)
print(a.add_(b))
print(a)

# sub
print('a - b:')
print(a - b)
print(torch.sub(a, b))
print(a.sub(b))
print(a.sub_(b))
print(a)

# multiple
print('a * b:')
print(a*b)
print(torch.mul(a, b))
print(a.mul(b))
print(a.mul_(b))
print(a)

# div
print('a - b:')
print(a/b)
print(torch.div(a, b))
print(a.div(b))
print(a.div_(b))
print(a)

# matrix calculation
a = torch.ones(2, 1)
b = torch.ones(1, 2)

# matrix multiply
print('a dot b:')
print(a @ b)
print(a.matmul(b))
print(torch.matmul(a, b))
print(torch.mm(a, b))
print(a.mm(b))

# high dimensional matrix
# when calculating the high dimensional matrix, it will only take the last 2 dimension of tensor
# in the case blow, matmul will take (m, n, 3, 4) from a and (m, n, 4, 3) from b
# the first 2 dimension of a and b should be the same value
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a.matmul(b), a.matmul(b).shape)

# exponential calculation
print('a^3:')
a = torch.tensor([1, 2])
print(torch.pow(a, 3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))
print(a)

# exponential calculation based with e
print('e^a:')
a = torch.tensor([1, 2], dtype=torch.double)
print(torch.exp(a))
print(torch.exp_(a))
print(a)
print(a.exp())
print(a.exp_())

# logarithmic calculation based with e
print('log(a):')
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.log(a))
print(torch.log_(a))
print(a.log())
print(a.log_())

# sqrt
print('root(a):')
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.sqrt(a))
print(torch.sqrt_(a))
print(a.sqrt())
print(a.sqrt_())

# round and floor calculation
a = torch.rand(2, 2)
a = a * 10
print(a)

print(torch.floor(a))
print(torch.ceil(a))
print(torch.round(a))
# only integer part
print(torch.trunc(a))
# only decimal part
print(torch.frac(a))
int(a % 2)

