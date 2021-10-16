import torch

# torch.cat(seq, dim=0, out=None)：按照已经存在的维度进行拼接
# torch.stack(seq, dim=0, out=None)：按照新的维度进行拼接
# torch.gather(input, dim, index, out=None)：在指定维度上按照索引赋值输出tensor

# torch.cat
a = torch.zeros((2, 4))
b = torch.ones((2, 4))
out = torch.cat((a, b), dim=1)
print(out)

# torch.stack
a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
print(a, b)
out = torch.stack((a, b), dim=0)
print(out)
print(out.shape) # 输出[2, 2, 3]，表示“两个2x3(2,2,3)”，dim等于几，就相当于在什么地方加一个新的维度


