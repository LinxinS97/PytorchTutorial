import torch
import numpy as np
import cv2

# torch.saves(state, dir) 保存模型
# torch.load(dir) 加载模型
# torch.get_num_threads() 获得用于并行化的CPU操作的OpenMP线程数
# torch.set_num_threads(int) 设定用于并行化cpu操作的OpenMP线程数
# torch.from_numpy(ndarry) 根据numpy创建tensor
# a.numpy() 直接获取tensor的numpy类型

# if torch.cuda.is_available():
#     device = torch.device("cuda") GPU
#     y = torch.ones_like(x, device=device) 直接创建一个GPU上的Tensor
#     x = x.to(device) 等价于.to("cuda")
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double)) # to还可以更改数据类型

# 使用opencv读取一张图片，读取出来的数据类型是numpy类型
im_data = cv2.imread("../../../mo.png")
# 显示图片：记得一定要配合waitKey
cv2.imshow('test1', im_data)
cv2.waitKey(0)
print(im_data)

# 从numpy初始化tensor，并使用to传入gpu中进行运算
out = torch.from_numpy(im_data)
out = out.to(torch.device('cuda'))
print(out.is_cuda)

# 在gpu中进行翻转运算后传入cpu并打印（如果不传入cpu则无法读取）
out = torch.flip(out, dims=[0])
out = out.to(torch.device('cpu'))
data = out.numpy()
cv2.imshow('test2', data)
cv2.waitKey(0)
