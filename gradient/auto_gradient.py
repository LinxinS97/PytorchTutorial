import torch

# 叶子张量: 计算图最头端的节点，只有叶子张量才能计算梯度，一般叶子张量都是用户创建的
# grad: 该tensor的梯度值，每次在计算backward时都需要将前一时刻的梯度归零，否则梯度会一直累加
# grad_fn: 叶子节点通常为None，只有结果节点的grad_fn才有效，用于指示梯度函数时哪种类型

# backward函数
# torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False)
# # tensor: 用于计算梯度的tensor, torch.autograd.backward(z) == z.backward()
# # grad_tensors: 在计算矩阵梯度时会用到，它其实也是一个tensor, shape需要和前面的tensor保持一致

# torch.autograd包含的几个重要参数：
# torch.autograd.enable_grad: 启动梯度计算的上下文管理器
# torch.autograd.no_grad: 禁止梯度计算的上下文管理器
# torch.autograd.set_grad_enabled(model): 设置是否对模型进行梯度计算的上下文管理器

# x = Variable(torch.ones(2, 2), requires_grad=True)
x = torch.ones(2, 2, requires_grad=True)

x.register_hook(lambda grad: grad * 2)  # 钩子函数，可以在求导时对梯度进行操作，只有执行backward时会被调用

y = x + 2
z = y * y * 3
z.backward(torch.ones(2, 2))
print(y.grad)  # None
print(x.grad)  # d(out)/dx
print(y.grad_fn)
print(z.grad_fn)


# torch.autograd.Function
# 每一个原始的自动求导运算实际上实在两个Tensor上运行的函数：forward和backward
# forward函数：计算从输入Tensors获得的输出Tensors
# backward函数：接受输出Tensors对于某个标量值的梯度，并且计算输入Tensors对于该相同标量值的梯度
class line(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b):
        # y = w*x + b
        ctx.save_for_backward(w, x, b)
        return w*x + b

    @staticmethod
    def backward(ctx, grad_out):
        w, x, b = ctx.saved_tensors
        grad_w = grad_out * x
        grad_x = grad_out * w
        grad_b = grad_out

        return grad_w, grad_x, grad_b


w = torch.rand(2, 2, requires_grad=True)
x = torch.rand(2, 2, requires_grad=True)
b = torch.rand(2, 2, requires_grad=True)

out = line.apply(w, x, b)
out.backward(gradient=torch.ones(2, 2), retain_graph=True)
# out.backward(gradient=torch.ones(2, 2))
print(w, x, b)
print(w.grad, x.grad, b.grad)
