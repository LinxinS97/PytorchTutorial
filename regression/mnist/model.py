import torch


# model：一个卷积层和一个线性层（用于分类）
# 一般我们会单独定义成一个类
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # 注意，max pooling层的stride的默认值为kernel_size，也就是2
        )

        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)  # 线性层输入一定是一个向量，所以这里将输出修改为batch * n, n=27*27*32
        out = self.fc(out)
        return out