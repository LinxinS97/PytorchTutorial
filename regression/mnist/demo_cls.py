import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
# 导入自定义网络
from model import CNN

import model

# DATA
train_data = dataset.MNIST(root='mnist',  # 文件存放目录
                           train=True,  # 选择其中的训练集
                           transform=transforms.ToTensor(),  # 转换成tensor
                           download=True)  # 下载

test_data = dataset.MNIST(root='mnist',  # 文件存放目录
                          train=False,  # 选择其中的训练集
                          transform=transforms.ToTensor(),  # 转换成tensor
                          download=False)  # 下载

# 调用DataLoader来划分batch
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)


# MODEL：一个卷积层和一个线性层（用于分类）
# 从model.py中导入CNN
cnn = CNN()
cnn = cnn.cuda()

# LOSS
loss_func = torch.nn.CrossEntropyLoss()

# OPTIMIZER
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# TRAIN
for epoch in range(10):
    loss = None
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch: {}, training_loss: {}".format(epoch + 1, loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        # input: batchsize
        # outputs = batchsize * cls_num
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)
    print("epoch: {}, acc: {}, test_loss: {}".format(epoch + 1, accuracy, loss_test))

# SAVE
torch.save(cnn.state_dict(), "model/mnist_params.pkl")
torch.save(cnn, "model/mnist_model.pkl")
