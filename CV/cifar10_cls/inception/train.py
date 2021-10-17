import torch
import torch.nn as nn
from model import InceptionNetSmall
from load_cifar10 import train_loader, test_loader, BATCH_SIZE
import os
import tensorboardX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # hyper parameter
    epochs = 200
    lr = 0.01
    num_class = 10

    # MODEL
    net = InceptionNetSmall(num_class).to(device)

    # LOSS
    loss_func = nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    if not os.path.exists("log"):
        os.mkdir("log")
    writer = tensorboardX.SummaryWriter("log")

    step_n = 0

    # TRAIN
    for epoch in range(epochs):
        print("epoch: ", epoch, ", lr: ", optimizer.state_dict()["param_groups"][0]["lr"])
        net.train()  # 如果出现了batchnorm，dropout这种训练与测试不同情况的层，可以在这里将网络设置为train模式

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()  # 梯度置0
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # print("step:{}, loss:{}".format(i, loss.item()))

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()

            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train correct", 100.0 * correct.item() / BATCH_SIZE, global_step=step_n)

            # print("step: ", i, "loss: ", loss.item(), "mini-batch correct: ", 1.0 * correct / BATCH_SIZE)

        step_n += 1
        torch.save(net.state_dict(), "model/ResNet_params_epoch{}.pth".format(epoch + 1))
        scheduler.step()  # 更新学习率

        sum_loss = 0
        sum_correct = 0
        for j, data in enumerate(test_loader):
            net.eval()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            sum_loss += loss.item()
            sum_correct += correct.item()

            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train correct", 100.0 * correct.item() / BATCH_SIZE, global_step=step_n)

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 1.0 / len(test_loader) / BATCH_SIZE
        print("epoch: ", epoch + 1, "loss: ", test_loss, ", test correct: ", test_correct)


if __name__ == '__main__':
    main()
