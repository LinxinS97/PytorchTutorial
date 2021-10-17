import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from model import resnet18
from load_cifar10 import test_loader, BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def main():
    net = resnet18(num_class=10)
    net.load_state_dict(torch.load('./model/ResNet_params_epoch2.pth'))
    net.to(device)

    sum_correct = 0
    for j, data in enumerate(test_loader):
        net.eval()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()
        sum_correct += correct.item()

    test_correct = sum_correct * 1.0 / len(test_loader) / BATCH_SIZE
    print("test correct: ", test_correct)

    pass


if __name__ == '__main__':
    main()
