import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import cv2

# 导入自定义网络
from model import CNN


cnn = CNN()
cnn.load_state_dict(torch.load('./model/mnist_params.pkl'))
cnn.cuda()

# data
test_data = dataset.MNIST(root='mnist',  # 文件存放目录
                          train=False,  # 选择其中的训练集
                          transform=transforms.ToTensor(),  # 转换成tensor
                          download=False)  # 下载

# 调用DataLoader来划分batch
test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

# eval/test
loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)

    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    # 显示图片, batch_size * 1 * 28 * 28
    # images = images.cpu().numpy()
    # labels = labels.cpu().numpy()
    # pred = pred.cpu().numpy()
    #
    # for idx in range(images.shape[0]):
    #     im_data = images[idx]
    #     im_label = labels[idx]
    #     im_pred = pred[idx]
    #
    #     # 将channel这个维度放到最后面
    #     im_data = im_data.transpose(1, 2, 0)
    #
    #     print("label: {}, pred: {}".format(im_label, im_pred))
    #     cv2.imshow("imdata", im_data)
    #     cv2.waitKey(0)

accuracy = accuracy / len(test_data)
print("acc: {}".format(accuracy))
