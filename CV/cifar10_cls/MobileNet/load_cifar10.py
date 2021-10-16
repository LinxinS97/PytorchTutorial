from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

BATCH_SIZE = 128

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

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")


train_transform = transforms.Compose([
    transforms.RandomResizedCrop((28, 28)),  # 随机裁剪，裁剪为28*28
    transforms.RandomHorizontalFlip(),  # 随机水平翻转（需要注意图片是否对翻转敏感）
    # transforms.RandomVerticalFlip(),  # 随机垂直反转
    # transforms.RandomRotation(90),  # -90~90°之间随机翻转
    # transforms.RandomGrayscale(0.1),  # 0.1的概率将图片转换成灰度图
    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),  # 亮度，对比度等和颜色相关的信息
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# 自定义dataset
class MyDataset(Dataset):
    # 完成对数据的读取和简单处理
    def __init__(self,
                 im_list,  # 图片list，每一个元素是一个路径
                 transform=None,  # 数据增强用的函数
                 loader=default_loader):  # 返回 Image对象
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            # linux
            # im_label_name = im_item.split('/')[-2]

            # windows
            im_label_name = im_item.split('\\')[-2]  # 获取到类别名称
            imgs.append([im_item, label_dict[im_label_name]])  # [图片矩阵，类别id]

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob('F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\TRAIN\*\*.png')
im_test_list = glob.glob('F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\TEST\*\*.png')

train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)

print("num_of_train: ", len(train_dataset))
print("num_of_test: ", len(test_dataset))
