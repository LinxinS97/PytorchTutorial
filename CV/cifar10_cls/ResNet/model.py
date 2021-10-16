import torch
import torch.nn as nn
import torch.nn.functional as F


# SKIP_GRAM
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=(1, 1)):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride[0] != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out


# MAIN
class ResNetModel(nn.Module):

    # 构建多个SKIP_GRAM层
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            # 只有第一层需要下采样，后续全部保持原样
            if i == 0:
                in_stride = stride
            else:
                in_stride = (1, 1)
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel

        return nn.Sequential(*layers_list)

    def __init__(self, res_block=ResBlock, num_class=10):
        super(ResNetModel, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(res_block, out_channel=64, stride=(1, 1), num_block=2)
        self.layer2 = self.make_layer(res_block, out_channel=128, stride=(2, 2), num_block=2)
        self.layer3 = self.make_layer(res_block, out_channel=256, stride=(2, 2), num_block=2)
        self.layer4 = self.make_layer(res_block, out_channel=512, stride=(2, 2), num_block=2)

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNet():
    return ResNetModel()
