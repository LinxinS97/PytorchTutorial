import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self, num_class):
        super(MobileNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_dw2 = self.conv_dw(32, 32, stride=(1, 1))
        self.conv_dw3 = self.conv_dw(32, 64, stride=(2, 2))

        self.conv_dw4 = self.conv_dw(64, 64, stride=(1, 1))
        self.conv_dw5 = self.conv_dw(64, 128, stride=(2, 2))

        self.conv_dw6 = self.conv_dw(128, 128, stride=(1, 1))
        self.conv_dw7 = self.conv_dw(128, 256, stride=(2, 2))

        self.conv_dw8 = self.conv_dw(256, 256, stride=(1, 1))
        self.conv_dw9 = self.conv_dw(256, 512, stride=(2, 2))

        self.fc = nn.Linear(512, num_class)


    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            # DEEPWISE，group就是分别卷积，结果不汇总
            nn.Conv2d(in_channel, in_channel,
                      kernel_size=(3, 3), stride=stride, padding=(1, 1),
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel,
                      kernel_size=(1, 1), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out


def MobileNetV1():
    return MobileNet(10)
