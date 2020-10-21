import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, pad):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.average_pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.average_pooling(out)
        out = self.leaky_relu(out)
        return out


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.block_1 = BasicBlock(3, 64, 1, 1)
        self.block_2 = BasicBlock(64, 128, 1, 1)
        self.block_3 = BasicBlock(128, 256, 1, 1)
        self.block_4 = BasicBlock(256, 512, 1, 1)
        self.block_5 = BasicBlock(512, 512, 1, 1)
        self.block_6 = BasicBlock(512, 1024, 1, 1)
        self.block_7 = BasicBlock(1024, 1024, 1, 1)
        self.fc_1 = nn.Linear(1024, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x




