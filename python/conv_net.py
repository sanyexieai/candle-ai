import torch
import torch.nn as nn
# 定义简单的神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
        #batch*1*28*28
        nn.Conv2d(1, 4, 3, padding=(1, 1)),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        #batch*4*14*14
        nn.Conv2d(4, 8, 3, padding=(1, 1)),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        #batch*8*7*7
        nn.Conv2d(8, 16, 3, padding=(1, 1)),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        #batch*16*3*3
        )
        self.fc = nn.Linear(16*3*3, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16*3*3)
        x = self.fc(x)
        return x