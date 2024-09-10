import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 定义每一层的卷积层并赋予名称
        self.conv1 = nn.Conv2d(1, 4, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(4, 8, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Linear(16*3*3, 10)

    def forward(self, x):
        x = self.conv1(x)        # 应用卷积层 1
        x = self.pool1(x)        # 应用池化层 1
        x = self.bn1(x)          # 应用批归一化层 1
        x = self.relu1(x)        # 应用激活函数 ReLU 1

        x = self.conv2(x)        # 应用卷积层 2
        x = self.pool2(x)        # 应用池化层 2
        x = self.bn2(x)          # 应用批归一化层 2
        x = self.relu2(x)        # 应用激活函数 ReLU 2

        x = self.conv3(x)        # 应用卷积层 3
        x = self.pool3(x)        # 应用池化层 3
        x = self.bn3(x)          # 应用批归一化层 3
        x = self.relu3(x)        # 应用激活函数 ReLU 3

        x = x.view(-1, 16*3*3)  # 展平
        x = self.fc(x)          # 应用全连接层
        return x