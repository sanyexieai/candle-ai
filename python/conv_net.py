import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 第一层卷积块
        # 输入通道数为1（灰度图），输出通道数为4，卷积核大小为3x3，padding为1保持特征图大小不变
        self.conv1 = nn.Conv2d(1, 4, 3, padding=(1, 1))
        # 批归一化层，对4个通道的特征图进行归一化
        self.bn1 = nn.BatchNorm2d(4)
        # ReLU激活函数，增加非线性表达能力
        self.relu1 = nn.ReLU()
        # 最大池化层，2x2的池化窗口，步长为2，将特征图尺寸减半
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二层卷积块
        # 输入通道数为4，输出通道数为8，卷积核大小为3x3，padding为1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=(1, 1))
        # 批归一化层，对8个通道的特征图进行归一化
        self.bn2 = nn.BatchNorm2d(8)
        # ReLU激活函数
        self.relu2 = nn.ReLU()
        # 最大池化层，2x2的池化窗口，步长为2
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三层卷积块
        # 输入通道数为8，输出通道数为16，卷积核大小为3x3，padding为1
        self.conv3 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        # 批归一化层，对16个通道的特征图进行归一化
        self.bn3 = nn.BatchNorm2d(16)
        # ReLU激活函数
        self.relu3 = nn.ReLU()
        # 最大池化层，2x2的池化窗口，步长为2
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 输入维度为16*3*3（16个通道，每个通道3x3的特征图），输出维度为10（对应10个类别）
        self.fc = nn.Linear(16*3*3, 10)

    def forward(self, x):
        # 第一层卷积块的前向传播
        x = self.conv1(x)        # 应用卷积层1，提取低级特征
        x = self.pool1(x)        # 应用池化层1，降低特征图尺寸
        x = self.bn1(x)          # 应用批归一化1，加速训练并提高模型稳定性
        x = self.relu1(x)        # 应用ReLU激活函数1，增加非线性

        # 第二层卷积块的前向传播
        x = self.conv2(x)        # 应用卷积层2，提取中级特征
        x = self.pool2(x)        # 应用池化层2，进一步降低特征图尺寸
        x = self.bn2(x)          # 应用批归一化2
        x = self.relu2(x)        # 应用ReLU激活函数2

        # 第三层卷积块的前向传播
        x = self.conv3(x)        # 应用卷积层3，提取高级特征
        x = self.pool3(x)        # 应用池化层3，最终降低特征图尺寸
        x = self.bn3(x)          # 应用批归一化3
        x = self.relu3(x)        # 应用ReLU激活函数3

        # 全连接层的前向传播
        x = x.view(-1, 16*3*3)  # 将特征图展平成一维向量
        x = self.fc(x)          # 应用全连接层，输出最终的分类结果
        return x