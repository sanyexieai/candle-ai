import torch
import torch.nn as nn

# 定义基本的残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        if self.shortcut:
            x += residual
        return x

# 定义整个网络
class DarkNet53(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet53, self).__init__()
        self.dark0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.dark1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            BasicBlock(64, 64, shortcut=True)
        )
        self.dark2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            BasicBlock(128, 128, shortcut=True),
            BasicBlock(128, 128, shortcut=True)
        )
        self.dark3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            *[BasicBlock(256, 256, shortcut=True) for _ in range(8)]
        )
        self.dark4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            *[BasicBlock(512, 512, shortcut=True) for _ in range(8)]
        )
        self.dark5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            *[BasicBlock(1024, 1024, shortcut=True) for _ in range(4)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out0 = self.dark0(x)
        out1 = self.dark1(out0)
        out2 = self.dark2(out1)
        out3 = self.dark3(out2)
        out4 = self.dark4(out3)
        out5 = self.dark5(out4)
        return out3, out4, out5

# 测试部分
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # 创建模型实例
    darknet = DarkNet53(num_classes=10).to(device)

    # 测试随机输入
    with torch.no_grad():
        darknet.eval()
        data = torch.rand(1, 3, 416, 416).to(device)
        try:
            out3, out4, out5 = darknet(data)
            print(f"out3 shape: {out3.shape}")  # 打印 out3 的形状
            print(f"out4 shape: {out4.shape}")  # 打印 out4 的形状
            print(f"out5 shape: {out5.shape}")  # 打印 out5 的形状
        except Exception as e:
            print(f"Error during forward pass: {e}")

    # 打印模型结构
    # print(darknet)
