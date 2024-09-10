import torch
import torch.nn as nn
from generate import CHAR_NUMBER, SEED

class CodeNet(nn.Module):
    def __init__(self):
        super(CodeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=15360, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=CHAR_NUMBER*len(SEED))
        )
 
    def forward(self, x):
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.layer5(x)
        print(x.shape)
        return x
 