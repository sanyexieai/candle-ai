import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiples:
    def __init__(self, depth, width, ratio):
        self.depth = depth
        self.width = width
        self.ratio = ratio
    
    @staticmethod
    def n():
        return Multiples(0.33, 0.25, 2.0)
    
    @staticmethod
    def s():
        return Multiples(0.33, 0.50, 2.0)
    
    @staticmethod
    def m():
        return Multiples(0.67, 0.75, 1.5)
    
    @staticmethod
    def l():
        return Multiples(1.00, 1.00, 1.0)
    
    @staticmethod
    def x():
        return Multiples(1.00, 1.25, 1.0)
    
    def filters(self):
        f1 = int(256 * self.width)
        f2 = int(512 * self.width)
        f3 = int(512 * self.width * self.ratio)
        return f1, f2, f3

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3)
    
    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = int(c2 * 1.0)  # channel factor
        self.cv1 = ConvBlock(c1, c_, 3)
        self.cv2 = ConvBlock(c_, c2, 3)
        self.residual = c1 == c2 and shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.residual else y

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ConvBlock(c1, 2 * self.c, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, 1)
        self.bottleneck = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBlock(c1, c_, 1)
        self.cv2 = ConvBlock(c_ * 4, c2, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(F.pad(x, [self.k//2]*4), self.k, stride=1)
        y2 = F.max_pool2d(F.pad(y1, [self.k//2]*4), self.k, stride=1)
        y3 = F.max_pool2d(F.pad(y2, [self.k//2]*4), self.k, stride=1)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class DarkNet(nn.Module):
    def __init__(self, m: Multiples):
        super().__init__()
        w, r, d = m.width, m.ratio, m.depth
        
        self.b1_0 = ConvBlock(3, int(64 * w), 3, 2, 1)
        self.b1_1 = ConvBlock(int(64 * w), int(128 * w), 3, 2, 1)
        
        self.b2_0 = C2f(int(128 * w), int(128 * w), round(3 * d), True)
        self.b2_1 = ConvBlock(int(128 * w), int(256 * w), 3, 2, 1)
        self.b2_2 = C2f(int(256 * w), int(256 * w), round(6 * d), True)
        
        self.b3_0 = ConvBlock(int(256 * w), int(512 * w), 3, 2, 1)
        self.b3_1 = C2f(int(512 * w), int(512 * w), round(6 * d), True)
        
        self.b4_0 = ConvBlock(int(512 * w), int(512 * w * r), 3, 2, 1)
        self.b4_1 = C2f(int(512 * w * r), int(512 * w * r), round(3 * d), True)
        
        self.b5 = SPPF(int(512 * w * r), int(512 * w * r), 5)

    def forward(self, x):
        x1 = self.b1_1(self.b1_0(x))
        x2 = self.b2_2(self.b2_1(self.b2_0(x1)))
        x3 = self.b3_1(self.b3_0(x2))
        x4 = self.b4_1(self.b4_0(x3))
        x5 = self.b5(x4)
        return x2, x3, x5

class YoloV8Neck(nn.Module):
    def __init__(self, m: Multiples):
        super().__init__()
        w, r, d = m.width, m.ratio, m.depth
        
        self.up = Upsample(2)
        n = round(3 * d)
        
        self.n1 = C2f(int(512 * w * (1 + r)), int(512 * w), n, False)
        self.n2 = C2f(int(768 * w), int(256 * w), n, False)
        self.n3 = ConvBlock(int(256 * w), int(256 * w), 3, 2, 1)
        self.n4 = C2f(int(768 * w), int(512 * w), n, False)
        self.n5 = ConvBlock(int(512 * w), int(512 * w), 3, 2, 1)
        self.n6 = C2f(int(512 * w * (1 + r)), int(512 * w * r), n, False)

    def forward(self, p3, p4, p5):
        x = self.n1(torch.cat([self.up(p5), p4], 1))
        head_1 = self.n2(torch.cat([self.up(x), p3], 1))
        head_2 = self.n4(torch.cat([self.n3(head_1), x], 1))
        head_3 = self.n6(torch.cat([self.n5(head_2), p5], 1))
        return head_1, head_2, head_3

class DetectionHead(nn.Module):
    def __init__(self, nc, filters):
        super().__init__()
        self.nc = nc  # number of classes
        self.ch = 16  # channels
        self.no = nc + self.ch * 4  # number of outputs
        
        c1 = max(filters[0], nc)
        c2 = max(filters[0] // 4, self.ch * 4)
        
        self.dfl = nn.Conv2d(self.ch, 1, 1)
        
        # Create cv2 and cv3 modules for each scale
        self.cv2 = nn.ModuleList([
            self._make_cv2(c2, self.ch, f) for f in filters
        ])
        self.cv3 = nn.ModuleList([
            self._make_cv3(c1, nc, f) for f in filters
        ])

    def _make_cv2(self, c2, ch, f):
        return nn.Sequential(
            ConvBlock(f, c2, 3, 1),
            ConvBlock(c2, c2, 3, 1),
            nn.Conv2d(c2, 4 * ch, 1)
        )

    def _make_cv3(self, c1, nc, f):
        return nn.Sequential(
            ConvBlock(f, c1, 3, 1),
            ConvBlock(c1, c1, 3, 1),
            nn.Conv2d(c1, nc, 1)
        )

    def forward(self, x0, x1, x2):
        shape = x0.shape  # BCHW
        
        # Process each scale
        def _forward_single(x, cv2, cv3):
            box = cv2(x)
            cls = cv3(x)
            
            # 重塑box预测
            box = box.view(shape[0], 4, self.ch, -1)  # (batch_size, 4, ch, h*w)
            box = box.permute(0, 2, 1, 3)  # (batch_size, ch, 4, h*w)
            box = self.dfl(box)  # (batch_size, 1, 4, h*w)
            box = box.squeeze(1)  # (batch_size, 4, h*w)
            box = box.permute(0, 2, 1)  # (batch_size, h*w, 4)
            
            # 重塑类别预测
            cls = cls.permute(0, 2, 3, 1).reshape(shape[0], -1, self.nc)  # (batch_size, h*w, nc)
            
            return box, cls
            
        box0, cls0 = _forward_single(x0, self.cv2[0], self.cv3[0])
        box1, cls1 = _forward_single(x1, self.cv2[1], self.cv3[1])
        box2, cls2 = _forward_single(x2, self.cv2[2], self.cv3[2])
        
        # 拼接不同尺度的预测
        box = torch.cat([box0, box1, box2], dim=1)  # (batch_size, total_anchors, 4)
        cls = torch.cat([cls0, cls1, cls2], dim=1)  # (batch_size, total_anchors, nc)
        
        # 应用sigmoid到类别预测
        cls = torch.sigmoid(cls)
        
        # 最终输出: (batch_size, total_anchors, 4+nc)
        return torch.cat([box, cls], dim=2)

class YoloV8(nn.Module):
    def __init__(self, m: Multiples, num_classes: int):
        super().__init__()
        self.net = DarkNet(m)
        self.fpn = YoloV8Neck(m)
        self.head = DetectionHead(num_classes, m.filters())

    def forward(self, x):
        x1, x2, x3 = self.net(x)
        x1, x2, x3 = self.fpn(x1, x2, x3)
        return self.head(x1, x2, x3) 