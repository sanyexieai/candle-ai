import torch
import torch.nn as nn
import warnings

def autopad(k, p=None):
    """自动填充
    
    Args:
        k: 卷积核大小
        p: 填充值
        
    Returns:
        填充值
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """标准卷积"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """初始化
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充值
            g: 分组数
            act: 是否使用激活函数
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """标准瓶颈结构"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """初始化
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组数
            e: 扩展比例
        """
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: 瓶颈层数量
            shortcut: 是否使用残差连接
            g: 分组数
            e: 扩展比例
        """
        super(C3, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # 连接后的通道数
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    """空间金字塔池化 - 快速版本"""
    def __init__(self, c1, c2, k=5):
        """初始化
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
        """
        super(SPPF, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2 * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    """连接层"""
    def __init__(self, dimension=1):
        """初始化
        
        Args:
            dimension: 连接维度
        """
        super(Concat, self).__init__()
        self.d = dimension
        
    def forward(self, x):
        return torch.cat(x, self.d)

class Detect(nn.Module):
    """检测头"""
    def __init__(self, nc=80, anchors=3, ch=()):
        """初始化
        
        Args:
            nc: 类别数量
            anchors: 每个尺度的锚框数量
            ch: 输入通道
        """
        super(Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 输出数量 (类别 + 框)
        self.nl = len(ch)  # 检测层数量
        self.na = anchors  # 每层的锚框数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        self.register_buffer('stride', torch.zeros(self.nl))  # 步长
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积
        
        # 初始化锚框
        self.register_buffer('anchors', torch.tensor([
            [[10, 13], [16, 30], [33, 23]],  # P3/8
            [[30, 61], [62, 45], [59, 119]],  # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ]).float())
        
    def forward(self, x):
        z = []  # 推理输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # 推理
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.stride[i]  # wh
                z.append(y.view(bs, -1, self.no))
                
        return x if self.training else (torch.cat(z, 1), x)
        
    @staticmethod
    def _make_grid(nx=20, ny=20):
        """生成网格
        
        Args:
            nx: x方向网格数量
            ny: y方向网格数量
            
        Returns:
            网格坐标
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float() 