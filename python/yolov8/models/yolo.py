import torch
import torch.nn as nn
import yaml
import math
from .common import Conv, C3, SPPF, Detect, Concat

class Model(nn.Module):
    def __init__(self, cfg='yolov8.yaml', ch=3, nc=None):
        """YOLOv8模型
        
        Args:
            cfg: 模型配置文件路径
            ch: 输入通道数
            nc: 类别数量
        """
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 模型字典
        else:
            import yaml  # for torch hub
            self.yaml_file = cfg  # 配置文件路径
            with open(cfg, encoding='utf-8') as f:
                self.yaml = yaml.safe_load(f)  # 模型字典
                
        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 输入通道
        if nc and nc != self.yaml['nc']:
            print(f"重写配置文件中的类别数量 nc={self.yaml['nc']} 为 nc={nc}")
            self.yaml['nc'] = nc  # 重写类别数量
            
        self.model, self.save = parse_model(self.yaml, ch=[ch])  # 模型, 保存点
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 默认名称
        self.inplace = self.yaml.get('inplace', True)
        
        # 设置超参数
        self.hyp = {
            'box': 0.05,  # 框损失权重
            'cls': 0.5,   # 类别损失权重
            'cls_pw': 1.0,  # 类别BCELoss正样本权重
            'obj': 1.0,   # 目标损失权重
            'obj_pw': 1.0,  # 目标BCELoss正样本权重
            'fl_gamma': 0.0,  # focal loss gamma
            'label_smoothing': 0.0  # 标签平滑
        }
        
        # 构建步幅
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self._initialize_biases()  # 仅在检测层运行
            
        # 初始化权重
        initialize_weights(self)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        return self._forward_once(x)
        
    def _forward_once(self, x):
        """单次前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        y = []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不是根节点
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从之前的层
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
        return x
        
    def _initialize_biases(self, cf=None):
        """初始化检测层的偏置
        
        Args:
            cf: 类别频率
        """
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            
def parse_model(d, ch):
    """解析模型配置
    
    Args:
        d: 模型字典
        ch: 输入通道列表
        
    Returns:
        模型列表和保存点
    """
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 每个位置的锚框数量
    no = na * (nc + 5)  # 每个锚框的输出数量
    
    layers, save, c2 = [], [], ch[-1]  # 层，保存，输出通道
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
                
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, C3, SPPF]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是输出
                c2 = make_divisible(c2 * gw, 8)
                
            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # 插入number参数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
            
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
    
def make_divisible(x, divisor):
    """使数字可被除数整除
    
    Args:
        x: 输入数字
        divisor: 除数
        
    Returns:
        可被整除的数字
    """
    return math.ceil(x / divisor) * divisor
    
def initialize_weights(model):
    """初始化模型权重
    
    Args:
        model: 模型
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 初始化权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 初始化偏置
            m.weight.requires_grad = True  # 设置需要梯度
            if m.bias is not None:
                m.bias.requires_grad = True
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
            m.weight.requires_grad = True  # 设置需要梯度
            m.bias.requires_grad = True
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True 