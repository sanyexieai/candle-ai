import torch
import torch.nn as nn
import math
import numpy as np

def smooth_BCE(eps=0.1):
    """平滑的BCE损失
    
    Args:
        eps: 平滑因子
        
    Returns:
        正负标签值
    """
    return 1.0 - 0.5 * eps, 0.5 * eps

class ComputeLoss:
    def __init__(self, model):
        """初始化损失计算器
        
        Args:
            model: YOLOv8模型
        """
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # 获取设备
        h = model.hyp  # 超参数
        
        # 定义标准化函数
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        
        # 类别和目标权重
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 正负标签平滑
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            self.BCEcls, self.BCEobj = FocalLoss(self.BCEcls, g), FocalLoss(self.BCEobj, g)
            
        det = model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = 0  # stride 8 index
        self.gr = 1
        
        self.nc = det.nc  # 类别数量
        self.nl = det.nl  # 检测层数量
        self.na = det.na  # 每层的锚框数量
        self.anchors = det.anchors  # 锚框
        self.device = device
        self.hyp = h
        
    def __call__(self, p, targets):  # predictions, targets
        """计算损失
        
        Args:
            p: 预测结果
            targets: 目标框
            
        Returns:
            总损失和各部分损失
        """
        lcls = torch.zeros(1, device=self.device, requires_grad=True)  # 类别损失
        lbox = torch.zeros(1, device=self.device, requires_grad=True)  # 框损失
        lobj = torch.zeros(1, device=self.device, requires_grad=True)  # 目标损失
        
        # 计算损失
        tcls, tbox, indices, anch = self.build_targets(p, targets)  # 构建目标
        
        # 计算损失
        for i, pi in enumerate(p):  # 每个检测层
            if i >= len(indices):  # 如果索引超出范围，跳过
                continue
                
            b, a, gj, gi = indices[i]  # 图像，锚框，网格y，网格x
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # 目标掩码
            
            n = b.shape[0]  # 目标数量
            if n:
                ps = pi[b, a, gj, gi]  # 预测子集
                
                # 回归损失
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2
                pbox = torch.cat((pxy, pwh), 1)  # 预测框
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(预测框, 目标框)
                lbox = lbox + (1.0 - iou).mean()  # iou损失
                
                # 目标损失
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou比率
                
                # 分类损失
                if self.nc > 1:  # cls损失（仅当类别数 > 1时）
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)  # 目标
                    t[range(n), tcls[i]] = self.cp
                    lcls = lcls + self.BCEcls(ps[:, 5:], t)  # BCE
                    
            lobj = lobj + self.BCEobj(pi[..., 4], tobj)  # obj损失
            
        # 应用权重
        lbox = self.hyp['box'] * lbox
        lobj = self.hyp['obj'] * lobj
        lcls = self.hyp['cls'] * lcls
        
        # 总损失
        loss = lbox + lobj + lcls
        
        # 返回损失和损失项
        loss_items = torch.stack((lbox.detach(), lobj.detach(), lcls.detach()))
        return loss * 3, loss_items
        
    def build_targets(self, p, targets):
        """构建训练目标
        
        Args:
            p: 预测
            targets: 目标
            
        Returns:
            tcls: 类别目标
            tbox: 框目标
            indices: 索引
            anch: 锚框
        """
        na = 3  # 每个尺度的锚框数量
        nt = targets.shape[0]  # 目标数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # 归一化增益
        
        if nt:
            ai = torch.arange(na, dtype=torch.float, device=targets.device).view(na, 1).repeat(1, nt)  # 锚框索引
            targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 追加锚框索引
            
            g = 0.5  # 偏移
            off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # 偏移
                            
            for i in range(self.nl):
                anchors = self.anchors[i]
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy 增益
                
                # 匹配目标到锚框
                t = targets * gain
                
                # 计算比率
                r = t[:, :, 4:6] / anchors[:, None]  # wh 比率
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # 比较
                t = t[j]  # 过滤
                
                # 偏移
                gxy = t[:, 2:4]  # 网格xy
                gxi = gain[[2, 3]] - gxy  # 逆序网格xy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                
                # 定义
                b, c = t[:, :2].long().T  # 图像, 类别
                gxy = t[:, 2:4]  # 网格xy
                gwh = t[:, 4:6]  # 网格wh
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # 网格xy索引
                
                # 追加
                a = t[:, 6].long()  # 锚框索引
                indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 图像, 锚框, 网格索引
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # 框
                anch.append(anchors[a])  # 锚框
                tcls.append(c)  # 类别
        
        # 如果没有目标，为每个检测层添加空张量
        if not nt:
            for i in range(self.nl):
                bs = p[i].shape[0]  # 批次大小
                indices.append((torch.zeros(0, device=targets.device).long(),  # 图像索引
                              torch.zeros(0, device=targets.device).long(),  # 锚框索引
                              torch.zeros(0, device=targets.device).long(),  # 网格y索引
                              torch.zeros(0, device=targets.device).long()))  # 网格x索引
                tbox.append(torch.zeros(0, 4, device=targets.device))  # 框
                anch.append(torch.zeros(0, 2, device=targets.device))  # 锚框
                tcls.append(torch.zeros(0, device=targets.device).long())  # 类别
            
        return tcls, tbox, indices, anch
        
    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """计算IOU
        
        Args:
            box1: 框1
            box2: 框2
            x1y1x2y2: 是否为xyxy格式
            GIoU: 是否使用GIoU
            DIoU: 是否使用DIoU
            CIoU: 是否使用CIoU
            eps: 数值稳定性
            
        Returns:
            IOU值
        """
        # 转换框格式
        if not x1y1x2y2:
            box1 = self.xywh2xyxy(box1)
            box2 = self.xywh2xyxy(box2)
            
        # 计算交集和并集
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        
        # 交集
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
                
        # 并集
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 凸包宽度
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 凸包高度
            if CIoU or DIoU:  # 距离IOU
                c2 = cw ** 2 + ch ** 2 + eps  # 凸包对角线平方
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                       (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心点距离平方
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # 凸包面积
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU
        
    def xywh2xyxy(self, x):
        """将xywh格式转换为xyxy格式
        
        Args:
            x: xywh格式的框
            
        Returns:
            xyxy格式的框
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # 左上x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # 左上y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # 右下x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # 右下y
        return y
        
class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Focal Loss
        
        Args:
            loss_fcn: 基础损失函数
            gamma: focal loss gamma
            alpha: focal loss alpha
        """
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 需要应用focal loss
        
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 