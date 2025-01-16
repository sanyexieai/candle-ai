import os
import glob
import re
import torch
import numpy as np

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """增量路径，如果路径已存在则添加后缀
    
    Args:
        path: 目标路径
        exist_ok: 是否允许路径已存在
        sep: 分隔符
        mkdir: 是否创建目录
        
    Returns:
        增量后的路径
    """
    path = str(path)  # os-agnostic
    if path == '' or exist_ok:  # 路径为空或允许存在
        return path
    pattern = re.compile(r'%s(\d+)' % sep) if sep else re.compile(r'(\d+)')  # 匹配数字
    
    # 检查路径是否存在
    dirs = glob.glob(f"{path}{sep}*")  # 类似路径
    matches = [pattern.search(d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # 提取数字
    n = max(i) + 1 if i else 2  # 增量数字
    path = f"{path}{sep}{n}"  # 更新路径
    if mkdir:
        os.makedirs(path, exist_ok=True)  # 创建目录
    return path

def xywh2xyxy(x):
    """将边界框从[x, y, w, h]转换为[x1, y1, x2, y2]格式
    
    Args:
        x: 边界框坐标 [x, y, w, h]
        
    Returns:
        边界框坐标 [x1, y1, x2, y2]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def box_iou(box1, box2):
    """计算两组边界框之间的IoU
    
    Args:
        box1: 第一组边界框 [N, 4]
        box2: 第二组边界框 [M, 4]
        
    Returns:
        IoU矩阵 [N, M]
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    # 计算交集
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
            torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    # 计算IoU
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300):
    """执行非极大值抑制
    
    Args:
        prediction: 模型预测结果 [batch, num_pred, num_classes + 5]
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        classes: 需要保留的类别
        max_det: 每张图片最多保留的检测框数量
        
    Returns:
        每张图片的检测结果列表 [num_det, 6], 6=[xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # 设置最大检测框数量
    max_det = min(max_det, 300)  # 限制最大检测框数量
    max_nms = 30000  # NMS前最大框数量限制
    time_limit = 10.0  # 每张图片的处理时间限制(秒)
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        
        # 如果没有框，继续下一张
        if not x.shape[0]:
            continue
            
        # 计算置信度
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # 转换为[x1, y1, x2, y2, conf, cls]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # 检查检测框数量
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            
        # 批量NMS
        c = x[:, 5:6] * max_det  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # 限制每张图片的检测框数量
            i = i[:max_det]
            
        output[xi] = x[i]
        
    return output 