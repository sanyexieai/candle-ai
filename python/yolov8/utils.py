import torch
import numpy as np
import torchvision

def make_anchors(xs0, xs1, xs2, strides=(8, 16, 32), grid_cell_offset=0.5):
    """生成锚点和步长张量"""
    anchor_points = []
    stride_tensor = []
    
    for xs, stride in zip([xs0, xs1, xs2], strides):
        _, _, h, w = xs.shape
        sx = torch.arange(w, device=xs.device, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(h, device=xs.device, dtype=torch.float32) + grid_cell_offset
        
        sx = sx.view(1, -1).repeat(h, 1).flatten()
        sy = sy.view(-1, 1).repeat(1, w).flatten()
        
        anchor_points.append(torch.stack([sx, sy], -1))
        stride_tensor.append(torch.full((h * w,), stride, device=xs.device, dtype=torch.float32))
    
    anchor_points = torch.cat(anchor_points, 0)
    stride_tensor = torch.cat(stride_tensor, 0).unsqueeze(1)
    
    return anchor_points.transpose(0, 1).unsqueeze(0), stride_tensor

def dist2bbox(distance, anchor_points):
    """将距离预测转换为边界框坐标"""
    lt, rb = torch.chunk(distance, 2, dim=1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    c_xy = (x1y1 + x2y2) * 0.5
    wh = x2y2 - x1y1
    return torch.cat([c_xy, wh], dim=1)

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    max_det=300,
    nm=0,
):
    """非极大值抑制
    Args:
        prediction: (batch_size, num_boxes, num_classes + 4)
        conf_thres: 置信度阈值
        iou_thres: NMS IOU阈值
        classes: 过滤特定类别
        max_det: 每张图片最大检测框数量
    Returns:
        list of detections, (x1, y1, x2, y2, conf, cls) for each detection
    """
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[-1] - 4  # number of classes
    xc = prediction[..., 4:].max(-1)[0] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [torch.zeros((0, 6), device=device)] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 4:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y 