import torch
from model import YoloV8, Multiples
from utils import non_max_suppression
import cv2
import numpy as np

def load_image(image_path, size=640):
    """加载和预处理图像"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整图像大小
    h, w = img.shape[:2]
    r = size / max(h, w)
    if r != 1:
        img = cv2.resize(img, (int(w * r), int(h * r)))
    
    # 填充到正方形
    new_img = np.full((size, size, 3), 114, dtype=np.uint8)
    new_h, new_w = img.shape[:2]
    new_img[:new_h, :new_w] = img
    
    # 归一化和转换为张量
    img = new_img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    
    return img.unsqueeze(0)

def draw_boxes(image, boxes, scores, labels, class_names):
    """在图像上绘制检测框"""
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[label]
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签
        label = f'{class_name} {score:.2f}'
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), c2, (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 2), 0, 0.6, (0, 0, 0), thickness=1)
    
    return image

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = YoloV8(Multiples.s(), num_classes=80)  # 使用small版本
    # 这里需要加载预训练权重
    # model.load_state_dict(torch.load('yolov8s.pt'))
    model = model.to(device)
    model.eval()
    
    # COCO类别名称
    class_names = ['person', 'bicycle', 'car', ...]  # 需要完整的COCO类别列表
    
    # 加载图像
    image_path = 'test.jpg'  # 替换为实际的图像路径
    img = load_image(image_path)
    img = img.to(device)
    
    # 推理
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred)
    
    # 处理检测结果
    for i, det in enumerate(pred):
        if len(det):
            # 加载原始图像用于绘制
            orig_img = cv2.imread(image_path)
            
            # 转换预测结果为numpy数组
            boxes = det[:, :4].cpu().numpy()
            scores = det[:, 4].cpu().numpy()
            labels = det[:, 5].cpu().numpy().astype(int)
            
            # 绘制检测框
            result_img = draw_boxes(orig_img, boxes, scores, labels, class_names)
            
            # 保存结果
            cv2.imwrite('result.jpg', result_img)

if __name__ == '__main__':
    main() 