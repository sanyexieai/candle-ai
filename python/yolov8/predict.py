import torch
import numpy as np
import cv2
import os
from pathlib import Path
from models.yolo import Model
from utils.general import non_max_suppression

class YOLOPredictor:
    def __init__(self, weights='weights/yolov8s.pt', cfg='models/yolov8.yaml', device='cuda'):
        """初始化YOLO预测器
        
        Args:
            weights: 权重文件路径
            cfg: 模型配置文件路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        # 检查权重文件是否存在
        if not os.path.exists(weights):
            raise FileNotFoundError(f"权重文件未找到: {weights}")
            
        # 检查CUDA是否可用
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            device = 'cpu'
            
        self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        try:
            # 创建模型
            self.model = Model(cfg, ch=3, nc=80).to(self.device)
            
            # 加载权重
            ckpt = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.model.eval()
            print("模型加载成功")
            
        except Exception as e:
            print(f"模型加载失败，详细错误:")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"模型加载失败: {str(e)}")
        
        # COCO类别名称
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                          'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                          'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                          'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                          'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                          'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
    def preprocess_image(self, img_path, size=640):
        """预处理图像
        
        Args:
            img_path: 图像路径或numpy数组
            size: 目标大小
            
        Returns:
            预处理后的图像张量
        """
        try:
            # 读取图像
            if isinstance(img_path, str):
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"图像文件未找到: {img_path}")
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"图像加载失败: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_path
                
            self.original_shape = img.shape[:2]  # 保存原始尺寸
            
            # 调整图像大小
            h, w = img.shape[:2]
            r = size / max(h, w)
            if r != 1:
                img = cv2.resize(img, (int(w * r), int(h * r)))
            
            # 填充到正方形
            new_img = np.full((size, size, 3), 114, dtype=np.uint8)
            new_h, new_w = img.shape[:2]
            new_img[:new_h, :new_w] = img
            
            # 转换为张量
            img = new_img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)  # 添加batch维度
            img = img.to(self.device)
            img = img / 255.0  # 归一化
            
            return img
            
        except Exception as e:
            raise RuntimeError(f"图像预处理失败: {str(e)}")
    
    def postprocess_boxes(self, boxes, scores, labels):
        """后处理检测框，将坐标转换回原始图像尺寸
        
        Args:
            boxes: 检测框坐标
            scores: 置信度分数
            labels: 类别标签
            
        Returns:
            调整后的检测框、分数和标签
        """
        if len(boxes) == 0:
            return boxes, scores, labels
            
        # 获取缩放比例
        orig_h, orig_w = self.original_shape
        scale = max(orig_h, orig_w) / 640
        
        # 调整坐标
        boxes = boxes.cpu().numpy()
        boxes = boxes * scale
        
        return boxes, scores, labels
    
    def draw_boxes(self, image, boxes, scores, labels):
        """在图像上绘制检测框
        
        Args:
            image: 原始图像
            boxes: 检测框坐标
            scores: 置信度分数
            labels: 类别标签
            
        Returns:
            绘制了检测框的图像
        """
        try:
            image = image.copy()
            for box, score, label_idx in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names[int(label_idx)]
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label_text = f'{label} {score:.2f}'
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x1, y1-text_h-baseline), (x1+text_w, y1), (0, 255, 0), -1)
                cv2.putText(image, label_text, (x1, y1-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
            return image
            
        except Exception as e:
            raise RuntimeError(f"绘制检测框失败: {str(e)}")
    
    @torch.no_grad()
    def predict(self, img_path, conf_thres=0.1, iou_thres=0.65):
        """执行预测
        
        Args:
            img_path: 图像路径或numpy数组
            conf_thres: 置信度阈值
            iou_thres: NMS IOU阈值
            
        Returns:
            boxes: 边界框坐标 [x1, y1, x2, y2]
            scores: 置信度分数
            labels: 类别标签
            processed_img: 处理后的图像
        """
        try:
            # 预处理图像
            img = self.preprocess_image(img_path)
            
            # 执行推理
            pred = self.model(img)
            
            # 执行NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # 处理结果
            if len(pred[0]) > 0:
                boxes = pred[0][:, :4]  # x1, y1, x2, y2
                scores = pred[0][:, 4]
                labels = pred[0][:, 5].long()
                
                # 将坐标转换回原始图像尺寸
                boxes, scores, labels = self.postprocess_boxes(boxes, scores, labels)
            else:
                print("未检测到任何目标")
                boxes = np.array([])
                scores = np.array([])
                labels = np.array([])
            
            # 读取原始图像用于可视化
            if isinstance(img_path, str):
                original_img = cv2.imread(img_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                original_img = img_path.copy()
                
            # 绘制结果
            processed_img = self.draw_boxes(original_img, boxes, scores, labels)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            
            return boxes, scores, labels, processed_img
            
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"预测失败: {str(e)}")

def main():
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认的模型和图像路径
        default_weights = os.path.join(current_dir, 'runs', 'train', 'exp2', 'epoch_0.pt')
        default_cfg = os.path.join(current_dir, 'models', 'yolov8.yaml')
        default_image = os.path.join(current_dir, 'images', 'bike.jpg')
        
        # 检查权重文件
        weights = os.getenv('YOLO_WEIGHTS', default_weights)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"权重文件未找到: {weights}\n请设置YOLO_WEIGHTS环境变量或将权重放在正确位置")
            
        # 检查配置文件
        cfg = os.getenv('YOLO_CFG', default_cfg)
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"配置文件未找到: {cfg}\n请设置YOLO_CFG环境变量或将配置文件放在正确位置")
            
        # 检查测试图片
        image_path = os.getenv('YOLO_TEST_IMAGE', default_image)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"测试图片未找到: {image_path}\n请设置YOLO_TEST_IMAGE环境变量或将图片放在正确位置")
        
        print(f"使用权重: {weights}")
        print(f"使用配置: {cfg}")
        print(f"测试图片: {image_path}")
        
        # 创建预测器
        predictor = YOLOPredictor(
            weights=weights,
            cfg=cfg,
            device='cuda'  # 如果没有GPU会自动切换到CPU
        )
        
        # 加载图像并预测
        print(f"\n开始处理图片...")
        boxes, scores, labels, result_img = predictor.predict(
            image_path,
            conf_thres=0.1,
            iou_thres=0.65
        )
        
        # 打印结果
        print("\n检测结果:")
        if len(boxes) > 0:
            for box, score, label_idx in zip(boxes, scores, labels):
                label = predictor.class_names[int(label_idx)]
                print(f'{label}: {score:.2f} at {box}')
        else:
            print("未检测到任何目标")
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(image_path), 'result.jpg')
        cv2.imwrite(output_path, result_img)
        print(f'\n结果已保存到: {output_path}')
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()