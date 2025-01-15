import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, cache_images=False):
        """初始化数据集加载器
        
        Args:
            path: 数据集路径，可以是图片目录或者包含图片路径的txt文件
            img_size: 输入图像大小
            batch_size: 批次大小
            augment: 是否使用数据增强
            hyp: 超参数字典
            rect: 是否使用矩形训练
            cache_images: 是否缓存图像到内存
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.mosaic = self.augment and not self.rect
        
        # 获取图片路径
        try:
            f = []  # 图片文件路径
            for p in path if isinstance(path, list) else [path]:
                p = str(p)
                if os.path.isfile(p):  # 文件
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(os.path.dirname(p)) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                elif os.path.isdir(p):  # 目录
                    f += glob.glob(os.path.join(p, '*.*'))
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['bmp', 'jpg', 'jpeg', 'png'])
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')
            
        n = len(self.img_files)
        assert n > 0, f'No images found in {path}'
        
        # 获取标签路径
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                           for x in self.img_files]
                           
        # 缓存标签
        self.labels = [np.zeros((0, 5))] * n
        
        # 读取所有标签
        nm, nf, ne = 0, 0, 0  # 缺失标签数量，找到标签数量，空标签数量
        for i, f in enumerate(self.label_files):
            try:
                if os.path.isfile(f):
                    l = np.loadtxt(f, dtype=np.float32).reshape(-1, 5)
                    if len(l):
                        assert l.shape[1] == 5, f'标签的列数必须为5: {f}'
                        assert (l >= 0).all(), f'标签值必须为正数: {f}'
                        assert (l[:, 1:] <= 1).all(), f'标签值必须小于等于1: {f}'
                        self.labels[i] = l
                        nf += 1
                    else:
                        ne += 1
                else:
                    nm += 1
            except Exception as e:
                print(f'警告: 跳过标签 {f}: {e}')
                
        print(f'标签统计: {nf} 个有效, {nm} 个缺失, {ne} 个空标签')
        
        # 缓存图像
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # 缓存大小
            print(f'缓存图像到内存 ({gb:.1f}GB)')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in range(n):
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = self.load_image(i)
                gb += self.imgs[i].nbytes
                print(f'{i + 1}/{n}: {gb / 1E9:.1f}GB')
                
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, index):
        # 加载图像和标签
        img = self.load_image(index)
        labels = self.labels[index].copy()
        
        # 数据增强
        if self.augment:
            img, labels = self.random_affine(img, labels)
            
        nL = len(labels)  # 标签数量
        if nL:
            # 转换标签格式 xywh -> xyxy
            labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
            
            # 归一化坐标
            labels[:, [2, 4]] /= img.shape[0]  # 高度
            labels[:, [1, 3]] /= img.shape[1]  # 宽度
            
        # 转换图像格式
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), torch.from_numpy(labels), self.img_files[index], (img.shape[1], img.shape[0])
        
    def load_image(self, index):
        """加载图像
        
        Args:
            index: 图像索引
            
        Returns:
            图像数组
        """
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, f'图像不存在 {path}'
        
        # 调整图像大小
        h0, w0 = img.shape[:2]  # 原始高度和宽度
        r = self.img_size / max(h0, w0)  # 缩放比例
        if r != 1:  # 如果需要缩放
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        
        # 填充到指定大小
        h, w = img.shape[:2]
        dh, dw = self.img_size - h, self.img_size - w
        if dh > 0 or dw > 0:  # 如果需要填充
            top, bottom = dh // 2, dh - (dh // 2)
            left, right = dw // 2, dw - (dw // 2)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img
        
    @staticmethod
    def collate_fn(batch):
        """数据批次整理函数
        
        Args:
            batch: 数据批次
            
        Returns:
            整理后的数据批次
        """
        img, label, path, shapes = zip(*batch)  # 转置
        for i, l in enumerate(label):
            l[:, 0] = i  # 添加目标索引
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
        
    def random_affine(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10):
        """随机仿射变换
        
        Args:
            img: 输入图像
            targets: 目标框
            degrees: 旋转角度范围
            translate: 平移范围
            scale: 缩放范围
            shear: 剪切范围
            
        Returns:
            变换后的图像和目标框
        """
        # 待实现
        return img, targets


def xywh2xyxy(x):
    """将 [x, y, w, h] 转换为 [x1, y1, x2, y2]
    
    Args:
        x: 输入坐标
        
    Returns:
        转换后的坐标
    """
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y 