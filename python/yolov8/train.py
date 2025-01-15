import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import increment_path
from utils.metrics import fitness
from tqdm import tqdm
import logging
from pathlib import Path

def train(hyp, opt):
    """训练函数"""
    logger = logging.getLogger(__name__)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    train_path = opt.train_path
    val_path = opt.val_path
    
    # 数据加载器
    train_dataset = LoadImagesAndLabels(
        train_path,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=opt.cache_images
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn
    )
    
    # 创建模型
    model = Model(opt.cfg, ch=3, nc=opt.num_classes).to(device)
    
    # 更新模型超参数
    model.hyp.update(hyp)
    
    # 优化器
    pg0, pg1, pg2 = [], [], []  # 参数分组优化
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)
            else:
                pg0.append(v)
                
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    
    # 学习率调度器
    lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 损失函数
    compute_loss = ComputeLoss(model)
    
    # 开始训练
    logger.info(f'开始训练 {opt.epochs} 轮...')
    for epoch in range(opt.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        mloss = torch.zeros(3, device=device)  # mean losses
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            # 前向传播
            pred = model(imgs)
            
            # 计算损失
            loss, loss_items = compute_loss(pred, targets)
            
            # 反向传播
            loss.backward()
            
            # 优化器步进
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新进度条
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            s = ''
            for j, loss_val in enumerate(['box', 'obj', 'cls']):
                s += f'{loss_val}: {mloss[j].item():.4f} - '
            pbar.set_description(
                f'Epoch {epoch}/{opt.epochs} - '
                f'mem {mem} - '
                f'{s[:-3]}'  # 移除最后的 ' - '
            )
            
        # 更新学习率
        scheduler.step()
        
        # 保存模型
        if epoch % opt.save_period == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyp': hyp,
                'opt': opt
            }
            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch}.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov8.yaml', help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='数据集配置文件路径')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='超参数文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--project', default='runs/train', help='保存结果的项目名称')
    parser.add_argument('--name', default='exp', help='保存结果的实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='是否覆盖已存在的实验目录')
    parser.add_argument('--cache-images', action='store_true', help='是否缓存图像到内存')
    parser.add_argument('--num-classes', type=int, default=80, help='类别数量')
    parser.add_argument('--save-period', type=int, default=10, help='保存检查点的间隔轮数')
    opt = parser.parse_args()

    # 加载数据集配置
    with open(opt.data, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    # 更新训练和验证数据集路径
    opt.train_path = os.path.join(data_cfg['path'], data_cfg['train'])
    opt.val_path = os.path.join(data_cfg['path'], data_cfg['val'])
    opt.num_classes = data_cfg['nc']

    # 加载超参数
    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    # 开始训练
    train(hyp, opt) 