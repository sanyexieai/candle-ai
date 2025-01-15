import numpy as np

def fitness(x):
    """计算适应度分数
    
    Args:
        x: [P, R, mAP@.5, mAP@.5-.95] 指标
        
    Returns:
        适应度分数
    """
    w = [0.0, 0.0, 0.1, 0.9]  # 权重
    return (x[:4] * w).sum() 