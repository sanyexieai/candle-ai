import os
import glob
import re

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