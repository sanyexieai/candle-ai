import os
from captcha.image import ImageCaptcha
import concurrent.futures
from pathlib import Path
import shutil
import random
 
CHAR_NUMBER = 4                                 # 字符数量
IMG_WIDTH = 160                                 # 图片宽度
IMG_HEIGHT = 60                                 # 图片高度
SEED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 字符池
 
def generate_captcha(num, output_dir, thread_name=0):
    """
    生成一定数量的验证码图片
    :param num: 生成数量
    :param output_dir: 存放验证码图片的文件夹路径
    :param thread_name: 线程名称
    :return: None
    """
    # 如果目录已存在，则先删除后再创建
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir()
 
    for i in range(num):
        img = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
        chars = "".join([random.choice(SEED) for _ in range(CHAR_NUMBER)])
        save_path = f"{output_dir}/{i + 1}-{chars}.png"
        img.write(chars, save_path)
        print(f"Thread {thread_name}: 已生成{i + 1}张验证码")
 
    print(f"Thread {thread_name}: 验证码图片生成完毕")
 
 
def main():
    train_path = "../data/train_captcha"
    test_path ="../data/test_captcha"
    #如果文件夹不存在 则创建
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(generate_captcha, 30000, train_path, 0)
        executor.submit(generate_captcha, 1000, test_path, 1)
 
 
if __name__ == '__main__':
    main()