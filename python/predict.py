import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from safetensors.torch import load_file
from conv_net import Net  # 导入你定义的网络
import time

def predict(image_path, model, device):
    start = time.time()
    
    # 图像加载
    load_start = time.time()
    image = Image.open(image_path)
    print(f"Image loading time: {time.time() - load_start:.4f}s")
    
    # 图像的预处理步骤
    transform = transforms.Compose([
        transforms.Grayscale(),  # 将图片转换为灰度图
        transforms.Resize((28, 28)),  # 缩放图片为28x28
        transforms.ToTensor(),  # 自动将像素值缩放到[0,1]范围
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    
    # 预处理
    preprocess_start = time.time()
    image = transform(image).unsqueeze(0).to(device)
    print(f"Preprocessing time: {time.time() - preprocess_start:.4f}s")
    
    # 推理
    inference_start = time.time()
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
    print(f"Inference time: {time.time() - inference_start:.4f}s")
    
    print(f"Total processing time: {time.time() - start:.4f}s")
    print("-------------------")
    
    # 添加相同的调试信息
    print("First few pixels:", image[0,0,0,:5].tolist())
    
    # 格式化输出张量，使其更易读
    print(f"Tensor[dims {list(output.shape)}; f32]")
    print(f"Predicted Digit - {prediction.item()}")
    print(f"\"{image_path}\": Predicted Digit - {prediction.item()}")

def main():
    start_total = time.time()
    
    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_weights = load_file("mnist_model.safetensors")
    model.load_state_dict(model_weights)
    model.eval()
    print(f"Model loading time: {time.time() - start_total:.4f}s")
    
    # 预测循环
    for filename in os.listdir('./'):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            predict(filename, model, device)
    
    print(f"Total execution time: {time.time() - start_total:.4f}s")

if __name__ == '__main__':
    main()
