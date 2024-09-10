import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from safetensors.torch import load_file
from conv_net import Net  # 导入你定义的网络

def predict(image_path, model, device):
    # 图像的预处理步骤
    transform = transforms.Compose([
        transforms.Grayscale(),  # 将图片转换为灰度图
        transforms.Resize((28, 28)),  # 缩放图片为28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 打开图像并进行预处理
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 增加一个批处理维度并转换为 PyTorch Tensor

    # 进行推理
    with torch.no_grad():  # 禁用梯度计算
        output = model(image)
        print(output)
        prediction = output.argmax(dim=1, keepdim=True)  # 获取预测结果
        print(prediction)
    
    print(f"{image_path}: Predicted Digit - {prediction.item()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载网络结构
    model = Net().to(device)

    # 从 .safetensors 文件加载模型权重
    model_weights = load_file("mnist_model.safetensors")  # 加载safetensors文件
    model.load_state_dict(model_weights)  # 将权重加载到模型中

    # 设置模型为评估模式
    model.eval()

    # 遍历当前文件夹下的所有图片文件
    for filename in os.listdir('./'):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            predict(filename, model, device)

if __name__ == '__main__':
    main()
