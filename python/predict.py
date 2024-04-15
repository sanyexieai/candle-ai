import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from conv_net import Net

def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(),  # 将图片转换为灰度图
        transforms.Resize((28, 28)),  # 缩放图片为28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 增加一个批处理维度
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = output.max(1, keepdim=True)[1]
        print(f"{image_path}: Predicted Digit - {prediction.item()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_model.pth"))

    # 遍历当前文件夹下的所有图片文件
    for filename in os.listdir('./'):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            predict(filename, model, device)

if __name__ == '__main__':
    main()
