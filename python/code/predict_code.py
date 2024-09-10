import os
import torch
from PIL import Image
from code_net import CodeNet
from loader import one_hot_decode
from safetensors.torch import load_file
from torchvision import transforms
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def predict(model, file_path):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale()
    ])
    with torch.no_grad():
        X = trans(Image.open(file_path)).reshape(1, 1, 60, 160)
        # 打印结果的形状
        print(X.shape) 
        pred = model(X)
        print(pred)
        print(pred.shape)
        text = one_hot_decode(pred)
        return text
 
 
def main():
    # 加载网络结构
    model = CodeNet().to(device)

    # 从 .safetensors 文件加载模型权重
    model_weights = load_file("code.safetensors")  # 加载safetensors文件
    model.load_state_dict(model_weights)  # 将权重加载到模型中
    model.eval()
 
    correct = 0
    test_dir = "../data/test_captcha"
    total = len(os.listdir(test_dir))
    for filename in os.listdir(test_dir):
        file_path = f"{test_dir}/{filename}"
        real_captcha = file_path.split("-")[-1].replace(".png", "")
        pred_captcha = predict(model, file_path)
        
        if pred_captcha == real_captcha:
            correct += 1
            print(f"{file_path}的预测结果为{pred_captcha}，预测正确")
        else:
            print(f"{file_path}的预测结果为{pred_captcha}，预测错误")
 
    accuracy = f"{correct / total * 100:.2f}%"
    print(accuracy)

main()