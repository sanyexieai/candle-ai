import torch
from torch import nn
from code_net import CodeNet
from loader import get_loader
from safetensors.torch import save_file  # safetensors 库导入
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(dataloader, model, loss_fn, optimizer):
    model.train()
 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
 
        pred = model(X)
        loss = loss_fn(pred, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if batch % 100 == 0:
            print(f"损失值: {loss:>7f}")
 
 
def main():
    model = CodeNet().to(device)
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_loader("../data/train_captcha")
    
    epoch = 25
    for t in range(epoch):
        print(f"训练周期 {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("\n")
 
    # 保存权重到 .safetensors 文件
    model_weights = model.state_dict()
    save_file(model_weights, "code.safetensors")  # 使用 safetensors 保存权重
    print("训练完成，模型已保存")
 
 
if __name__ == "__main__":
    main()