import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from conv_net import Net
from safetensors.torch import save_file
import numpy as np

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    losses = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        losses.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            min_loss = min(losses) if losses else 0
            max_loss = max(losses) if losses else 0
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]')
            print(f'Train Loss: {avg_loss:.6f} (Min: {min_loss:.6f}, Max: {max_loss:.6f})')
            print(f'Train Accuracy: {accuracy:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}\n')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return accuracy

def main():
    # 检查CUDA是否可用，并打印设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

    best_accuracy = 0
    # 训练模型
    for epoch in range(1, 5):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存最佳模型
            model_weights = model.state_dict()
            save_file(model_weights, "mnist_model_best.safetensors")
    
    # 保存最终模型
    model_weights = model.state_dict()
    save_file(model_weights, "mnist_model_final.safetensors")

if __name__ == '__main__':
    main()
