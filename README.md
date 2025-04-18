# 图像分类算法实现 - MNIST

这是一个入门级的图像分类算法实现项目，旨在通过两种不同的编程语言（Python 和 Rust）分别实现 MNIST 数据集的图像分类任务。

## 项目概述

- 使用 **Python (PyTorch)** 进行模型的训练，训练后的模型以 `.safetensors` 格式保存。
- 使用 **Rust (Candle)** 加载已经训练好的 `.safetensors` 文件，并进行图像分类的预测。
- 项目已经成功编译为 WebAssembly (WASM) 版本，可以在 Web 环境中运行。

## 实现细节

1. **Python 实现 (PyTorch)**  
   使用 PyTorch 构建卷积神经网络（CNN），对 MNIST 数据集进行训练。训练好的模型保存为 `.safetensors` 文件格式，以便后续使用 Rust 加载和预测。

2. **Rust 实现 (Candle)**  
   使用 Candle 库加载由 Python 保存的 `.safetensors` 模型文件，并对输入的图片进行预测。Rust 实现部分目前已成功编译为 WASM，可以在 Web 浏览器中进行推理。

## 已实现功能

- **模型训练：** 使用 PyTorch 实现 CNN 模型，对 MNIST 数据集进行训练并保存模型。
- **模型推理：** 使用 Rust (Candle) 实现从 `.safetensors` 文件加载模型并进行图像分类预测。
- **WASM 编译：** Rust 版本已编译为 WebAssembly，可以在 Web 环境中运行。


## 使用说明

### Python 端 (PyTorch)

1. 训练模型并保存为 `.safetensors` 文件：

```bash
python train.py
```
2. 运行预测
```
python predict.py
```
### Rust 端 (Candle)
1. 使用 Rust 加载 .safetensors 文件并进行预测：
```
cargo run --release
```
3. 编译为 WebAssembly：
```
wasm-pack build --target web
```
