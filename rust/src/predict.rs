use anyhow::{Context, Ok, Result};
use candle_core::{DType, Device, D};
use candle_nn::{VarBuilder};
use candle_core::Tensor;
use shared::{model::Model, number_model::ConvNet};
use std::path::PathBuf;

// 定义均值和标准差
const MEAN: f32 = 0.1307;
const STD: f32 = 0.3081;

// 归一化 Tensor
pub fn normalize(tensor: &Tensor, image_mean: f32, image_std: f32) -> Result<Tensor> {
        // 创建均值和标准差张量
        let mean_tensor = Tensor::from_vec(vec![image_mean], (1,), &Device::Cpu)?;
        let std_tensor = Tensor::from_vec(vec![image_std], (1,), &Device::Cpu)?;

        // 广播均值和标准差张量到 `tensor` 的形状
        let mean_tensor = mean_tensor.broadcast_as(tensor.shape())?;
        let std_tensor = std_tensor.broadcast_as(tensor.shape())?;

        // 进行标准化处理
        let tensor = (tensor - mean_tensor) / std_tensor;
        Ok(tensor?)
}


fn main() -> Result<()> {
    // 设置计算设备（CUDA 如果可用）
    let device = Device::cuda_if_available(0).context("Failed to find CUDA device")?;
    // 加载模型权重 (.safetensors)
    let safetensors_path = "../mnist_model.safetensors";
    // 创建 VarBuilder 并初始化神经网络
    let vb =unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)? };
    let model = ConvNet::load(vb).context("Failed to create ConvNet model")?;

    // 遍历当前目录下的所有图片文件
    for entry in std::fs::read_dir("../").context("Failed to read directory")? {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();
        if path.is_file() && (path.extension().map_or(false, |e| e == "png" || e == "jpg")) {
            let image_name = PathBuf::from(path);

            // 打开并处理图像
            let original_image = image::ImageReader::open(&image_name)
                .context("Failed to open image file")?
                .decode()
                .context("Image decoding error")?;
            // 图像处理和预测代码
            let width = 28;
            let height = 28;
            // 处理图像并转换为灰度
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_luma8().into_raw();

            // 创建一个二维 Tensor
            let mut image_t = Tensor::from_vec(
                data,
                (height as usize, width as usize), // 这里的形状是 (height, width)
                &device,
            )?;
            // image_t = normalize(&mut image_t, MEAN, STD)?;

            // 添加批量维度和通道维度
            image_t = image_t.unsqueeze(0)?; // 形状变为 (1, height, width)
            image_t = image_t.unsqueeze(0)?; // 形状变为 (1, 1, height, width)

            //
            // 转换为浮点型
            image_t = (image_t.to_dtype(DType::F32))?;
            image_t = normalize(&image_t, MEAN, STD)?;
            let predictions = model.forward(&image_t, false);
            // 获取预测结果
            let pred = predictions?
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
            println!("File {:?} predicted as {:?}", image_name, pred);
        }
    }

    Ok(())
}
