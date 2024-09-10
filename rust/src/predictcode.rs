use anyhow::{Context, Ok, Result,anyhow};
use candle_core::{DType, Device, D,Error};
use candle_nn::{VarBuilder};
use candle_core::Tensor;
use shared::{model::Model, code_model::CodeNet};
use std::path::PathBuf;


const SEED: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub fn one_hot_decode(pred_result: &Tensor) -> Result<String> {
    // 获取 pred_result 的形状
    let shape = pred_result.shape();
    
    // 假设 SEED 长度是 36
    let num_classes = SEED.len();

    // 获取形状的维度
    let dims = shape.dims();  // 获取每个维度的大小
    
    if dims.len() != 2 {
        return Err(candle_core::Error::DimOutOfRange {
            shape: shape.clone(),
            dim: dims.len() as i32,
            op: "one_hot_decode",
        }.into());
    }

    let (batch_size, total_features) = (dims[0], dims[1]);

    // 确保 total_features 是 num_classes 的倍数
    if total_features % num_classes != 0 {
        return Err(candle_core::Error::DimOutOfRange {
            shape: shape.clone(),
            dim: num_classes as i32,
            op: "one_hot_decode",
        }.into());
    }

    let new_batch_size = total_features / num_classes;
    if batch_size != 1 {
        return Err(candle_core::Error::DimOutOfRange {
            shape: shape.clone(),
            dim: 1,
            op: "one_hot_decode",
        }.into());
    }

    // 转换 pred_result 形状从 [1, total_features] 到 [new_batch_size, num_classes]
    let pred_result = pred_result.reshape(&[new_batch_size, num_classes])?;
    // 提取数据
    let data = pred_result.flatten_all()?.to_vec1::<f32>()?; 

    // 从索引创建字符串
    let mut index_list = Vec::with_capacity(new_batch_size);
    for i in 0..new_batch_size {
        let start_index = i * num_classes;
        let end_index = start_index + num_classes;
        let row = &data[start_index..end_index];

        let mut max_index = 0;
        let mut max_value = row[0];
        for (j, &value) in row.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = j;
            }
        }
        index_list.push(max_index);
    }

    let text: String = index_list.iter()
        .map(|&i| SEED.chars().nth(i).unwrap())
        .collect();
    
    Ok(text)
}


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
    let safetensors_path = "../code.safetensors";
    // 创建 VarBuilder 并初始化神经网络
    let vb =unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)? };
    let model = CodeNet::load(vb).context("Failed to create CodeNet model")?;

    // 遍历当前目录下的所有图片文件
    for entry in std::fs::read_dir("../code/").context("Failed to read directory")? {
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
            let width = 160;
            let height = 60;
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
            // 添加批量维度和通道维度
            image_t = image_t.unsqueeze(0)?; // 形状变为 (1, height, width)
            image_t = image_t.unsqueeze(0)?; // 形状变为 (1, 1, height, width)

            //
            // 转换为浮点型
            image_t = (image_t.to_dtype(DType::F32))?;
            image_t = normalize(&image_t, MEAN, STD)?;
            let predictions = model.forward(&image_t, false)?;
            // 获取预测结果
            let pred = one_hot_decode(&predictions)?;
            println!("File {:?} predicted as {:?}", image_name, pred);
        }
    }

    Ok(())
}
