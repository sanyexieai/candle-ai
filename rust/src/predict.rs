use anyhow::{Context, Ok, Result};
use candle_core::{DType, Device, D, IndexOp};
use candle_nn::{VarBuilder};
use candle_core::Tensor;
use shared::{model::Model, number_model::ConvNet};
use std::path::PathBuf;
use image::imageops;
use std::time::Instant;

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
        let tensor = (tensor - mean_tensor)
            .map_err(|e| anyhow::anyhow!("减法运算失败: {}", e))?;
        let tensor = tensor.div(&std_tensor)
            .map_err(|e| anyhow::anyhow!("除法运算失败: {}", e))?;
        Ok(tensor)
}

// 添加一个辅助函数来转换Duration为秒
fn duration_to_seconds(d: std::time::Duration) -> f64 {
    d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9
}

fn main() -> Result<()> {
    let start_total = Instant::now();
    
    // 1. 使用固定路径，避免重复计算
    let safetensors_path = "../mnist_model.safetensors";
    
    // 2. 确定使用的设备
    let device = Device::cuda_if_available(0).context("Failed to find CUDA device")?;
    println!("Using device: {:?}", device);
    
    // 3. 预先分配内存
    let mut buffer = vec![0f32; 28 * 28];
    
    // 4. 一次性加载模型
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)? };
    let model = ConvNet::load(vb).context("Failed to create ConvNet model")?;
    println!("Model loading time: {:.4}s", duration_to_seconds(start_total.elapsed()));

    // 5. 预先创建均值和标准差张量，避免重复创建
    let mean_tensor = Tensor::from_vec(vec![MEAN], (1,), &device)?;
    let std_tensor = Tensor::from_vec(vec![STD], (1,), &device)?;

    for entry in std::fs::read_dir("../").context("Failed to read directory")? {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();
        if path.is_file() && (path.extension().map_or(false, |e| e == "png" || e == "jpg")) {
            let start = Instant::now();
            let image_name = PathBuf::from(path);

            // 打开并处理图像
            let load_start = Instant::now();
            let original_image = image::ImageReader::open(&image_name)
                .with_context(|| format!("Failed to open image file: {:?}", image_name))?
                .decode()
                .context("Image decoding error")?;
            println!("Image loading time: {:.4}s", duration_to_seconds(load_start.elapsed()));

            // 图像处理和预测代码
            let width = 28;
            let height = 28;
            // 使用更简单的调整算法
            let preprocess_start = Instant::now();

            // 1. 直接在resize时指定输出格式为灰度图
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::Nearest,
            ).into_luma8();

            // 2. 一次性完成数据转���，避免中间步骤
            buffer.clear();
            buffer.extend(
                img.into_raw()
                    .into_iter()
                    .map(|x| x as f32 / 255.0)
            );

            // 3. 使用更高效的Tensor创建方式
            let mut image_t = Tensor::from_vec(buffer.clone(), (1, 1, 28, 28), &device)?;

            // 4. 使用预先创建的均值和标准差张量
            let mean_broadcast = mean_tensor.broadcast_as(image_t.shape())?;
            let std_broadcast = std_tensor.broadcast_as(image_t.shape())?;
            image_t = ((image_t - mean_broadcast)? / std_broadcast)?;

            println!("Preprocessing time: {:.4}s", duration_to_seconds(preprocess_start.elapsed()));

            // 在标准化之后，打印一些像素值进行对比
            let pixels = vec![
                image_t.i(0)?.i(0)?.i(0)?.i(0)?.to_scalar::<f32>()?,
                image_t.i(0)?.i(0)?.i(0)?.i(1)?.to_scalar::<f32>()?,
                image_t.i(0)?.i(0)?.i(0)?.i(2)?.to_scalar::<f32>()?,
                image_t.i(0)?.i(0)?.i(0)?.i(3)?.to_scalar::<f32>()?,
                image_t.i(0)?.i(0)?.i(0)?.i(4)?.to_scalar::<f32>()?
            ];
            println!("First few pixels: {:?}", pixels);

            // 推理
            let inference_start = Instant::now();
            let predictions = model.forward(&image_t, false)?;
            let pred = predictions
                .argmax(D::Minus1)?
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .to_scalar::<f32>()?;
            println!("Inference time: {:.4}s", duration_to_seconds(inference_start.elapsed()));

            println!("Total processing time: {:.4}s", duration_to_seconds(start.elapsed()));
            println!("-------------------");

            println!("Predicted Digit - {}", pred);  // 打印预测数字
            println!("\"{:?}\": Predicted Digit - {}", image_name, pred);  // 添加引号使格式一致
        }
    }

    println!("Total execution time: {:.4}s", duration_to_seconds(start_total.elapsed()));
    Ok(())
}
