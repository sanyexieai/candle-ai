use wasm_bindgen::prelude::*;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder};
use shared::{model::Model, number_model::ConvNet};

#[wasm_bindgen]
pub struct ConvNetModel {
    model: ConvNet,
    device: Device,
}

#[wasm_bindgen]
impl ConvNetModel {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<u8>) -> Result<ConvNetModel, JsError> {
        let device = &Device::Cpu;  // 使用CPU，因为在浏览器中运行不支持CUDA
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        // 反序列化模型权重
        let model = ConvNet::load(vb)?;
        Ok(Self { model, device: device.clone() })
    }

    pub fn predict_image(&self, image_data: Vec<u8>, width: usize, height: usize) ->  Result<JsValue, JsError> {
        let image = image::load_from_memory(&image_data)?.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        let data = image.to_luma8().into_raw();

        // 创建一个二维 Tensor
        let mut image_t = Tensor::from_vec(
            data,
            (height as usize, width as usize), // 这里的形状是 (height, width)
            &self.device,
        )?;

        // 添加批量维度和通道维度
        image_t = image_t.unsqueeze(0)?; // 形状变为 (1, height, width)
        image_t = image_t.unsqueeze(0)?; // 形状变为 (1, 1, height, width)

        // 转换为浮点型
        let image_t = (image_t.to_dtype(DType::F32))?;
        // // 转换为浮点型，并进行归一化
        // let image_t = (image_t.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = self.model.forward(&image_t, false);
        // 获取预测结果
        let pred = predictions?
        .argmax(D::Minus1)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
        println!("predicted as {:?}", pred);
        Ok(pred.into())
    }
}
fn main() {
    console_error_panic_hook::set_once();
}
