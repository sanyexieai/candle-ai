use wasm_bindgen::prelude::*;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use shared::{model::Model, number_model::ConvNet};
use web_sys::Performance;

const MEAN: f32 = 0.1307;
const STD: f32 = 0.3081;

#[wasm_bindgen]
pub struct ImagePredictor {
    model: ConvNet,
    device: Device,
    performance: Performance,
}

#[wasm_bindgen]
impl ImagePredictor {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<u8>) -> Result<ImagePredictor, JsError> {
        console_error_panic_hook::set_once();
        let device = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        let model = ConvNet::load(vb)?;
        let window = web_sys::window().ok_or_else(|| JsError::new("No window found"))?;
        let performance = window.performance().ok_or_else(|| JsError::new("No performance found"))?;
        Ok(Self { model, device: device.clone(), performance })
    }

    pub fn predict(&self, image_data: Vec<u8>) -> Result<JsValue, JsError> {
        let start = self.performance.now();

        // 固定输入图片大小
        const WIDTH: usize = 28;
        const HEIGHT: usize = 28;

        let image = image::load_from_memory(&image_data)?
            .resize_exact(
                WIDTH as u32,
                HEIGHT as u32,
                image::imageops::FilterType::Nearest,
            );

        // 转换像素值到0-1范围
        let data: Vec<f32> = image.to_luma8()
            .into_raw()
            .into_iter()
            .map(|x| x as f32 / 255.0)
            .collect();
        
        let mut image_t = Tensor::from_vec(data, (HEIGHT, WIDTH), &self.device)?;
        image_t = image_t.unsqueeze(0)?;
        image_t = image_t.unsqueeze(0)?;
        
        // 添加归一化处理
        let mean = Tensor::new(&[MEAN], &self.device)?;
        let std = Tensor::new(&[STD], &self.device)?;
        let mean = mean.broadcast_as(image_t.shape())?;
        let std = std.broadcast_as(image_t.shape())?;
        let image_t = ((image_t - mean)? / std)?;

        let predictions = self.model.forward(&image_t, false)?;
        let pred = predictions
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let duration = self.performance.now() - start;
        let result = format!("预测结果: {}\n耗时: {:.2}ms", pred, duration);
        Ok(JsValue::from_str(&result))
    }
}
