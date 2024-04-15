use wasm_bindgen::prelude::*;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder};
use shared::model::{ConvNet};

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
        let model = ConvNet::new(vb.clone())?;
        Ok(Self { model, device: device.clone() })
    }

    pub fn predict_image(&self, image_data: Vec<u8>, width: usize, height: usize) ->  Result<JsValue, JsError> {
        let image = image::load_from_memory(&image_data)?.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        let data = image.to_luma8().into_raw();
        let image_t = Tensor::from_vec(
            data,
            (width * height),
            &self.device,
        )?;

        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = self.model.forward(&image_t, false)?.squeeze(0)?;
        let pred = predictions
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        Ok(pred.into())
    }
}
fn main() {
    console_error_panic_hook::set_once();
}
