use shared::model::ConvNet;
use candle_nn::{VarBuilder, VarMap};
use candle_core::{DType, Tensor, D};


fn main() -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::new(vs.clone())?;

    // 加载训练好的模型权重
    varmap.load("model.candle")?;

    // 遍历当前目录下的所有图片文件
    for entry in std::fs::read_dir(".")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && (path.extension().map_or(false, |e| e == "png" || e == "jpg")) {
            let image_name = std::path::PathBuf::from(path);
            let original_image = image::io::Reader::open(&image_name)?
                .decode()
                .map_err(candle_core::Error::wrap)?;
            let width=28;
            let height= 28;
            let image_t = {
                let img = original_image.resize_exact(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::CatmullRom,
                );
                // let data = img.to_rgb8().into_raw();
                //转换为灰度图
                let data = img.to_luma8().into_raw();
                Tensor::from_vec(
                    data,
                    width*height,
                    &dev,
                )?
                // .permute((2, 0, 1))?
            };
            let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
            let predictions = model.forward(&image_t,false)?.squeeze(0)?;
            let pred = predictions
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
            println!("File {:?} predicted as {:?}", image_name, pred);
        }
    }

    Ok(())
}
 