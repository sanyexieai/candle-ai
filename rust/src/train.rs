use candle_core::{DType, D, Device, Result, Tensor};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::prelude::*;
use shared::{model::Model, number_model::ConvNet}; // For shuffling

struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = m.train_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::load(vs.clone())?;

    if let Some(load) = &args.load {
        println!("Loading weights from {load}");
        varmap.load(load)?;
    }

    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    let n_batches = train_images.dim(0)? / BSIZE; // 计算总批次数
    let total_samples = train_images.dim(0)?; // 训练样本总数

    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;

        for batch_idx in 0..n_batches {
            // 计算当前批次的起始和结束索引
            let start_idx = batch_idx * BSIZE;
            let end_idx = usize::min(start_idx + BSIZE, total_samples);
            let actual_bsize = end_idx - start_idx;

            let train_images_batch = train_images.narrow(0, start_idx, actual_bsize)?;
            let train_labels_batch = train_labels.narrow(0, start_idx, actual_bsize)?;

            let logits = model.forward(&train_images_batch, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels_batch)?;

            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;

            // 打印训练损失
            println!(
                "Epoch: {} [{}/{} ({:.0}%)]\tLoss: {:.6}",
                epoch,
                start_idx + actual_bsize,
                total_samples,
                100. * (start_idx + actual_bsize) as f32 / total_samples as f32,
                loss.to_vec0::<f32>()?
            );
        }

        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        println!(
            "{:4} | Train Loss: {:8.5} | Test Accuracy: {:5.2}%",
            epoch,
            avg_loss,
            100. * test_accuracy
        );
    }

    if let Some(save) = &args.save {
        println!("Saving trained weights in {save}");
        //保存为.safetensors格式
        
        varmap.save(save)?;
    }

    Ok(())
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 5)]
    epochs: usize,

    #[arg(long)]
    save: Option<String>,

    #[arg(long)]
    load: Option<String>,

    #[arg(long, default_value = r"D:\code\rust\candle-ai\data\MNIST\raw\")]
    local_mnist: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let default_learning_rate = 0.0001;

    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load: args.load,
        save: args.save,
    };

    training_loop_cnn(m, &training_args)
}
