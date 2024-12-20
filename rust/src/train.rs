use candle_core::{DType, D, Device, Result, Tensor};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::prelude::*;
use shared::{model::Model, number_model::ConvNet};

struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
    patience: usize,
}

fn preprocess_data(images: &Tensor) -> Result<Tensor> {
    let batch_size = images.dim(0)?;
    // 归一化到 [0,1]
    let images = images.to_dtype(DType::F32)?;
    let images = (images.to_dtype(DType::F64)? / 255.0)?.to_dtype(DType::F32)?;
    // 标准化处理
    let images = ((images.to_dtype(DType::F64)? - 0.1307)? / 0.3081)?.to_dtype(DType::F32)?;
    // 重塑为 [batch_size, channels, height, width]
    let images = images.reshape((batch_size, 1, 28, 28))?;
    Ok(images)
}

fn shuffle_data(images: &Tensor, labels: &Tensor) -> Result<(Tensor, Tensor)> {
    let n_samples = images.dim(0)?;
    let mut indices: Vec<i64> = (0..n_samples as i64).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);
    
    let indices_tensor = Tensor::from_vec(indices.clone(), (n_samples,), &images.device())?;
    let shuffled_images = images.index_select(&indices_tensor, 0)?;
    let shuffled_labels = labels.index_select(&indices_tensor, 0)?;
    
    Ok((shuffled_images, shuffled_labels))
}

fn get_learning_rate(initial_lr: f64, epoch: usize, total_epochs: usize) -> f64 {
    let min_lr = initial_lr * 0.01;
    let progress = epoch as f64 / total_epochs as f64;
    min_lr + (initial_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos()) * 0.5
}

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let dev = candle_core::Device::cuda_if_available(0)?;

    // 预处理数据
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = preprocess_data(&m.train_images)?.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = preprocess_data(&m.test_images)?.to_device(&dev)?;

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

    let n_batches = train_images.dim(0)? / BSIZE;
    let total_samples = train_images.dim(0)?;

    // 早停相关状态
    let mut best_accuracy = 0f32;
    let mut patience_counter = 0;
    let mut best_model_path = String::from("best_model.safetensors");

    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;
        
        // 在每个epoch开始时打乱数据
        let (shuffled_images, shuffled_labels) = shuffle_data(&train_images, &train_labels)?;

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * BSIZE;
            let end_idx = usize::min(start_idx + BSIZE, total_samples);
            let actual_bsize = end_idx - start_idx;

            let train_images_batch = shuffled_images.narrow(0, start_idx, actual_bsize)?;
            let train_labels_batch = shuffled_labels.narrow(0, start_idx, actual_bsize)?;

            let logits = model.forward(&train_images_batch, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels_batch)?;

            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;

            if batch_idx % 100 == 0 {
                println!(
                    "Epoch: {} [{}/{} ({:.0}%)]\tLoss: {:.6}",
                    epoch,
                    start_idx + actual_bsize,
                    total_samples,
                    100. * (start_idx + actual_bsize) as f32 / total_samples as f32,
                    loss.to_vec0::<f32>()?
                );
            }
        }

        let avg_loss = sum_loss / n_batches as f32;

        // 评估测试集
        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        println!(
            "{:4} | Train Loss: {:8.5} | Test Accuracy: {:5.2}% | LR: {:.6}",
            epoch,
            avg_loss,
            100. * test_accuracy,
            get_learning_rate(args.learning_rate, epoch, args.epochs)
        );

        // 更新学习率
        opt.set_learning_rate(get_learning_rate(args.learning_rate, epoch, args.epochs));

        // 早停检查
        if test_accuracy > best_accuracy {
            best_accuracy = test_accuracy;
            patience_counter = 0;
            // 保存最佳模型
            if let Some(save) = &args.save {
                best_model_path = save.clone();
                println!("Saving best model with accuracy {:.2}% to {}", 100. * best_accuracy, best_model_path);
                varmap.save(&best_model_path)?;
            }
        } else {
            patience_counter += 1;
            if patience_counter >= args.patience {
                println!("Early stopping triggered after {} epochs without improvement", args.patience);
                break;
            }
        }
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

    #[arg(long, default_value_t = 5)]
    patience: usize,
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
        patience: args.patience,
    };

    training_loop_cnn(m, &training_args)
}
