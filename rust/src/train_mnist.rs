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
    let images = images.to_dtype(DType::F32)?;
    let images = (images / 255.0)?;
    let images = ((images - 0.1307)? / 0.3081)?;
    images.reshape((batch_size, 1, 28, 28))
}

// 数据增强函数
fn augment_batch(images: &Tensor) -> Result<Tensor> {
    // 添加随机噪声
    let noise = Tensor::randn(0.0, 1.0, images.shape(), images.device())?;
    let noise = (noise * 0.1)?;
    let images = (images + noise)?;
    
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

// 改进的学习率调度
fn get_learning_rate(initial_lr: f64, epoch: usize, total_epochs: usize) -> f64 {
    // 前5个epoch进行预热
    if epoch < 5 {
        return initial_lr * (epoch as f64 / 5.0);
    }
    
    // 余弦退火
    let min_lr = initial_lr * 0.01;
    let progress = (epoch - 5) as f64 / (total_epochs - 5) as f64;
    min_lr + (initial_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos()) * 0.5
}

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 128; // 增大批次大小

    let dev = match Device::cuda_if_available(0) {
        Ok(cuda_dev) => {
            println!("Training on CUDA GPU");
            cuda_dev
        }
        Err(e) => {
            println!("WARNING: CUDA not available ({}), falling back to CPU", e);
            Device::Cpu
        }
    };

    println!("Preprocessing and moving data to device...");
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = preprocess_data(&m.train_images)?.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = preprocess_data(&m.test_images)?.to_device(&dev)?;
    println!("Data successfully moved to device");

    println!("Initializing model...");
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::load(vs.clone())?;
    println!("Model initialized successfully");

    if let Some(load) = &args.load {
        println!("Loading weights from {load}");
        varmap.load(load)?;
    }

    // 优化器参数调整
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        weight_decay: 0.01, // 增加权重衰减
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    let n_batches = train_images.dim(0)? / BSIZE;
    let total_samples = train_images.dim(0)?;

    let mut best_accuracy = 0f32;
    let mut patience_counter = 0;
    let mut best_model_path = String::from("best_model.safetensors");

    // 训练循环
    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;
        let mut correct_train = 0usize;
        let mut total_train = 0usize;
        
        // 打乱数据
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

            // 计算训练准确率
            let predicted = logits.argmax(D::Minus1)?;
            let correct = predicted.eq(&train_labels_batch)?.to_dtype(DType::U32)?;
            correct_train += correct.sum_all()?.to_scalar::<u32>()? as usize;
            total_train += actual_bsize;

            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;

            if batch_idx % 50 == 0 {
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
        let train_accuracy = correct_train as f32 / total_train as f32;

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
            "Epoch {:4} | Train Loss: {:8.5} | Train Acc: {:5.2}% | Test Acc: {:5.2}% | LR: {:.6}",
            epoch,
            avg_loss,
            100. * train_accuracy,
            100. * test_accuracy,
            get_learning_rate(args.learning_rate, epoch, args.epochs)
        );

        // 更新学习率
        opt.set_learning_rate(get_learning_rate(args.learning_rate, epoch, args.epochs));

        // 早停检查
        if test_accuracy > best_accuracy {
            best_accuracy = test_accuracy;
            patience_counter = 0;
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

    #[arg(long, default_value_t = 50)]
    epochs: usize,

    #[arg(long,default_value = r"best_model.safetensors")]
    save: Option<String>,

    #[arg(long)]
    load: Option<String>,

    #[arg(long, default_value = "data/MNIST/raw")]
    local_mnist: Option<String>,

    #[arg(long, default_value_t = 5)]
    patience: usize,
}

pub fn main() -> anyhow::Result<()> {
    // 简化CUDA检查
    println!("Checking CUDA availability...");
    match Device::cuda_if_available(0) {
        Ok(_) => println!("CUDA is available"),
        Err(e) => println!("CUDA not available: {}", e),
    }

    let args = Args::parse();

    let m = if let Some(directory) = args.local_mnist {
        println!("Attempting to load MNIST dataset from: {}", directory);
        
        // 创建目录如果不存在
        if let Err(e) = std::fs::create_dir_all(&directory) {
            println!("Warning: Failed to create directory {}: {}", directory, e);
        }
        
        match candle_datasets::vision::mnist::load_dir(&directory) {
            Ok(dataset) => {
                println!("Successfully loaded MNIST dataset from local directory");
                dataset
            }
            Err(e) => {
                println!("Failed to load from local directory: {}. Attempting to download...", e);
                // 如果本地加载失败，尝试下载
                candle_datasets::vision::mnist::load()?
            }
        }
    } else {
        println!("No local directory specified, downloading MNIST dataset...");
        candle_datasets::vision::mnist::load()?
    };

    println!("Successfully loaded dataset with following shapes:");
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
