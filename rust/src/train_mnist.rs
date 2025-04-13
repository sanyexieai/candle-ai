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
    // 移除数据增强或减小噪声强度
    // let noise = Tensor::randn(0.0f32, 1.0f32, images.shape(), images.device())?;
    // let scale = Tensor::new(0.01f32, images.device())?;  // 如果保留，减小噪声强度
    // let noise = noise.broadcast_mul(&scale)?;
    // let images = (images + noise)?;
    Ok(images.clone())
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
    initial_lr  // 保持固定学习率
}

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;  // 与 Python 版本一致

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

    // 分析数据分布
    println!("\nAnalyzing data distribution:");
    let train_mean = train_images.mean_all()?.to_scalar::<f32>()?;
    let train_std = {
        let mean_tensor = Tensor::new(train_mean, &dev)?;
        let mean_tensor = mean_tensor.broadcast_as(train_images.shape())?;
        let diff = (train_images.clone() - mean_tensor)?;
        let squared = diff.sqr()?;
        let mean_squared = squared.mean_all()?.to_scalar::<f32>()?;
        mean_squared.sqrt()
    };
    println!("Training data - Mean: {:.4}, Std: {:.4}", train_mean, train_std);
    
    let test_mean = test_images.mean_all()?.to_scalar::<f32>()?;
    let test_std = {
        let mean_tensor = Tensor::new(test_mean, &dev)?;
        let mean_tensor = mean_tensor.broadcast_as(test_images.shape())?;
        let diff = (test_images.clone() - mean_tensor)?;
        let squared = diff.sqr()?;
        let mean_squared = squared.mean_all()?.to_scalar::<f32>()?;
        mean_squared.sqrt()
    };
    println!("Test data - Mean: {:.4}, Std: {:.4}", test_mean, test_std);

    println!("\nInitializing model...");
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
        lr: 0.001,  // 与 Python 版本一致
        weight_decay: 0.0001,  // 降低权重衰减
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    let n_batches = train_images.dim(0)? / BSIZE;
    let total_samples = train_images.dim(0)?;

    let mut best_accuracy = 0f32;
    let mut patience_counter = 0;
    let mut best_model_path = String::from("best_model.safetensors");
    let mut learning_rates = Vec::new();
    let mut train_losses = Vec::new();
    let mut test_accuracies = Vec::new();

    println!("\nStarting training with parameters:");
    println!("Batch size: {}", BSIZE);
    println!("Initial learning rate: {}", args.learning_rate);
    println!("Epochs: {}", args.epochs);
    println!("Patience: {}", args.patience);

    // 训练循环
    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;
        let mut correct_train = 0usize;
        let mut total_train = 0usize;
        let mut batch_losses = Vec::new();
        
        // 打乱数据
        let (shuffled_images, shuffled_labels) = shuffle_data(&train_images, &train_labels)?;

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * BSIZE;
            let end_idx = usize::min(start_idx + BSIZE, total_samples);
            let actual_bsize = end_idx - start_idx;

            let train_images_batch = shuffled_images.narrow(0, start_idx, actual_bsize)?;
            let train_labels_batch = shuffled_labels.narrow(0, start_idx, actual_bsize)?;

            // 应用数据增强
            let train_images_batch = augment_batch(&train_images_batch)?;

            let logits = model.forward(&train_images_batch, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels_batch)?;

            // 计算训练准确率
            let predicted = logits.argmax(D::Minus1)?;
            let correct = predicted.eq(&train_labels_batch)?.to_dtype(DType::U32)?;
            correct_train += correct.sum_all()?.to_scalar::<u32>()? as usize;
            total_train += actual_bsize;

            opt.backward_step(&loss)?;
            let batch_loss = loss.to_vec0::<f32>()?;
            sum_loss += batch_loss;
            batch_losses.push(batch_loss);

            if batch_idx % 50 == 0 {
                let current_lr = opt.learning_rate();
                println!(
                    "Epoch: {} [{}/{} ({:.0}%)]\tLoss: {:.6}\tLR: {:.6}",
                    epoch,
                    start_idx + actual_bsize,
                    total_samples,
                    100. * (start_idx + actual_bsize) as f32 / total_samples as f32,
                    batch_loss,
                    current_lr
                );
            }
        }

        let avg_loss = sum_loss / n_batches as f32;
        let train_accuracy = correct_train as f32 / total_train as f32;
        train_losses.push(avg_loss);

        // 评估测试集
        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        test_accuracies.push(test_accuracy);

        // 计算当前学习率
        let current_lr = get_learning_rate(args.learning_rate, epoch, args.epochs);
        learning_rates.push(current_lr);
        opt.set_learning_rate(current_lr);

        // 打印详细的训练统计信息
        println!("\nEpoch {} Summary:", epoch);
        println!("Train Loss: {:.6} (Min: {:.6}, Max: {:.6})", 
            avg_loss,
            batch_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            batch_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );
        println!("Train Accuracy: {:.2}%", 100. * train_accuracy);
        println!("Test Accuracy: {:.2}%", 100. * test_accuracy);
        println!("Learning Rate: {:.6}", current_lr);

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
                println!("\nEarly stopping triggered after {} epochs without improvement", args.patience);
                println!("Best accuracy achieved: {:.2}%", 100. * best_accuracy);
                break;
            }
        }
    }

    // 打印最终训练统计信息
    println!("\nTraining Summary:");
    println!("Best Test Accuracy: {:.2}%", 100. * best_accuracy);
    println!("Final Learning Rate: {:.6}", learning_rates.last().unwrap_or(&args.learning_rate));
    println!("Total Epochs Trained: {}", train_losses.len());
    
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

    #[arg(long, default_value = "./data/MNIST/raw")]
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
