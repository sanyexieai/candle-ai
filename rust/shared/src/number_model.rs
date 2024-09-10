use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder};
use std::fmt::Debug;

use crate::model::Model;


#[derive(Debug)]
pub struct ConvNet {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    fc: Linear,
}
impl Model for ConvNet {
     fn load(vb: VarBuilder) -> Result<Self> {
        let conv2d_config = Conv2dConfig {
            padding: 1, // 设置 padding
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = candle_nn::conv2d(1, 4, 3, conv2d_config, vb.pp("conv1"))?;
        let batch_norm_bonfig=BatchNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        let bn1: BatchNorm = candle_nn::batch_norm(4, batch_norm_bonfig, vb.pp("bn1"))?;

        let conv2_config = Conv2dConfig {
            padding: 1, // 设置 padding
            ..Default::default()
        };
        let conv2 = candle_nn::conv2d(4, 8, 3, conv2_config, vb.pp("conv2"))?;

        let bn2 = candle_nn::batch_norm(8, BatchNormConfig::default(), vb.pp("bn2"))?;

        let conv3_config = Conv2dConfig {
            padding: 1, // 设置 padding
            ..Default::default()
        };
        let conv3 = candle_nn::conv2d(8, 16, 3, conv3_config, vb.pp("conv3"))?;

        let bn3 = candle_nn::batch_norm(16, BatchNormConfig::default(), vb.pp("bn3"))?;

        let fc = candle_nn::linear(16 * 3 * 3, 10, vb.pp("fc"))?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            fc,
        })
    }
     fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // // 打印输入张量的形状
        // println!("Input shape: {:?}", xs.shape());
        // Self::print_tensor_values(&xs, "Input")?;
        let x1 = xs
            .apply(&self.conv1)? // 应用第一个卷积层
            .max_pool2d(2)? // Max pooling: 使用 2x2 核，池化操作
            .apply_t(&self.bn1, train)? // 应用批归一化
            .relu()?;
        let x2 = x1
            .apply(&self.conv2)? // 应用第二个卷积层
            .max_pool2d(2)? // Max pooling: 使用 2x2 核，池化操作
            .apply_t(&self.bn2, train)? // 应用批归一化
            .relu()?;
        let x3 = x2
            .apply(&self.conv3)? // 应用第三个卷积层
            .max_pool2d(2)? // Max pooling: 使用 2x2 核，池化操作
            .apply_t(&self.bn3, train)? // 应用批归一化
            .relu()?;

        let x4 = x3
            .flatten_from(1)? // 展平为 (batch, 16*3*3)
            .apply(&self.fc)?;
        Ok(x4)
    }
}
