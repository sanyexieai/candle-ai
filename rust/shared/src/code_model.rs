use candle_nn::{Activation, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Sequential, VarBuilder,Dropout};
use candle_core::{IndexOp, Result, Tensor,Error};
use std::fmt::{self, Debug};
use crate::model::Model;

pub struct CodeNet {
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    layer51: Sequential,
    layer54: Sequential,
}
impl CodeNet {
    pub fn flatten_tensor(tensor: &Tensor, start_dim: usize) -> Result<Tensor> {
        // 获取张量的形状
        let shape = tensor.shape();
        
        // 获取所有维度
        let dims = shape.dims();  // 假设 `dims` 返回一个 Vec<i64>，表示每个维度的大小
        
        // 检查 start_dim 是否在有效范围内
        if start_dim >= dims.len() {
            return Err(Error::DimOutOfRange {
                shape: tensor.shape().clone(),
                dim: dims.len() as i32,
                op: "flatten_tensor",
            }
            .bt())?;
        }
    
        // 计算新形状
        let mut new_shape = Vec::with_capacity(dims.len() - start_dim + 1);
        
        // 添加起始维度之前的维度
        for i in 0..start_dim {
            new_shape.push(dims[i]);
        }
    
        // 计算展平后的维度
        let mut flattened_dim = 1;
        for i in start_dim..dims.len() {
            flattened_dim *= dims[i];
        }
        new_shape.push(flattened_dim);
        print!("new_shape: {:?}", new_shape);
        // 调整形状
        tensor.reshape(new_shape)
       
    }
}

impl Model for CodeNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv1_config = Conv2dConfig {
            padding: 1, // 设置 padding
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        // 定义 layer1
        let layer1 = candle_nn::seq()
        .add(candle_nn::conv2d(1, 64, 3, conv1_config.clone(), vb.pp("layer1.0"))?)
        .add_fn(|xs| xs.relu())
        .add_fn(|xs|xs.max_pool2d(2));
        // 定义 layer2
        let layer2 = candle_nn::seq()
        .add(candle_nn::conv2d(64, 128, 3, conv1_config.clone(), vb.pp("layer2.0"))?)
        .add_fn(|xs| xs.relu())
        .add_fn(|xs|xs.max_pool2d(2));
        // 定义 layer3
        let layer3 = candle_nn::seq()
        .add(candle_nn::conv2d(128, 256, 3, conv1_config.clone(), vb.pp("layer3.0"))?)
        .add_fn(|xs| xs.relu())
        .add_fn(|xs|xs.max_pool2d(2));
        // 定义 layer4
        let layer4 = candle_nn::seq()
        .add(candle_nn::conv2d(256, 512, 3, conv1_config.clone(), vb.pp("layer4.0"))?)
        .add_fn(|xs| xs.relu())
        .add_fn(|xs|xs.max_pool2d(2));
        // 定义 layer5.1
        let layer51 = candle_nn::seq()
        .add_fn(|xs|CodeNet::flatten_tensor(xs,1))
        .add(candle_nn::linear(
            15360,
            4096,
            vb.pp("layer5.1"),
        )?);
        // 定义 layer5.4
        let layer54 = candle_nn::seq()
        .add(candle_nn::linear(
            4096,
            144,
            vb.pp("layer5.4"),
        )?);
        Ok(Self {
            layer1,
            layer2,
            layer3,
            layer4,
            layer51,
            layer54,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let dropout1 = candle_nn::Dropout::new(0.5);
        // 打印输入张量的形状
        println!("Input shape: {:?}", xs.shape());
        let mut x =  xs
        .apply(&self.layer1)?;
        println!("layer1 shape: {:?}", x.shape());
        x = x.apply(&self.layer2)?;
        println!("layer2 shape: {:?}", x.shape());
        x = x.apply(&self.layer3)?;
        println!("layer3 shape: {:?}", x.shape());
        x = x.apply(&self.layer4)?;
        println!("layer4 shape: {:?}", x.shape());
        x = x.apply(&self.layer51)?;
        x = x.apply_t(&dropout1, train)?.relu()?;
        x = x.apply(&self.layer54)?;
        println!("layer54 shape: {:?}", x.shape());
        Ok(x)
    }
    
}