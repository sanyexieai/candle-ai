use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Linear, VarBuilder, ops, Module};

pub trait Model {
    fn load(vb: VarBuilder) -> Result<Self> where Self: Sized;
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

pub struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl ConvNet {
    fn new(vb: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 3, Default::default(), vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 3, Default::default(), vb.pp("conv2"))?;
        let fc1 = candle_nn::linear(9216, 128, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(128, 10, vb.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
    }
}

impl Model for ConvNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        Self::new(vb)
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?;
        
        let xs = if train {
            ops::dropout(&xs, 0.25)?
        } else {
            xs
        };
        
        let xs = xs.flatten_from(1)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = xs.relu()?;
        
        let xs = if train {
            ops::dropout(&xs, 0.5)?
        } else {
            xs
        };
        
        self.fc2.forward(&xs)
    }
}