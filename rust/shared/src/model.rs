use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Linear, VarBuilder, ops, Module};

pub trait Model {
    fn load(vb: VarBuilder) -> Result<Self> where Self: Sized;
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}