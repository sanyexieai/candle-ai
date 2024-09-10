use candle_nn::{VarBuilder};
use candle_core::{IndexOp, Result, Tensor};
pub trait Model {
    fn load(vb: VarBuilder)-> Result<Self> where Self: Sized ;
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}