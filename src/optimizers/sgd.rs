use crate::{layers::layer::Layer, optimizers::optimizer::OptimizerStep};

pub struct SGD {
    /// Learning Rate
    lr: f64
}

impl SGD {
    pub fn new(lr: f64) -> Self { Self { lr } }
}

impl Optimizer