use crate::{tensor::Tensor, loss::loss::Loss};
pub struct MSE {}

impl Loss for MSE {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f