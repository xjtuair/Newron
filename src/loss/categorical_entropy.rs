use crate::{tensor::Tensor, loss::loss::Loss, utils, layers::softmax::Softmax};
pub struct CategoricalEntropy {}

impl Loss for CategoricalEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        let m = y_true.shape[0]