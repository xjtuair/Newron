use crate::{tensor::Tensor, loss::loss::Loss, utils, layers::softmax::Softmax};
pub struct CategoricalEntropy {}

impl Loss for CategoricalEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        let m = y_true.shape[0];

        let indices = utils::one_hot_encoded_tensor_to_indices(y_true);

        // Even if last layer is Softmax, we softmax-it again for numerical stability
        let softmax_value = Softmax::softmax(y_pred);

        let mut p = Vec::new();
        for (row, indice) in indices.iter().enumerate() {
     