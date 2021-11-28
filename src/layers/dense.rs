use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Dense {
    input: Tensor,
    weights: Tensor,
    biases: Tensor,
    weights_grad: 