use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Dropout {
    input: Tensor,
    prob: f64,
    // Store the seed so the Dropout struct can increment it
    // to g