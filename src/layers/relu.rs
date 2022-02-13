use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct ReLU {
    input: Tensor
}

impl Layer for ReLU {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("ReLU"),
            output_shape: self.inp