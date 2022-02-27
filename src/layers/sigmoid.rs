use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Sigmoid {
    input: Tensor
}

impl Layer for Sigmoid {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("Sigmoid"),
            output_shape: self.input.shape.to_vec()