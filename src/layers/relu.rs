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
            output_shape: self.input.shape.to_vec(),
            trainable_param: 0,
            non_trainable_param: 0,
        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        self.input = input;
        self.input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {