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
            output_shape: self.input.shape.to_vec(),
            trainable_param: 0,
            non_trainable_param: 0,

        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        self.input = input;
        self.input.map(|x| Sigmoid::sigmoid(x))
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {
        let tanh_grad = self.input.map(|x| Sigmoid::sigmoid_prime(x));
        gradient.mult_el(&tanh_grad)
    }

    fn get_params_list(&self) -> Vec<Le