use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Dense {
    input: Tensor,
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor
}

impl Dense {
    pub fn new(input_units: usize, output_units: usize, seed: u32) -> Dense {
        // initialize with random values following special normal distribution
        // allowing theoritical faster convergence (Xavier Initialization)
        let variance_w = 2.0 / (input_units + output_units) as f64;
        let variance_b = 2.0 / (output_units) as f64;
        Dense {
            input: Tensor::new(vec![], vec![]),
            weights: Tensor::random_normal(vec![input_units, output_units], 0.0, variance_w, seed),
            biases: Tensor::random_normal(vec![1, output_units], 1.0, variance_b, seed),
            weights_grad: Tensor::new(vec![], vec![]),
            biases_grad: Tensor::new(vec![], vec![])
        }
    }
}

impl Layer for Dense {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("Dense"),
            output_shape: self.weights.shape.to_vec(),
            trainable_param: self.weights.shape.iter().product(),
            non_trainable_param: 0,
        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        // Perform an affine transformation:
        // f(x) = <W*x> + b
        
        // input shape: [batch, input_units]
        // output shape: [batch, output units]
        self.input = input;
        // panic!("input: {:?}", &self.input * &self.weights);

        &self.input * &self.weights + &self.biases
        // &self.input.dot(&self.weights) 