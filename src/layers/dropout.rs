use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Dropout {
    input: Tensor,
    prob: f64,
    // Store the seed so the Dropout struct can increment it
    // to generate new masks at each forward pass
    seed: u32, 
    mask: Tensor
}

impl Dropout {
    pub fn new(prob: f64, seed: u32) -> Dropout {
        // panic of prob is lesser than an arbitrary small value
        // since we use inverse dropout
        // (so we divide 1 by prob = infinity when prob is close to zero)
        
        if (1.0 - prob) < 0.01 {
            panic!("Dropout prob {} is to small to be computed efficiently !", prob);
        }
        
        Dropout {
            input: Tensor::new(vec![], vec![]),
            prob,
            seed,
            mask: Tensor::new(vec![], vec![])
        }
    }
}

impl Layer for Dropout {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("Dropout {:.2}%", self.prob*100.0),
            output_shape: vec![],
            trainable_param: 0,
            non_trainable_param: 0,
        }
    }

    fn forward(&mut s