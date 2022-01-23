
use crate::tensor::Tensor;
use std::fmt;

pub enum LearnableParams {
    Weights,
    Biases
}

#[derive(Debug)]
pub struct LayerInfo {
    pub layer_type: String,
    pub output_shape: Vec<usize>,
    pub trainable_param: usize,
    pub non_trainable_param: usize,
}

pub trait Layer {
    fn forward(&mut self, input: Tensor, training: bool) -> Tensor;
    fn backward(&mut self, gradient: &Tensor) -> Tensor;
    fn get_info(&self) -> LayerInfo;
    fn get_params_list(&self) -> Vec<LearnableParams>;
    fn get_param(&mut self, param: &LearnableParams) -> &mut Tensor;