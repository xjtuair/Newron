pub mod layer;
pub mod relu;
pub mod tanh;
pub mod dense;
pub mod softmax;
pub mod sigmoid;
pub mod dropout;

pub enum LayerEnum {
    Dense {inpu