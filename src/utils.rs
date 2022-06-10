// Some utility functions
use crate::tensor::Tensor;

// Invert integer
pub(crate) fn swap_endian(val: u32) -> u32 {
    let result = ((val << 8) & 