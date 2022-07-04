// Some utility functions
use crate::tensor::Tensor;

// Invert integer
pub(crate) fn swap_endian(val: u32) -> u32 {
    let result = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (result << 16) | (result >> 16);
}

// Convert an array of 4 u8 into a u32
pub(crate) fn as_u32_le(ar