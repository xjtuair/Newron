// Some utility functions
use crate::tensor::Tensor;

// Invert integer
pub(crate) fn swap_endian(val: u32) -> u32 {
    let result = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (result << 16) | (result >> 16);
}

// Convert an array of 4 u8 into a u32
pub(crate) fn as_u32_le(array: &[u8; 4]) -> u32 {
    ((array[0] as u32) <<  0) +
    ((array[1] as u32) <<  8) +
    ((array[2] as u32) << 16) +
    ((array[3] as u32) << 24)
}
