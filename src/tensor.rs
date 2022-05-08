
// Implement basic tensor structure

use crate::random::Rand;

use std::cmp;
use std::fmt;
use std::ops::{Add, Index, Mul, Sub, SubAssign, Div};
use std::f64::consts::PI;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new Tensor from `data` with the `shape` specified.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor { data, shape }
    }

    /// Creates a Tensor filled with zeroes with the `shape` specified.
    pub fn zero(shape: Vec<usize>) -> Tensor {
        Tensor {
            data: vec![0.0; shape.iter().product()],
            shape,
        }
    }

    /// Creates a Tensor filled with ones with the `shape` specified.
    pub fn one(shape: Vec<usize>) -> Tensor {
        Tensor {
            data: vec![1.0; shape.iter().product()],
            shape,
        }
    }

    /// Creates a Tensor filled with uniformly distributed random values
    /// between -1 and +1 with the `shape` specified.
    pub fn random(shape: Vec<usize>, seed: u32) -> Tensor {
        let mut rng = Rand::new(seed);

        let number_values = shape.iter().product();
        let data: Vec<f64> = (0..number_values).map(|_| rng.rand_float()).collect();
        Tensor { data, shape }