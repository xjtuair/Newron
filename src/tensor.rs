
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
    }

    /// Generates a Tensor filled with random values following a normal distribution
    /// with parameters mu and sigma specified (mean/stdev)
    pub fn random_normal(shape: Vec<usize>, mean: f64, stdev: f64, seed: u32) -> Tensor {
        // We use the Box-Muller method to generate random normal values
        // Formula: sqrt(-2*ln(rand()))*cos(2*Pi*rand()) * stdev + mean
        let mut rng = Rand::new(seed);

        let number_values = shape.iter().product();

        let data: Vec<f64> = (0..number_values).map(|_| (
            // formula
            ((-2.0 * rng.rand_float().ln()).sqrt() * (2.0 * PI * rng.rand_float()).cos()) * stdev + mean
        )).collect();

        Tensor { data, shape }
    }

    pub fn mask(shape: &Vec<usize>, prob: f64, seed: u32) -> Tensor {
        let mut result = vec![];
        let number_values = shape.iter().product();

        for i in 0..number_values {
            let t = (prob * number_values as f64) as usize;
            if i < t {
                result.push(0.0);
            } else {
                result.push(1.0 / (1.0-prob));
            }
        }
        let mut rng = Rand::new(seed);
        rng.shuffle(&mut result[..]);

        Tensor::new(result, shape.to_vec())
    }


    /// Creates new matrix based on the transposed `self` Tensor
    pub fn get_transpose(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());

        for col in 0..self.shape[1] {
            for row in 0..self.shape[0] {
                data.push(self.get_value(row, col));
            }
        }

        Tensor {
            data,
            shape: vec![self.shape[1], self.shape[0]],
        }
    }

    /// Compute the mean of the matrix along the `axis` specified.
    /// 0 = along the column, 1 = along the row
    pub fn get_mean(&self, axis: usize) -> Tensor {
        // TODO: refactor with same logic as get_max (or better!)
        let mut data = Vec::new();

        let other_axis = if axis == 1 { 0 } else { 1 };

        // the wording is not quite exact here
        // row and col is indeed the row and col for axis == 0
        // but row become col and col become when axis == 1
        for row in 0..self.shape[other_axis] {
            let mut acc = 0.0;
            for col in 0..self.shape[axis] {
                if axis == 0 {
                    acc += self.get_value(col, row);
                } else {
                    acc += self.get_value(row, col);
                }
                
            }
            data.push(acc / self.shape[axis] as f64);
        }

        let shape = vec![1, data.len()];
        Tensor::new(data, shape)
    }

    /// Get the Tensor containing max values along the `axis` specified
    /// 0 = along the column, 1 = along the row
    pub fn get_max(&self, axis: usize) -> Tensor {
        let mut data = Vec::new();

        if axis == 0 {
            for row in 0..self.shape[1] {
                let mut max = self.get_value(0, row);
                for col in 0..self.shape[axis] {
                    let val = self.get_value(col, row);
                    if val > max {
                        max = val;
                    }
                }
                data.push(max);
            }
        } else {
            for col in 0..self.shape[0] {
                let mut max = self.get_value(col, 0);
                for row in 0..self.shape[axis] {
                    let val = self.get_value(col, row);
                    if val > max {
                        max = val;
                    }
                }
                data.push(max);
            }
        }

        let shape = if axis == 0 {vec![1, data.len()]} else {vec![data.len(), 1]};
        Tensor::new(data, shape)
    }

    /// Get the Tensor containing the sum of all values along the `axis` specified
    /// 0 = along the column, 1 = along the row
    pub fn get_sum(&self, axis: usize) -> Tensor {
        let mut data = Vec::new();

        if axis == 0 {
            for row in 0..self.shape[1] {
                let mut sum = 0.0;
                for col in 0..self.shape[axis] {
                    sum += self.get_value(col, row);
                }
                data.push(sum);
            }
        } else {
            for col in 0..self.shape[0] {
                let mut sum = 0.0;
                for row in 0..self.shape[axis] {
                    sum += self.get_value(col, row);
                }
                data.push(sum);
            }
        }

        let shape = if axis == 0 {vec![1, data.len()]} else {vec![data.len(), 1]};
        Tensor::new(data, shape)
    }

    /// Get 2d positioned value
    // 'data' is a flat array of f64
    pub fn get_value(&self, x: usize, y: usize) -> f64 {
        self.data[x * &self.shape[1] + y]
    }

    /// Get i-th row of the matrix. Return a new Tensor.
    /// Only for 2 dimensionals Tensor (matrix)
    pub fn get_row(&self, i: usize) -> Tensor {
        // TODO: add check for dimension
        Tensor {
            data: self.data[i * &self.shape[1]..(i + 1) * &self.shape[1]].to_vec(),
            shape: vec![1, self.shape[1]],
        }
    }

    /// Get all rows from a vector containing indices
    pub fn get_rows(&self, indices: &[usize]) -> Tensor {
        let mut data = Vec::new();
        for i in indices {
            data.extend(self.get_row(*i).data.iter());
        }