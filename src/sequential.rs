
/// The Sequential model is a linear stack of layers.
use std::cmp;

use crate::layers::layer::Layer;
use crate::layers::*;
use crate::layers::LayerEnum;
use crate::metrics::Metric;
use crate::metrics::*;
use crate::tensor::Tensor;
use crate::dataset::{Dataset, RowType, ColumnType};
use crate::{loss::loss::Loss, random::Rand, optimizers::optimizer::OptimizerStep, optimizers::sgd::SGD};
use crate::loss::categorical_entropy::CategoricalEntropy;
use crate::utils;

struct Batch {
    inputs: Tensor,
    targets: Tensor
}

pub struct Sequential {
    pub layers_enum: Vec<LayerEnum>,
    pub layers: Vec<Box<dyn Layer>>,
    loss: Box<dyn Loss>,
    optim: Box<dyn OptimizerStep>,
    metrics: Vec<Metric>,
    seed: u32,
}

impl Sequential {
    /// Create a new empty Sequential model.
    pub fn new() -> Sequential {
        Sequential {
            layers_enum: vec![],
            layers: vec![],
            loss: Box::new(CategoricalEntropy{}),
            optim: Box::new(SGD::new(0.02)),
            metrics: vec![],
            seed: 0,
        }
    }

    /// Seed the random number generator
    pub fn set_seed(&mut self, s: u32) {
        self.seed = s;
    }

    /// Add a layer to the model
    pub fn add(&mut self, layer: LayerEnum) {
        self.layers_enum.push(layer);
    }



    /// Get a summary of the model
    pub fn summary(&self) {
        println!("Sequential model using {} layers.", self.layers.len());
        println!("
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================");

        let mut trainable_param_sum = 0;
        let mut non_trainable_param_sum = 0;

        for layer in &self.layers {
            let layer_info = layer.get_info(); 
            let layer_type = utils::fit_string_to_length(layer_info.layer_type, 29);
            let output_shape_str = "(".to_string() + &layer_info.output_shape
                        .iter()
                        .map(|u| u.to_string())
                        .collect::<Vec<_>>()
                        .join(", ") + ")";
            let output_shape = utils::fit_string_to_length(output_shape_str, 26);

            let trainable_param = utils::fit_string_to_length(layer_info.trainable_param.to_string(), 10);
            
            println!("{}{}{}", layer_type, output_shape, trainable_param);

            trainable_param_sum += layer_info.trainable_param;
            non_trainable_param_sum += layer_info.non_trainable_param;

        }
        let total_params = trainable_param_sum + non_trainable_param_sum;
        println!("=================================================================");
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", trainable_param_sum);
        println!("Non trainable params: {}", non_trainable_param_sum);

    }

    pub fn compile<T: 'static + Loss, U: 'static + OptimizerStep>(&mut self, loss: T, optim: U, metrics: Vec<Metric>) {
        // Set options
        self.loss = Box::new(loss);
        self.optim = Box::new(optim);
        self.metrics = metrics;

        // Build layers
        self.layers.clear();
        for layer in &self.layers_enum {
            self.layers.push(
                match layer {
                    LayerEnum::Dense { input_units, output_units } => {
                        Box::new(dense::Dense::new(*input_units, *output_units, self.seed))
                    }
                    LayerEnum::ReLU => {
                        Box::new(relu::ReLU::new())
                    }
                    LayerEnum::Softmax => {
                        Box::new(softmax::Softmax::new())
                    }
                    LayerEnum::TanH => {
                        Box::new(tanh::TanH::new())
                    }
                    LayerEnum::Sigmoid => {
                        Box::new(sigmoid::Sigmoid::new())
                    }
                    LayerEnum::Dropout { prob } => {
                        // Shape of Dropout is the same as last layer
                        Box::new(dropout::Dropout::new(*prob, self.seed))
                    }
                }
            );
        }
    }

    // Return the last layer output given an input
    fn forward_propagation(&mut self, input: Tensor, train: bool) -> Tensor {
        // Compute activations of all network layers by applying them sequentially.

        let mut activations: Vec<Tensor> = Vec::new();
        activations.push(input);
        
        // Iterate throught all layers, starting with `input`
        for layer in self.layers.iter_mut() {
            let activation = layer.forward(activations.last().unwrap().clone(), train);
            activations.push(activation);
        }

        assert_eq!(activations.len(), self.layers.len() + 1);
        activations.last().unwrap().clone()
    }

    fn backward_propagation(&mut self, gradient: Tensor) -> Tensor {
        let mut gradients = Vec::new();
        gradients.push(gradient);

        for layer in self.layers.iter_mut().rev() {
            let gradient = layer.backward(gradients.last().unwrap());
            gradients.push(gradient);
        }

        gradients.last().unwrap().clone()
    }

    /// Return a vector containing all batch
    /// if `shuffle` is set to true, batches are randomized
    fn get_batches(&mut self, dataset: &Dataset, batch_size: usize, shuffle: bool) -> Vec<Batch> {
        let x_train = dataset.get_tensor(RowType::Train, ColumnType::Feature); 
        let y_train = dataset.get_tensor(RowType::Train, ColumnType::Target);
        
        let mut indices = (0..x_train.shape[0]).collect::<Vec<usize>>();

        if shuffle {
            let mut rand = Rand::new(self.seed);
            rand.shuffle(&mut indices[..]);
            self.seed += 1;
        }

        let mut result = Vec::new();

        for batch_index in (0..x_train.shape[0]).rev().skip(batch_size - 1).step_by(batch_size).rev() {
            let batch_indices: &[usize] = &indices[batch_index..batch_index + batch_size];

            let x_batch = x_train.get_rows(batch_indices);
            let y_batch = y_train.get_rows(batch_indices);

            result.push(Batch {inputs: x_batch, targets: y_batch});