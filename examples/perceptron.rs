use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::Metric;
use newron::optimizers::sgd::SGD;

fn main() {
    // Let's create a toy dataset
    let dataset = Dataset::from_raw_data(vec![
        //   X_0, X_1, X_2, Y
        vec![1.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0,