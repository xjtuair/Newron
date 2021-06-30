use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::Metric;
use newron::optimizers::sgd::SGD;

fn main() {
    // Let's create a toy dataset
    let dataset = Dataset::from_raw_data(vec![
