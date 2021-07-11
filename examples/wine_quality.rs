use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::optimizers::sgd::SGD;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::Metric;

fn main() {
    let mut dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();
    dataset.split_train_test(0.8, true);

    println!("{:?}", dataset);