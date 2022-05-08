
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
