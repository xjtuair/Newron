use std::fmt;
use std::path::Path;
use std::cmp;
use std::fs::File;
use std::io::{Read, BufReader, BufRead};
use std::str::FromStr;

use crate::tensor::Tensor;
use crate::{random::Rand, utils};

#[derive(PartialEq, Debug)]
pub enum ColumnType {
    Feature, // column is a feature used to train models
    Target,  // column is a target to predict
    Skip     // column not used by the model
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RowType {
    Train,  // row is used for training
    Test,   // row is preserved for test
    Skip    // row is ignored
}

#[derive(Debug)]
pub struct ColumnMetadata {
    name: String,
    column_type: ColumnType
}

#[derive(Debug)]
pub struct Row {
    data: Vec<f64>,
    row_type: Row