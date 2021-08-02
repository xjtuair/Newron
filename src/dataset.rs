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
    row_type: RowType
}

#[derive(Debug)]
pub enum DatasetError {
    FileNotFound,
    BadFormat(String),
}

/// Use `Dataset` to load your dataset and train
/// a model on it.
pub struct Dataset {
    // Contains all data for dataset
    data: Vec<Row>,
    columns_metadata: Vec<ColumnMetadata>
}

impl Dataset {
    /// Load a dataset from a Vector of Vector of floats.
    /// By default, the last colunm is use as a target and the others as
    /// training features. Use `set_train_cols` and `set_target_cols` to
    /// change this behaviour.
    /// Header is automatically generated : 'X_0', 'X_1', ..., 'Y'. Use
    /// `set_header` to change it.
    pub fn from_raw_data(data: Vec<Vec<f64>>) -> Result<Dataset, DatasetError> {
        let cols = data[0].len();

        let mut columns_metadata = Vec::new();

        // test that all rows in 'data' have equal lengths
        if data.iter().any(|ref v| v.len() != data[0].len()) {
            return Err(DatasetError::BadFormat(format!("All rows must have equal lengths.")));
        }

        // iterate through training features
        for i in 0..cols - 1 {
            columns_metadata.push(ColumnMetadata {name: format!("X_{