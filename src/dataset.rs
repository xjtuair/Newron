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
    Feature, // column is a featur