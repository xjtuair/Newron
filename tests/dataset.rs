#[cfg(test)]
mod dataset_tests {
    use newron::dataset::Dataset;
    use std::path::Path;
    #[test]
    // This test asserts a good implementation of
    // debug + display trait