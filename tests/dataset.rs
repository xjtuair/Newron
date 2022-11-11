#[cfg(test)]
mod dataset_tests {
    use newron::dataset::Dataset;
    use std::path::Path;
    #[test]
    // This test asserts a good implementation of
    // debug + display trait + loading from raw data
    fn test_dataset_debug_trait() {
        let dataset = Dataset::from_raw_data(vec![
            //   X_0, X_1, X_2, Y
            vec![1.0, 0.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
        ])
        .unwrap();

        let result = "X_0 | X_1 | X_2 | Y\n\
                      1   | 0   | 1   | 1\n\
                      0   | 1   | 