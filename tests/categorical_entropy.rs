#[cfg(test)]
mod categorical_entropy_tests {
    use newron::tensor::Tensor;
    use newron::loss::loss::Loss;
    use newron::loss::categorical_entropy::CategoricalEntropy;
    use newron::utils;

    #[test]
    fn test_categorical_entropy_loss() {
        let loss = CategoricalEntropy{};

        // Test 3 dimensions (batch = 3 samples)
        let predictions = Tensor::new(vec![0.2, 0.6, 0.2,
                                                         0.8, 0.2, 0.0,
                                                         1.0, 0.0, 0.0,
                                                         0.5, 0.5, 0.0,
  