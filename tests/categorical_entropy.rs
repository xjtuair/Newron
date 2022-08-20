#[cfg(test)]
mod categorical_entropy_tests {
    use newron::tensor::Tensor;
    use newron::loss::loss::Loss;
    use newron::loss::categorical_entropy::CategoricalEntropy;
    use newron::utils;

    #[test]
    fn test_categorical_entropy_loss() {
        let loss = CategoricalEntropy{};

        // Test 3 d