#[cfg(test)]
mod metrics_tests {
    use newron::metrics::*;
    use newron::tensor::Tensor;
    use newron::utils;

    fn setup() -> (Tensor, Tensor) {
        let predictions: Tensor = Tensor::new(vec![0.4, 0.6, 
                                                   0.1, 0.9, 
                                                   0.3, 0.7, 
                                                   1.0, 0.0], vec![4, 2]);

        let true_values: Tensor = Tensor::new(vec![0.4, 0.6, 
                                                   0.1, 0.9