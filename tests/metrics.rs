#[cfg(test)]
mod metrics_tests {
    use newron::metrics::*;
    use newron::tensor::Tensor;
    use newron::utils;

    fn setup() -> (Tensor, Tensor) {
        let predictions: Tensor = Tensor::new(vec![0.4, 0.6, 
                                            