#!/bin/bash
cd ./datasets

# Download Wine Quality (white) dataset
wine_url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv";
curl -X GET $wine_url > ./winequality-white.csv;

# Download MNIST Fashion dataset
training_images="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
training_labels="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
test_images="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
test_labels="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t