#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<random>

#include "matrix.h"

// Multi Layer perceptron NN for classification
class NeuralNetMLP {
  private:
    int num_features, num_classes;
    std::vector<int> num_hidden;

    double mean = 0.0, std = 1.0;
    unsigned int seed = 42;
    
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<double> dist;

    Matrix<float> initialize_weights(int row_count, int col_count) {
      Matrix<float> weights(row_count, col_count);

      for(int j=0; j < row_count; j++) {
        for(int k=0; k < col_count; k++) {
          weights[j][k] = dist(gen);
        }
      }

      return weights;
    }

    static Matrix<float> sigmoid_prime(Matrix<float> z);
    static Matrix<float> softmax_prime(Matrix<float> z);
    static Matrix<float> mse_loss_prime(Matrix<float> output_activations, Matrix<float> y_onehot);
    static Matrix<float> categorical_cross_entropy_loss_prime(Matrix<float> output_activations, Matrix<float> y_onehot);

  public:
    static Matrix<float> sigmoid(Matrix<float> z);
    static Matrix<float> softmax(Matrix<float> z);

    static float mse_loss(Matrix<float> output_activations, Matrix<float> y_onehot);
    static float categorical_cross_entropy_loss(Matrix<float> output_activations, Matrix<float> y_onehot);

    Matrix<float> int_to_onehot(std::vector<int> y);
    Matrix<float> biases_output, weights_output, output_weighted_inputs, output_activations;
    std::vector<Matrix<float>> biases_hidden, weights_hidden, hidden_weighted_inputs, hidden_activations;
    
    void forward(Matrix<float> x, Matrix<float> (*activation_function)(Matrix<float>));
    void backward(Matrix<float> x, Matrix<float> y_onehot, float learning_rate=0.01);

    Matrix<float> (*activation_function)(Matrix<float> z);
    Matrix<float> (*activation_function_prime)(Matrix<float> z);

    float (*loss_function)(Matrix<float> output_activations, Matrix<float> y_onehot);
    Matrix<float> (*loss_function_prime)(Matrix<float> output_activations, Matrix<float> y_onehot);

    NeuralNetMLP(int num_classes, int num_features, std::vector<int> num_hidden, std::string activation_type="sigmoid", std::string loss_type="mse"): num_classes(num_classes),
    num_features(num_features), num_hidden(num_hidden), dist(mean, std), gen(rd()) {

      if(activation_type == "softmax") {
        activation_function = &NeuralNetMLP::softmax;
        activation_function_prime = &NeuralNetMLP::softmax_prime;
      } else {
        activation_function = &NeuralNetMLP::sigmoid;
        activation_function_prime = &NeuralNetMLP::sigmoid_prime;
      }
      
      if(loss_type == "categorical_cross_entropy") {
        loss_function = &NeuralNetMLP::categorical_cross_entropy_loss;
        loss_function_prime = &NeuralNetMLP::categorical_cross_entropy_loss_prime;
      } else {
        loss_function = &NeuralNetMLP::mse_loss;
        loss_function_prime = &NeuralNetMLP::mse_loss_prime;
      }
      
      biases_hidden.resize(num_hidden.size());
      weights_hidden.resize(num_hidden.size());

      hidden_weighted_inputs.resize(num_hidden.size());
      hidden_activations.resize(num_hidden.size());

      int row_count, column_count;
      
      for(int i=0; i < num_hidden.size(); i++) {
        row_count = num_hidden[i];
        column_count = i==0 ? num_features : num_hidden[i-1];

        weights_hidden[i] = initialize_weights(row_count, column_count);
      }

      weights_output = initialize_weights(num_classes, num_hidden[num_hidden.size()-1]);
    }
};