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

    Matrix<double> initialize_weights(int row_count, int col_count) {
      Matrix<double> weights(row_count, col_count);

      for(int j=0; j < row_count; j++) {
        for(int k=0; k < col_count; k++) {
          weights[j][k] = dist(gen);
        }
      }

      return weights;
    }

    Matrix<double> calculate_delta(Matrix<double> y_onehot, size_t row_index);

    static Matrix<double> sigmoid_prime(Matrix<double> z);
    static Matrix<double> softmax_prime(Matrix<double> z);
    static Matrix<double> mse_loss_prime(Matrix<double> output_activations, Matrix<double> y_onehot);
    static Matrix<double> categorical_cross_entropy_loss_prime(Matrix<double> output_activations, Matrix<double> y_onehot);

  public:
    std::string loss_type, activation_type;

    static Matrix<double> sigmoid(Matrix<double> z);
    static Matrix<double> softmax(Matrix<double> z);

    static double mse_loss(Matrix<double> output_activations, Matrix<double> y_onehot);
    static double categorical_cross_entropy_loss(Matrix<double> output_activations, Matrix<double> y_onehot);

    Matrix<double> int_to_onehot(Matrix<float> y);
    Matrix<double> biases_output, weights_output, output_weighted_inputs, output_activations;
    std::vector<Matrix<double>> biases_hidden, weights_hidden, hidden_weighted_inputs, hidden_activations;
    
    void forward(Matrix<double> x);
    void backward(Matrix<double> x, Matrix<double> y_onehot, float learning_rate=0.01);
    void faster_backward(Matrix<double> x, Matrix<double> y_onehot, float learning_rate=0.01);

    Matrix<double> (*activation_function)(Matrix<double> z);
    Matrix<double> (*activation_function_prime)(Matrix<double> z);
    Matrix<double> delt_calc(Matrix<double> y_onehot);
    double (*loss_function)(Matrix<double> output_activations, Matrix<double> y_onehot);
    Matrix<double> (*loss_function_prime)(Matrix<double> output_activations, Matrix<double> y_onehot);

    NeuralNetMLP(int num_classes, int num_features, std::vector<int> num_hidden, std::string activation_type="sigmoid", std::string loss_type="mse"): num_classes(num_classes),
    num_features(num_features), num_hidden(num_hidden), activation_type(activation_type), loss_type(loss_type), dist(mean, std), gen(rd()) {

      if(activation_type == "softmax") {
        activation_function = &NeuralNetMLP::softmax;
        activation_function_prime = &NeuralNetMLP::softmax_prime;
      } else {
        activation_function = &NeuralNetMLP::sigmoid;
        activation_function_prime = &NeuralNetMLP::sigmoid_prime;
      }
      
      if((loss_type == "categorical_cross_entropy") || (loss_type == "cce")) {
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