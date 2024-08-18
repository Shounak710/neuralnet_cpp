#include "neural_net.h"
#include "matrix.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

Matrix<float> NeuralNetMLP::int_to_onehot(vector<int> y) {
  Matrix<float> y_onehot(y.size(), num_classes);

  for(int i=0; i < y.size(); i++) {
    y_onehot[i][y[i]] = 1;
  }

  return y_onehot;
}

void NeuralNetMLP::forward(Matrix<float> x, Matrix<float> (*activation_function)(Matrix<float>)) {  
  Matrix<float> prev_weights;

  for(int i=0; i < num_hidden.size(); i++) {
    if(i==0) {
      prev_weights = x;
    } else {
      prev_weights = hidden_activations[i-1];
    }

    biases_hidden[i] = Matrix<float>(prev_weights.row_count, weights_hidden[i].row_count);

    hidden_weighted_inputs[i] = prev_weights * weights_hidden[i].Tr() + biases_hidden[i];
    hidden_activations[i] = activation_function(hidden_weighted_inputs[i]);
  }

  biases_output = Matrix<float>(hidden_activations[hidden_activations.size()-1].row_count, weights_output.row_count);

  output_weighted_inputs = hidden_activations[hidden_activations.size()-1] * weights_output.Tr() + biases_output;
  output_activations = activation_function(output_weighted_inputs);
}

void NeuralNetMLP::backward(Matrix<float> x, Matrix<float> y_onehot, float learning_rate) {
  Matrix<float> lp = loss_function_prime(output_activations, y_onehot);
  Matrix<float> delta =  (activation_function_prime(output_weighted_inputs)).el_mult(lp);

  /* cout << "initiating backpropagation" << endl;

  cout << "b_out: " << biases_output.shape();
  cout << " delta: " << delta.shape();

  cout << "delta Tr: " << delta.Tr().shape() << " w_out: " << weights_output.shape() << " w_hid: " << weights_hidden[weights_hidden.size() - 1].shape() << endl;
  */

  biases_output = biases_output - delta.scalar_mult(learning_rate);
  weights_output = weights_output - (delta.Tr() * weights_hidden[weights_hidden.size()-1].Tr()).scalar_mult(learning_rate);

  cout << "delta shape: " << delta.shape() << " biases_output_shape: " << biases_output.shape() << " weight shape: " << weights_output.shape() << endl;
  // cout << "output weight shape: " << weights_output.shape();

  for(int i=weights_hidden.size() - 2; i >= 0; i--) {
    Matrix<float> weight;
    if(i == weights_hidden.size() - 2) {
      weight = weights_output;
    } else {
      weight = weights_hidden[i + 2];
    }

    cout << "delta shape: " << delta.shape() << " weights_hidden_shape: " << hidden_weighted_inputs[i+1].shape() << " weight shape: " << weight.shape() << endl;
    delta = activation_function_prime(hidden_weighted_inputs[i+1]).Tr() * delta * weight;
    
    weights_hidden[i+1] = weights_hidden[i+1] - (delta.Tr() * hidden_activations[i]).scalar_mult(learning_rate);
    biases_hidden[i+1] = biases_hidden[i+1] - delta.scalar_mult(learning_rate);
    
    /* cout << "i: " << i << endl;
    cout << "delta shape: " << delta.shape();
    cout << " activation shape: " << hidden_activations[i].shape();
    cout << " weight shape: " << weights_hidden[i+1].shape() << endl; */
  }
}

Matrix<float> NeuralNetMLP::sigmoid(Matrix<float> z) {
  Matrix<float> res(z.row_count, z.col_count);

  for(int i=0; i < z.row_count; i++) {
    transform(z[i].begin(), z[i].end(), res[i].begin(), [](float x) {
      return (1.0 / (1 + exp(-x)));
    });
  }

  return res;
}

Matrix<float> NeuralNetMLP::sigmoid_prime(Matrix<float> z) {
  Matrix<float> sigm = sigmoid(z);
  Matrix<float> diff = Matrix<float>(z.row_count, z.col_count, 1.0) - sigm;

  cout << "sigm size: " << sigm.shape() << endl;
  cout << "diff size: " << diff.shape() << endl;

  return sigm.el_mult(diff);
}

Matrix<float> NeuralNetMLP::softmax(Matrix<float> z) {
  Matrix<float> res(z.row_count, z.col_count);
  float sum;

  for(int i=0; i < z.row_count; i++) {
    transform(z[i].begin(), z[i].end(), res[i].begin(), [](float x) {
      return exp(x);
    });

    sum = accumulate(res[i].begin(), res[i].end(), 0.0);

    transform(res[i].begin(), res[i].end(), res[i].begin(), [&sum](float x) {
      return x / sum;
    });
  }

  return res;
}

Matrix<float> NeuralNetMLP::softmax_prime(Matrix<float> z) {
  return Matrix<float>(1, 1);
}

float NeuralNetMLP::categorical_cross_entropy_loss(Matrix<float> output_activations, Matrix<float> y_onehot) {
  return 0.0;
}

Matrix<float> NeuralNetMLP::categorical_cross_entropy_loss_prime(Matrix<float> output_activations, Matrix<float> y_onehot) {
  return Matrix<float>({{0.0}});
}

float NeuralNetMLP::mse_loss(Matrix<float> output_activations, Matrix<float> y_onehot) {
  Matrix<float> loss = (y_onehot - output_activations);

  transform(loss.data.begin(), loss.data.end(), loss.data.begin(), [](vector<float> m) {
    transform(m.begin(), m.end(), m.begin(), [](float n) {
      return n * n;
    });

    return m;
  });

  return loss.mean();
}

Matrix<float> NeuralNetMLP::mse_loss_prime(Matrix<float> output_activations, Matrix<float> y_onehot) {
  return (output_activations - y_onehot);
}