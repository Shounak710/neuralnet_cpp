#include "neural_net.h"
#include "matrix.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

Matrix<double> NeuralNetMLP::int_to_onehot(Matrix<float> y) {
  Matrix<double> y_onehot(y.row_count, num_classes);

  for(int i=0; i < y.row_count; i++) {
    if(y[i].size() == 0) {
      cout << "y: ";
      for(auto d : y.data) {
        for(auto f : d) {
          cout << f << " ";
        }
      }
      cout << endl;
    }
    /* cout << "entered i: " << i << endl;
    cout << "y[i] size: " << y[i].size() << endl;
    cout << "y[i][0]: " << y[i][0] << endl;
    cout << "y_onehot shape: " << y_onehot.shape() << endl; */
    y_onehot[i][y[i][0]] = 1;
    // cout << "exited i: " << i << endl;
  }

  return y_onehot;
}

bool contains_nan(Matrix<double> m) {
    bool b = false;

    for(int i=0; i < m.data.size(); i++) {
      for(int j=0; j < m[i].size(); j++) {
        if(isnan(m[i][j]) || isinf(m[i][j])) {
          cout << "i: " << i << " j: " << j << " val: " << m[i][j] << " ";
          return true;
        }
      }
    }

    return b;
  }

void NeuralNetMLP::forward(Matrix<double> x) {
  Matrix<double> prev_weights;

  if(contains_nan(x)) cout << "x contains nan !" << endl;
  for(int i=0; i < num_hidden.size(); i++) {
    if(i==0) {
      prev_weights = x;
    } else {
      prev_weights = hidden_activations[i-1];
    }

    if((biases_hidden[i].row_count != prev_weights.row_count) || (biases_hidden[i].col_count != weights_hidden[i].row_count)) {
      biases_hidden[i] = Matrix<double>(prev_weights.row_count, weights_hidden[i].row_count);

      if(contains_nan(biases_hidden[i])) {
        cout << "i: " << i << endl;
        cout << "biases hidden: " << biases_hidden[i] << endl;

        throw std::runtime_error("fp Nan encountered at inner biases");
      }
    }

    hidden_weighted_inputs[i] = prev_weights * weights_hidden[i].Tr() + biases_hidden[i];
    hidden_activations[i] = activation_function(hidden_weighted_inputs[i]);
    
    if(contains_nan(hidden_weighted_inputs[i]) || contains_nan(hidden_activations[i])) {
      cout << "i: " << i << endl;
      cout << "hwi: " << hidden_weighted_inputs[i] << endl;
      cout << "hactv: " << activation_function(hidden_weighted_inputs[i]);

      throw std::runtime_error("fp Nan encountered at inner hwi and hactv");
    }
  }

  if((biases_output.row_count != hidden_activations[hidden_activations.size()-1].row_count) || (biases_output.col_count != weights_output.row_count)) {
    biases_output = Matrix<double>(hidden_activations[hidden_activations.size()-1].row_count, weights_output.row_count);

    if(contains_nan(biases_output)) {
      cout << "bo: " << biases_output << endl;

      throw std::runtime_error("fp Nan encountered at output biases");
    }
  }

  output_weighted_inputs = hidden_activations[hidden_activations.size()-1] * weights_output.Tr() + biases_output;

  if(contains_nan(output_weighted_inputs)) {
    cout << "weights output: " << weights_output.Tr() << endl;
    cout << "hal: " << hidden_activations[hidden_activations.size()-1];
    cout << "bo: " << biases_output << endl;
    cout << "owi: " << output_weighted_inputs << endl;

    throw std::runtime_error("fp Nan encountered at owi");
  }

  output_activations = activation_function(output_weighted_inputs);
  if(contains_nan(output_activations)) {
    cout << "owi: " << output_weighted_inputs << endl;
    cout << "oa: " << output_activations << endl;

    throw std::runtime_error("fp Nan encountered at oa");
  }
}

Matrix<double> NeuralNetMLP::calculate_delta(Matrix<double> y_onehot) {
  Matrix<double> delta(output_activations.row_count, output_activations.col_count);
  cout << "loss type: " << loss_type << endl;

  if((loss_type == "cce") || (loss_type == "categorical_cross_entropy")) {
    delta = output_activations - y_onehot;
  } else if(loss_type == "mse") {
    // for MSE loss
    if(activation_type == "sigmoid") {
      Matrix<double> sigm_deriv = sigmoid_prime(output_weighted_inputs);
      delta = (output_activations - y_onehot).el_mult(sigm_deriv);
    } else if(activation_type == "softmax") {
      for(int i=0; i < output_activations.row_count; i++) {
        double sum = 0.0;

        for(int j=0; j < output_activations.col_count; j++) {
          sum += (output_activations[i][j] - y_onehot[i][j]) * output_activations[i][j];
          
        }

        for(int k=0; k < output_activations.col_count; k++) {
          delta[i][k] = output_activations[i][k] * (output_activations[i][k] - y_onehot[i][k] - sum);
        }
      }
    }
  } else {
    throw std::runtime_error("Unsupported loss function type " + loss_type);
  }

  // delta = delta.col_mean();

  return delta;
}

void NeuralNetMLP::backward(Matrix<double> x, Matrix<double> y_onehot, float learning_rate) {
  // Matrix<double> delta =  (activation_function_prime(output_weighted_inputs)) * (loss_function_prime(output_activations, y_onehot));
  Matrix<double> delta = calculate_delta(y_onehot);
  if(contains_nan(delta)) {
    cout << "delta 1: " << delta << endl;
    cout << "output wt inp " << output_weighted_inputs << endl;
    cout << "output actv " << output_activations << endl;
    
    throw std::runtime_error("Nan encountered at delta 1");
  }

  biases_output = biases_output - delta.scalar_mult(learning_rate);
  weights_output = weights_output - (delta.Tr() * hidden_activations[hidden_activations.size()-1]).scalar_mult(learning_rate);

  if(contains_nan(biases_output) || contains_nan(weights_output)) {
    cout << "biases output: " << biases_output << endl;
    cout << "output wt " << weights_output << endl;
    cout << "last hddn actv " << hidden_activations[hidden_activations.size() - 1] << endl;
    
    throw std::runtime_error("Nan encountered at output WnB");
  }

  for(int i=weights_hidden.size() - 2; i >= 0; i--) {
    Matrix<double> weight;
    if(i == weights_hidden.size() - 2) {
      weight = weights_output;
    } else {
      weight = weights_hidden[i + 2];
    }

    if(contains_nan(weight)) {
      cout << "i: " << i << endl;
      cout << "wt: " << weight << endl;

      throw std::runtime_error("Nan encountered at inner wt");
    }

    delta = activation_function_prime(hidden_weighted_inputs[i+1]).Tr() * delta * weight;
    
    if(contains_nan(delta)) {
      cout << "i: " << i << endl;
      // cout << "delta inner: " << delta << endl;
      cout << "hwi: " << contains_nan(hidden_weighted_inputs[i+1]) << endl;
      cout << "ahwi: " << contains_nan(activation_function_prime(hidden_weighted_inputs[i+1]));

      throw std::runtime_error("Nan encountered at inner delta");
    }

    weights_hidden[i+1] = weights_hidden[i+1] - (delta.Tr() * hidden_activations[i]).scalar_mult(learning_rate);
    biases_hidden[i+1] = biases_hidden[i+1] - delta.scalar_mult(learning_rate);

    if(contains_nan(weights_hidden[i+1]) || contains_nan(biases_hidden[i+1])) {
      cout << "i: " << i << endl;
      cout << "delta inner: " << delta << endl;
      cout << "hact: " << hidden_activations[i] << endl;

      throw std::runtime_error("Nan encountered at inner WnB");
    }
  }
}

Matrix<double> NeuralNetMLP::sigmoid(Matrix<double> z) {
  Matrix<double> res(z.row_count, z.col_count);
  double max_el;

  for(int i=0; i < z.row_count; i++) {
    max_el = *max_element(z[i].begin(), z[i].end());

    for(int j=0; j < z.col_count; j++) {
      res[i][j] = (exp(z[i][j]) / (1 + exp(z[i][j])));

      if(isnan(res[i][j]) || isinf(res[i][j])) {
        res[i][j] = (exp(z[i][j] - max_el)) / (exp(-max_el) + exp(z[i][j] - max_el));
      }

      if(isnan(res[i][j]) || isinf(res[i][j])) {
        res[i][j] = 1;

        // cout << "nan encountered at res" << endl;
        // cout << "z: " << z[i][j] << endl;
        // cout << "max el: " << max_el << endl;
        // cout << "z - max_el: " << z[i][j] - max_el << endl;
        // cout << "res: " << res << endl;

        // throw std::runtime_error("sigmoid res nan");
      }
    }
  }

  return res;
}

Matrix<double> NeuralNetMLP::sigmoid_prime(Matrix<double> z) {
  Matrix<double> sigm = sigmoid(z);
  Matrix<double> diff = Matrix<double>(z.row_count, z.col_count, 1.0) - sigm;

  // cout << "sigm size: " << sigm.shape() << endl;
  // cout << "diff size: " << diff.shape() << endl;

  return sigm.el_mult(diff);
}

Matrix<double> NeuralNetMLP::softmax(Matrix<double> z) {
  Matrix<double> res(z.row_count, z.col_count);
  vector<double> sums(z.row_count, 0.0);
  double max_el;

  for(int i=0; i < z.row_count; i++) {
    max_el = *max_element(z[i].begin(), z[i].end());

    for(int j=0; j < z.col_count; j++) {
      res[i][j] = exp(z[i][j]-max_el);
      sums[i] += res[i][j];

      if(isnan(res[i][j]) || isinf(res[i][j])) {
        cout << "nan encountered at softmax res" << endl;
        cout << "z: " << z[i][j];
        cout << "max el: " << max_el;
        cout << "res: " << res[i][j];

        throw std::runtime_error("res nan");
      }
    }
  }

  for(int i=0; i < res.row_count; i++) {
    for(int j=0; j < res.col_count; j++) {
      res[i][j] /= sums[i];

      if(isnan(res[i][j]) || isinf(res[i][j])) {
        cout << "nan encountered at res / sum" << endl;
        cout << "z: " << z[i][j] << endl;
        cout << "exp(z): " << (long double) exp(z[i][j]) << endl;
        cout << "res: " << res[i][j];
        cout << "sum: " << sums[i];

        throw std::runtime_error("res nan");
      }
    }
  }

  return res;
}

Matrix<double> NeuralNetMLP::softmax_prime(Matrix<double> z) {
  cout << "softmax prime z dim: " << z.shape() << endl;
  Matrix<double> softm = softmax(z);
  Matrix<double> jacobian = Matrix<double>(z.row_count, z.row_count);

  for(int i=0; i < z.row_count; i++) {
    for(int j=0; j < z.col_count; j++) {
      if(i == j) {
        jacobian[i][j] = softm[i][0] * (1 - softm[i][0]);
      } else {
        jacobian[i][j] = -softm[i][0] * softm[j][0];
      }
    }
  }
  // cout << "softm size: " << sigm.shape() << endl;
  // cout << "diff size: " << diff.shape() << endl;

  return jacobian;
}

double NeuralNetMLP::categorical_cross_entropy_loss(Matrix<double> output_activations, Matrix<double> y_onehot) {
  const double epsilon = 1e-12;
  double sum = 0.0;

  for(int i=0; i < y_onehot.row_count; i++) {
    for(int j=0; j < y_onehot.col_count; j++) {
      /*cout << "y onehot ij: " << y_onehot[i][j] << endl;
      cout << "output act: " << output_activations[i][j] << endl;
      cout << "log oa: " << log(output_activations[i][j]) << endl;
      cout << "sum add: " << y_onehot[i][j] * log(max(epsilon, output_activations[i][j])) << endl;*/
      sum += y_onehot[i][j] * log(max(epsilon, output_activations[i][j]));
    }
  }

  if(contains_nan(y_onehot) || contains_nan(output_activations)) {
    cout << "oa: " << output_activations << endl;
    cout << "yonehot: " << y_onehot;

    throw std::runtime_error("fp Nan encountered at cce loss");
  }

  return -sum / (y_onehot.row_count * y_onehot.col_count);
}

Matrix<double> NeuralNetMLP::categorical_cross_entropy_loss_prime(Matrix<double> output_activations, Matrix<double> y_onehot) {
  return (output_activations - y_onehot);
}

double NeuralNetMLP::mse_loss(Matrix<double> output_activations, Matrix<double> y_onehot) {
  Matrix<double> loss = (y_onehot - output_activations);
  
  long double sum = 0.0;

  // cout << "mse loss" << endl;
  // cout << "loss matrix: " << output_activations << endl;

  for (const auto& row : loss.data) {
    for(const auto& el : row) {
      sum += el * el;
    }
  }

  // cout << loss << endl;

  if(contains_nan(loss)) {
    cout << "oa: " << output_activations << endl;
    cout << "yonehot: " << y_onehot;
    cout << "loss: " << loss << endl;

    throw std::runtime_error("fp Nan encountered at loss");
  }

  return sum / (loss.row_count * loss.col_count);
}

Matrix<double> NeuralNetMLP::mse_loss_prime(Matrix<double> output_activations, Matrix<double> y_onehot) {
  return (output_activations - y_onehot);
}