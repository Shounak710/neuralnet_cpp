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
    
    y_onehot[i][y[i][0]] = 1;
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

void NeuralNetMLP::align_matrix_dimensions(Matrix<double> &target_matrix, Matrix<double> &reference_matrix) {
  if((target_matrix.row_count != reference_matrix.row_count) || (target_matrix.col_count != reference_matrix.col_count)) {
    target_matrix = Matrix<double>(reference_matrix.row_count, reference_matrix.col_count);
  }
}

void NeuralNetMLP::forward(Matrix<double> x) {
  Matrix<double> prev_activations;

  if(contains_nan(x)) cout << "x contains nan !" << endl;
  for(int i=0; i < num_hidden.size(); i++) {
    if(i==0) {
      prev_activations = x;
    } else {
      prev_activations = hidden_activations[i-1];
    }

    hidden_weighted_inputs[i] = Matrix<double>(prev_activations.row_count, biases_hidden[i].row_count);
    align_matrix_dimensions(hidden_activations[i], hidden_weighted_inputs[i]);
    
    for(int j=0; j < prev_activations.row_count; j++) {
      hidden_weighted_inputs[i][j] = (weights_hidden[i] * Matrix<double>({prev_activations[j]}).Tr() + biases_hidden[i]).Tr()[0];
      // hidden_activations[i][j] = activation_function(Matrix<double>({hidden_weighted_inputs[i][j]}))[0];

      if(i > 0 && (j > 0)) {
        cout << "wt hidden " << i << ": " << endl << weights_hidden[i] << endl << endl;
        
        cout << "prev act: " << endl << Matrix<double>({prev_activations[j]}).Tr() << endl << endl;
        cout << "prod: " << endl << weights_hidden[i] * Matrix<double>({prev_activations[j]}).Tr() << endl << endl;
        cout << "bias " << i << ": " << endl << biases_hidden[i] << endl << endl;
        cout << "hwi: " << endl << (weights_hidden[i] * Matrix<double>({prev_activations[j]}).Tr() + biases_hidden[i]).Tr() << endl;
        cout << "hai: " << endl << activation_function((weights_hidden[i] * Matrix<double>({prev_activations[j]}).Tr() + biases_hidden[i]).Tr()) << endl;

        cout << "prev act: " << endl << Matrix<double>({prev_activations[j-1]}).Tr() << endl << endl;
        cout << "prod: " << endl << weights_hidden[i] * Matrix<double>({prev_activations[j-1]}).Tr() << endl << endl;
        cout << "bias " << i << ": " << endl << biases_hidden[i] << endl << endl;
        cout << "hwi: " << endl << Matrix<double>({hidden_weighted_inputs[i][j-1]}) << endl;
        cout << "hai: " << endl << activation_function((weights_hidden[i] * Matrix<double>({prev_activations[j-1]}).Tr() + biases_hidden[i]).Tr()) << endl;

        cout << "hwi compare: " << (hidden_weighted_inputs[i][j] == hidden_weighted_inputs[i][j-1]) << endl;
        cout << "pa compare: " << (prev_activations[j] == prev_activations[j-1]) << endl;
        cout << "hai compare: " << (hidden_activations[i][j] == hidden_activations[i][j-1]) << endl;

        throw std::runtime_error("checking");
      }
    }
    // if(i == 0) {
    //   cout << "prev act: " << prev_activations << endl;
    //   cout << "hwi: " << hidden_weighted_inputs[i] << endl;

    //   throw std::runtime_error("");
    // }
    hidden_activations[i] = activation_function(hidden_weighted_inputs[i]);
  }

  output_weighted_inputs = Matrix<double>(hidden_activations.back().row_count, biases_output.row_count);

  align_matrix_dimensions(output_activations, output_weighted_inputs);
  
  for(int k=0; k < hidden_activations.back().row_count; k++) {
    output_weighted_inputs[k] = (weights_output * Matrix<double>({hidden_activations.back()[k]}).Tr() + biases_output).Tr()[0];
    output_activations[k] = activation_function(Matrix<double>({output_weighted_inputs[k]}))[0];
  }
}

Matrix<double> NeuralNetMLP::calculate_delta(Matrix<double> y_onehot) const {
  Matrix<double> delta(output_activations.row_count, output_activations.col_count);
  // cout << "loss type: " << loss_type << endl;

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

  delta = delta.col_mean();

  return delta.Tr();
}

void NeuralNetMLP::backward(Matrix<double> x, Matrix<double> y_onehot, float learning_rate) {
  Matrix<double> delta = calculate_delta(y_onehot);
  
  cout << "delta shape: " << delta.shape() << endl;
  cout << "bo shape: " << biases_output.shape() << endl;
  cout << "wo shape: " << weights_output.shape() << endl;
  cout << "hab shape: " << hidden_activations.back().shape() << endl;

  biases_output = biases_output - delta.scalar_mult(learning_rate);

  Matrix<double> dha = (delta * hidden_activations.back().col_mean());

  weights_output = weights_output - dha.scalar_mult(learning_rate * delta.row_count);
  
  for(int i=weights_hidden.size() - 2; i >= 0; i--) {
    Matrix<double> weight;
    if(i == weights_hidden.size() - 2) {
      weight = weights_output;
    } else {
      weight = weights_hidden[i + 2];
    }

    Matrix<double> dw = weight.Tr() * delta;

    if(loss_type == "cce" && activation_type == "softmax") {
      delta = activation_function_prime(hidden_weighted_inputs[i+1]).Tr() * dw;
    } else {
      delta = activation_function_prime(hidden_weighted_inputs[i+1]).col_mean().Tr().el_mult(dw);
    }
    
    weights_hidden[i+1] = weights_hidden[i+1] - ((delta * (hidden_activations[i].col_mean()))).scalar_mult(learning_rate * delta.row_count);
    biases_hidden[i+1] = biases_hidden[i+1] - delta.scalar_mult(learning_rate);
  }
}

double NeuralNetMLP::compute_accuracy(Matrix<double> x, Matrix<float> y) {
  int correct_pred_count = 0, pred_label;

  forward(x);
  
  for(int i=0; i < output_activations.row_count; i++) {
    if(i > 0) {
      cout << "feature values equal to prev row ? row " << i << " " << (x[i] == x[i-1]) << " " << endl;
      cout << "output wt input values equal to prev row ? row " << i << " " << (output_weighted_inputs[i] == output_weighted_inputs[i-1]) << " " << endl;
      cout << "output act values equal to prev row ? row " << i << " " << (output_activations[i] == output_activations[i-1]) << " " << endl;
    }
    cout << "output act: " << endl;
    for(auto j : output_weighted_inputs[i]) cout << j << " ";
    for(auto j : output_activations[i]) cout << j << " ";
    cout << endl;

    auto max_it = std::max_element(output_activations[i].begin(), output_activations[i].end());
    pred_label = std::distance(output_activations[i].begin(), max_it);
    cout << "pred label: " << pred_label << " y: " << y[i][0] << " ";
    if(pred_label == y[i][0]) correct_pred_count++;
    cout << "verdict: " << (pred_label == y[i][0]) << endl;
  }

  return (double) correct_pred_count / output_activations.row_count;
}

Matrix<double> NeuralNetMLP::sigmoid(const Matrix<double>& z) {
  Matrix<double> res(z.row_count, z.col_count);
  double max_el;

  for(int i = 0; i < z.row_count; ++i) {
    max_el = *max_element(z[i].begin(), z[i].end());

    for(int j = 0; j < z.col_count; ++j) {
      double exp_neg = exp(-z[i][j]);
      res[i][j] = 1 / (1 + exp_neg);

      // If there are issues with extreme values, the adjustment can be applied directly:
      if (isnan(res[i][j]) || isinf(res[i][j])) {
          double exp_adj = exp(z[i][j] - max_el);
          res[i][j] = exp_adj / (exp_adj + exp(-max_el));
      }
    }
  }

  return res;
}

Matrix<double> NeuralNetMLP::sigmoid_prime(const Matrix<double>& z) {
  Matrix<double> sigm = sigmoid(z);
  Matrix<double> diff = Matrix<double>(z.row_count, z.col_count, 1.0) - sigm;

  // cout << "sigm size: " << sigm.shape() << endl;
  // cout << "diff size: " << diff.shape() << endl;

  return sigm.el_mult(diff);
}

Matrix<double> NeuralNetMLP::softmax(const Matrix<double>& z) {
  Matrix<double> res(z.row_count, z.col_count);
  vector<double> sums(z.row_count, 0.0);
  double max_el;

  for(int i=0; i < z.row_count; i++) {
    max_el = *max_element(z[i].begin(), z[i].end());

    for(int j=0; j < z.col_count; j++) {
      res[i][j] = exp(z[i][j]-max_el);
      sums[i] += res[i][j];

      // if(isnan(res[i][j]) || isinf(res[i][j])) {
      //   cout << "nan encountered at softmax res" << endl;
      //   cout << "z: " << z[i][j];
      //   cout << "max el: " << max_el;
      //   cout << "res: " << res[i][j];

      //   throw std::runtime_error("res nan");
      // }
    }
  }

  for(int i=0; i < res.row_count; i++) {
    for(int j=0; j < res.col_count; j++) {
      res[i][j] /= sums[i];

      // if(isnan(res[i][j]) || isinf(res[i][j])) {
      //   cout << "nan encountered at res / sum" << endl;
      //   cout << "z: " << z[i][j] << endl;
      //   cout << "exp(z): " << (long double) exp(z[i][j]) << endl;
      //   cout << "res: " << res[i][j];
      //   cout << "sum: " << sums[i];

      //   throw std::runtime_error("res nan");
      // }
    }
  }

  return res;
}

Matrix<double> NeuralNetMLP::softmax_prime(const Matrix<double>& z) {
  // cout << "softmax prime z dim: " << z.shape() << endl;
  Matrix<double> softm = softmax(z);
  
  Matrix<double> jacobian = Matrix<double>(z.col_count, z.col_count);

  for(int i=0; i < z.col_count; i++) {
    for(int j=0; j < z.col_count; j++) {
      if(i == j) {
        jacobian[i][j] = softm[0][j] * (1 - softm[0][j]);
      } else {
        jacobian[i][j] = -softm[0][i] * softm[0][j];
      }
    }
  }
  // cout << "jacobian calc" << endl;
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