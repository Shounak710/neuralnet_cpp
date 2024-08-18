#include "neural_net.cpp"
#include "matrix.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int>layer_counts = {3, 5, 6, 7};
  NeuralNetMLP nn(4, 3, layer_counts);

  vector<vector<float>> data = {
    {3.0, 4.0, 5.0},
    {5.0, 6.0, 7.0},
    {6.0, 7.0, 8.0},
    {7.0, 8.0, 9.0},
    {8.0, 9.0, 10.0},
    {9.0, 10.0, 11.0}
  };

  vector<int> y = {0, 1, 0, 2, 3, 2};
  Matrix<float> y_onehot = nn.int_to_onehot(y);
  cout << y_onehot << endl;

  Matrix<float> x(data);

  cout << "sigmoid res: " << endl;
  nn.forward(x);

  cout << "Output Activations: " << endl;
  nn.output_activations.shape();

  cout << "Output Biases: " << endl;
  nn.biases_output.shape();

  cout << "Hidden Activations: " << endl;
  for(auto i : nn.hidden_activations) {
    i.shape();
  }
  cout << endl;

  cout << "Hidden Biases: " << endl;
  for(auto i : nn.biases_hidden) {
    i.shape();
  }
  cout << endl;


  cout << nn.output_activations << endl;
  cout << nn.hidden_activations[0] << endl;

  cout << "softmax res: " << endl;
  nn.forward(x);
  cout << nn.hidden_activations[0] << endl;

  cout << nn.output_activations << endl;

  cout << "Output Activations: " << endl;
  nn.output_activations.shape();

  cout << "Output Biases: " << endl;
  nn.biases_output.shape();

  cout << "Hidden Activations: " << endl;
  for(auto i : nn.hidden_activations) {
    i.shape();
  }
  cout << endl;

  cout << "Hidden Biases: " << endl;
  for(auto i : nn.biases_hidden) {
    i.shape();
  }
  cout << endl;

  cout << "activations - yonehot" << endl;
  cout << nn.output_activations - y_onehot << endl;

  nn.output_activations.shape();
  nn.weights_output.shape();
  // cout << nn.weights_output.Tr() * x + nn.biases_output << endl;

  Matrix<float> zs = nn.hidden_activations[nn.hidden_activations.size()-1] * nn.weights_output.Tr() + nn.biases_output;
  cout << "zs: " << zs << endl;
  cout << "sigmoid: " << nn.sigmoid(zs) << endl;

  Matrix<float> delta = (nn.output_activations - y_onehot).Tr() * (nn.sigmoid(zs) * nn.sigmoid(Matrix<float>(zs.row_count, zs.col_count, 1.0) - zs).Tr());
  cout << delta << endl;

  cout << "hidden activation last: " << nn.hidden_activations[nn.hidden_activations.size() - 1] << endl;

  // Matrix<float> new_weights = delta * nn.hidden_activations[nn.hidden_activations.size() - 1];
  // new_weights.shape();

  // nn.weights_output.shape();

  // delta = nn.weights_hidden[nn.weights_hidden.size()-1] * delta.transpose();
  // (delta * nn.hidden_activations[nn.hidden_activations.size() - 2]).shape();
  // cout << endl << endl;
  // nn.weights_hidden[nn.weights_hidden.size() - 1].shape();

  nn.backward(x, y_onehot);

  NeuralNetMLP test(4, 3, layer_counts, "softmax");

  cout << endl;
  for(int i=0; i < 10; i++) {
    test.forward(x);
    test.backward(x, y_onehot, 0.01);

    cout << "Iteration " << i << " MSE Loss: " << test.mse_loss(test.output_activations, y_onehot) << endl;
  }
}