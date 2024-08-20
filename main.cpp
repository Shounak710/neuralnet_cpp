#include "neural_net.cpp"
#include "matrix.h"
#include "dataset.h"
#include "csv_reader.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
  CSVReader reader(',');

  Dataset ds("image_data.csv", "labels.csv");

  // Matrix<float> y = reader.readCSV("image_data.csv");
  auto split_data = ds.train_test_split(0.5);
  Matrix<float> X_train = get<0>(split_data);
  Matrix<float> y_train = get<1>(split_data);

  Matrix<float> y_onehot = ds.int_to_onehot(y_train[0]);

  vector<int>layer_counts = {3, 5, 6, 7};

  NeuralNetMLP nn(y_onehot.col_count, ds.X.col_count, layer_counts, "softmax");

  cout << endl;
  for(int i=0; i < 10; i++) {
    cout << "i: " << i << endl;
    nn.forward(X_train);
    nn.backward(X_train, y_onehot, 0.01);

    cout << "Iteration " << i << " MSE Softmax Loss: " << nn.mse_loss(nn.output_activations, y_onehot) << endl;
  }
}