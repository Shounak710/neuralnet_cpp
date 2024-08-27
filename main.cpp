#include "neural_net.cpp"
#include "matrix.h"
#include "dataset.h"
#include "train.h"
#include "csv_reader.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
  string dataset_filename="image_data.csv", labels_filename="labels.csv";
  // CSVReader dataset_reader(dataset_filename);
  // CSVReader labels_reader(labels_filename);

  // labels_reader.read_line_number(45);
  vector<int>layer_counts = {3, 5, 6, 7};

  Dataset ds(dataset_filename, labels_filename);

  // cout << "first line" << endl;
  // cout << Matrix<float>({CSVReader(dataset_filename).read_line_number(1)}) << endl;

  cout << "num classes: " << ds.num_classes << " num_features: " << ds.num_features << endl;

  NeuralNetMLP nn(ds.num_classes, ds.num_features, layer_counts, "sigmoid", "mse");

  Train t(&nn, &ds, 5, 35, 0.5);
  t.train(0.001);

  cout << "losses: " << endl;

  for(auto loss : t.losses) {
    cout << loss << endl;
  }
  // // Matrix<float> y = reader.readCSV("image_data.csv");
  // auto split_data = ds.train_test_split(0.5);
  // Matrix<float> X_train = get<0>(split_data);
  // Matrix<float> y_train = get<1>(split_data);

  // Matrix<float> y_onehot = ds.int_to_onehot(y_train[0]);

  // ;

  // NeuralNetMLP nn(y_onehot.col_count, ds.X.col_count, layer_counts, "softmax");

  // cout << endl;
  // for(int i=0; i < 10; i++) {
  //   cout << "i: " << i << endl;
  //   nn.forward(X_train);
  //   nn.backward(X_train, y_onehot, 0.01);

  //   cout << "Iteration " << i << " MSE Softmax Loss: " << nn.mse_loss(nn.output_activations, y_onehot) << endl;
  // }
}