#include "neural_net.cpp"
#include "matrix.h"
#include "dataset.h"
#include "train.h"
#include "csv_reader.h"

#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

const int N = 1000;
Matrix<int> A(N, N);
Matrix<int> B(N, N);
Matrix<int> C(N, N);

int main() {
  string dataset_filename="image_data.csv", labels_filename="labels.csv";

  // Matrix<double> a({{1.0, 2.0}, {2, 3}});
  // Matrix<double> b({{2.0, 1.0}, {1, 4}});

  // cout << a * b;
  // Matrix<Matrix<double>> m1({{a}, {b}});
  // Matrix<Matrix<double>> m2({{b}, {a}});

  // cout << "printing matrix of matrix" << endl;
  // for(auto i : m1.data) {
  //   for(auto j : i) {
  //     cout << (j.shape());
  //   }
  // }
  // cout << endl;

  // cout << "a: " << endl << a << endl;
  // cout << "b: " << endl << b << endl;
  
  // cout << a.el_mult(b) << endl;
  // cout << m1.el_mult(m2) << endl;

  // cout << "row mult: " << endl;
  // cout << a.row_mult(b) << endl;

  vector<int>layer_counts = {3, 5, 6, 7};

  Dataset ds(dataset_filename, labels_filename);

  // cout << "num classes: " << ds.num_classes << " num_features: " << ds.num_features << endl;

  NeuralNetMLP nn(ds.num_classes, ds.num_features, layer_counts, "softmax", "mse");

  Train t(&nn, &ds, 10, 50, 0.50);
  t.train(0.5);

  cout << "losses: " << endl;

  for(auto loss : t.losses) {
    cout << loss << endl;
  }
}