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

  /* Matrix<double> a({{1.0, 2.0}, {2, 3}});
  Matrix<double> b({{2.0, 1.0}, {1, 4}});

  Matrix<Matrix<double>> m1({{a}, {b}});
  Matrix<Matrix<double>> m2({{b}, {a}});

  cout << "printing matrix of matrix" << endl;
  for(auto i : m1.data) {
    for(auto j : i) {
      cout << (j.shape());
    }
  }
  cout << endl;

  cout << "m1: " << m1 << endl;
  cout << "m2: " << m2 << endl;
  cout << m1.el_mult(m2) << endl; */

  vector<int>layer_counts = {3, 5, 6, 7};

  Dataset ds(dataset_filename, labels_filename);

  cout << "num classes: " << ds.num_classes << " num_features: " << ds.num_features << endl;

  NeuralNetMLP nn(ds.num_classes, ds.num_features, layer_counts, "sigmoid", "cce");

  Train t(&nn, &ds, 10, 569, 0.5);
  t.train(0.01);

  cout << "losses: " << endl;

  for(auto loss : t.losses) {
    cout << loss << endl;
  }
}