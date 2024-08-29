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

  vector<int>layer_counts = {3, 5, 6, 7};

  Dataset ds(dataset_filename, labels_filename);

  cout << "num classes: " << ds.num_classes << " num_features: " << ds.num_features << endl;

  NeuralNetMLP nn(ds.num_classes, ds.num_features, layer_counts, "softmax", "cce");

  Train t(&nn, &ds, 10, 3500, 0.85);
  t.train(0.1);

  cout << "losses: " << endl;

  for(auto loss : t.losses) {
    cout << loss << endl;
  }
}