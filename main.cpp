#include "neural_net.cpp"
#include "matrix.h"
#include "dataset.h"
#include "train.h"
#include "csv_reader.h"

#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
  string dataset_filename="image_data.csv", labels_filename="labels.csv";
  // double res = -1.44045 * 0.731059 - 0.40998 * 0.5 + 0.0942367 * 0.5;
  // cout << "res: " << res << endl;

  // Matrix<double> wt({{-1.44045, -0.40998, 0.0942367}});
  // Matrix<double> act({{0.731059, 0.5, 0.5}, {1, 2, 3}, {0.6789, -0.5698, 0.5432}});

  // cout << "wt: " << endl << wt << endl;
  // cout << "wttr: " << endl << wt.transpose() << endl << endl;

  // cout << "act: " << endl << act << endl;
  // cout << "acttr: " << endl << act.Tr() << endl;
  // cout << "mult: " << wt * act.Tr() << endl;
  // cout << "col mean first: " << wt.Tr() * act.col_mean();

  // Matrix<Matrix<double>> n;
  // for(auto j : act.data) {
  //   n.data.push_back({wt.Tr() * Matrix<double>({j})});
  // }
  // n.update_shape();
  // cout << "m first then col mean: " << n.col_mean()[0][0] << endl;
  // Matrix<double> a({{1.0, 2.0}, {2, 3}});
  // Matrix<double> b({{3.0, 4.0}, {5, 6}});
  // cout << "a + b: " << endl << a + b << endl << endl;
  // cout << "a - b: " << endl << a - b << endl << endl;
  // cout << "a * b: " << endl << a * b << endl << endl;
  // cout << "a.el_mult(b): " << endl << a.el_mult(b) << endl << endl;
  // cout << "a.scalar_mult(c): " << endl << a.scalar_mult(3) << endl << endl;
  // cout << "a col mean: " << endl << a.col_mean() << endl << endl;
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
  
  // cout << a.scalar_mult(1000) << endl;
  // cout << m1.el_mult(m2) << endl;

  // cout << "row mult: " << endl;
  // cout << a.row_mult(b) << endl;

  // vector<int>layer_counts = {500, 250, 150, 70, 40, 20};
  // vector<int>layer_counts(200, 20);
  vector<int>layer_counts = {50};

  auto start = chrono::high_resolution_clock::now();

  Dataset ds(dataset_filename, labels_filename);

  cout << "start nn" << endl;
  NeuralNetMLP nn(ds.num_classes, ds.num_features, layer_counts, "sigmoid", "mse");
  cout << "end nn" << endl;

  int num_epochs = 150;
  Train t(&nn, &ds, num_epochs, 100, 0.15);
  cout << "mid train" << endl;
  t.train(0.1);
  cout << "end train" << endl;
  auto end = chrono::high_resolution_clock::now();
  
  chrono::duration<double> duration = end - start;

  cout << "Time taken for training model: " << duration.count() << " seconds" << endl;
  cout << "Average time taken per epoch: " << duration.count() / num_epochs << " seconds" << endl;

  cout << "losses: " << endl;

  for(auto loss : t.losses) {
    cout << loss << " ";
  }
  cout << endl;
}