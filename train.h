#pragma once

#include "neural_net.h"
#include "dataset.h"
#include "mmap_csv_reader.h"
#include "csv_reader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

struct Train {
  private:
    Dataset* dataset;
    CSVReader label_reader;
    MMapCSVReader dataset_reader;
    unordered_map<int, vector<double>> training_data;

    int last_read_index = -1;
    float train_test_split_size;

    // void get_train_test_indices() {
    //   std::tuple<std::vector<int>, std::vector<int>> train_test_indices = dataset->train_test_split_indices(train_test_split_size);
    //   training_data_line_numbers = std::get<0>(train_test_indices);
    //   test_data_line_numbers = std::get<1>(train_test_indices);
    // }

    // Matrix<double> int_to_onehot(Matrix<float> y, int num_classes) {
    //   cout << "entered y onehot" << endl;
    //   cout << "num classes: " << num_classes << endl;
    //   Matrix<double> y_onehot(y.row_count, num_classes);

    //   for(int i=0; i < y.row_count; i++) {
    //     if((i > y_onehot.row_count-1)) cout << "Seg fault culprit row: " << i << " " << y_onehot.row_count << endl;
    //     if(((int) y[i][0] > y_onehot.col_count-1)) cout << "Seg fault culprit col: " << (int) y[i][0] << " " << y_onehot.col_count << endl;
    //     cout << "i: " << i << " y[i][0]: " << (int) y[i][0] << " row count: " << y_onehot.row_count << " col count: " << y_onehot.col_count << endl;
    //     cout << "el: " << y_onehot[i][(int) y[i][0]] << endl;
    //     y_onehot[i][(int) y[i][0]] = 1;
    //   }

    //   cout << "exited y onehot" << endl;
    //   return y_onehot;
    // }

    void load_training_data() {
      vector<vector<double>> td = dataset_reader.getlines_from_mmap(dataset->train_indices);

      for(int i=0; i < dataset->train_indices.size(); i++) {
        cout << "i: " << i << endl;
        training_data[dataset->train_indices[i] + 1] = td[i];
      }
    }

    std::tuple<Matrix<double>, Matrix<float>> readBatch() {
      Matrix<double> batch_dataset;
      Matrix<float> batch_label;
      // cout << "last read index: " << last_read_index << endl;

      for(int i=0; i < batch_size; i++) {
        int line_number = dataset->train_indices[last_read_index+1] + 1;
        cout << "i: " << i << endl;
        cout << "reading line number: " << line_number << endl;
        batch_dataset.data.push_back(training_data[line_number]);

        cout << "line number: " << line_number << endl;
        cout << "training data: " << training_data[line_number].size() << endl;

        if(training_data[line_number].size() == 0) {
          throw std::runtime_error("empty data");
        }

        vector<float> y = dataset->y[line_number-1];

        batch_label.data.push_back(y);
        last_read_index += 1;

        cout << "read line" << endl;
      }

      batch_dataset.update_shape();
      batch_label.update_shape();

      cout << "batch data set shape: " << batch_dataset.data[0].size() << endl;
      cout << "here" << endl;
      return std::make_tuple(batch_dataset, batch_label);
    }

  public:
    int num_epochs, batch_size;
    NeuralNetMLP* model;
    std::vector<int> training_data_line_numbers, test_data_line_numbers;
    std::vector<double> losses;

    Train(NeuralNetMLP* model, Dataset* dataset, int num_epochs, int batch_size=500, float train_test_split_size = 0.2): model(model),
    dataset(dataset), num_epochs(num_epochs), batch_size(batch_size), dataset_reader(dataset->dataset_filepath), label_reader(dataset->labels_filepath),
    train_test_split_size(train_test_split_size) {

      dataset->train_test_split_indices(train_test_split_size);
      load_training_data();
    }

    void train(float learning_rate = 0.01) {
      Matrix<double> y_onehot;

      for(int i=0; i < num_epochs; i++) {
        dataset->shuffle_indices(dataset->train_indices);

        std::cout << "Training epoch " << i << " #####################" << std::endl;

        // cout << "last read index: " << last_read_index << " dtis: " << dataset->train_indices.size() << endl;
        cout << "training set size: " << dataset->train_indices.size() << endl;
        
        // std::tuple<Matrix<float>, Matrix<float>> data = readBatch();
        // y_onehot = dataset->int_to_onehot(std::get<1>(data)[0]);
        // cout << "batch size: " << batch_size << " y_onehot shape: " << y_onehot.shape() << endl;

        while(last_read_index < (int) dataset->train_indices.size()-1) {
          std::tuple<Matrix<double>, Matrix<float>> data = readBatch();
          // last_read_index += batch_size;

          cout << "X shape: " << std::get<0>(data).shape() << "y shape: " << std::get<1>(data).shape() << endl;
          y_onehot = model->int_to_onehot(std::get<1>(data));
          // y_onehot = int_to_onehot(std::get<1>(data), dataset->num_classes);
          cout << "y onehot shape: " << y_onehot.shape() << endl;

          // std::cout << std::get<0>(data).shape() << std::endl;
          model->forward(std::get<0>(data));
          cout << "forward done" << endl;
          
          cout << "starting backpropagation" << endl;
          model->backward(std::get<0>(data), y_onehot, learning_rate);
          cout << "backward done" << endl;
        }

        last_read_index = -1;

        // cout << "activation size: " << model->output_activations.shape() << endl;
        double loss = model->loss_function(model->output_activations, y_onehot);
        // std::cout << "Loss after epoch " << i+1 << ": " << std::to_string(loss) << std::endl;
        losses.push_back(loss);

        dataset->shuffle_indices(dataset->train_indices);
      }
    }
};