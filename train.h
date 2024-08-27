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
      vector<vector<double>> td = dataset_reader.getlines_from_mmap(dataset->train_line_numbers);

      for(int i=0; i < dataset->train_line_numbers.size(); i++) {
        training_data[dataset->train_line_numbers[i]] = td[i];
      }
    }

    std::tuple<Matrix<double>, Matrix<float>> readBatch() {
      Matrix<double> batch_dataset;
      Matrix<float> batch_label;

      for(int i=0; i < batch_size; i++) {
        int line_number = dataset->train_line_numbers[last_read_index+1];
        batch_dataset.data.push_back(training_data[line_number]);

        if(training_data[line_number].size() == 0) {
          throw std::runtime_error("empty data");
        }

        vector<float> y = dataset->y[line_number-1];

        batch_label.data.push_back(y);
        last_read_index += 1;
      }

      batch_dataset.update_shape();
      batch_label.update_shape();

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
        dataset->shuffle_vec(dataset->train_line_numbers);

        std::cout << "Training epoch " << i << " #####################" << std::endl;

        while(last_read_index < (int) dataset->train_line_numbers.size()-1) {
          std::tuple<Matrix<double>, Matrix<float>> data = readBatch();
          
          y_onehot = model->int_to_onehot(std::get<1>(data));
          
          model->forward(std::get<0>(data));
          
          model->backward(std::get<0>(data), y_onehot, learning_rate);
        }

        last_read_index = -1;

        double loss = model->loss_function(model->output_activations, y_onehot);

        losses.push_back(loss);
      }
    }
};