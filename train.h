#pragma once

#include "neural_net.h"
#include "dataset.h"
#include "csv_reader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

struct Train {
  private:
    Dataset* dataset;
    CSVReader dataset_reader, label_reader;

    int last_read_index = -1;

  public:
  int num_epochs, batch_size;
  NeuralNetMLP* model;
  std::vector<int> training_data_line_numbers, test_data_line_numbers;
  std::vector<float> losses;

  Train(NeuralNetMLP* model, Dataset* dataset, int num_epochs, int batch_size=500, float train_test_split_size = 0.2): model(model),
  dataset(dataset), num_epochs(num_epochs), batch_size(batch_size) {
    CSVReader dataset_reader(dataset->dataset_filepath);
    CSVReader label_reader(dataset->labels_filepath);
  }

  void get_train_test_indices() {
    std::tuple<std::vector<int>, std::vector<int>> train_test_indices = dataset->train_test_split_indices(train_test_split_size);
    training_data_line_numbers = std::get<0>(train_test_indices);
    test_data_line_numbers = std::get<1>(train_test_indices);
  }

  std::tuple<Matrix<float>, Matrix<float>> readBatch() {
    Matrix<float> batch_dataset, batch_label;
    
    for(int i=0; i < batch_size; i++) {
      if(last_read_index == training_data_line_numbers.size()-1) break;

      batch_dataset.data.push_back(dataset_reader.read_line_number(training_data_line_numbers[last_read_index+1]));
      batch_label.data.push_back(label_reader.read_line_number(training_data_line_numbers[last_read_index+1]));

      last_read_index += 1;
    }

    batch_dataset.update_shape();
    batch_label.update_shape();

    return std::make_tuple(batch_dataset, batch_label);
  }

  void train(float learning_rate = 0.01) {
    Matrix<float> y_onehot;

    for(int i=0; i < num_epochs; i++) {
      std::cout << "Training epoch " << i << " #####################" << std::endl;
      get_train_test_indices();

      while(last_read_index < training_data_line_numbers.size()) {
        std::tuple<Matrix<float>, Matrix<float>> data = readBatch();
        y_onehot = dataset->int_to_onehot(std::get<1>(data)[0]);

        model->forward(std::get<0>(data));
        model->backward(std::get<0>(data), y_onehot, learning_rate);
      }

      last_read_index = -1;
      float loss = model->loss_function(model->output_activations, y_onehot);
      std::cout << "Loss after epoch " << i << ": " << std::to_string(loss) << std::endl;
      losses.push_back(loss);
    }
  }
};