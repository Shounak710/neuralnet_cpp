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

    // void get_train_test_indices() {
    //   std::tuple<std::vector<int>, std::vector<int>> train_test_indices = dataset->train_test_split_indices(train_test_split_size);
    //   training_data_line_numbers = std::get<0>(train_test_indices);
    //   test_data_line_numbers = std::get<1>(train_test_indices);
    // }

    Matrix<float> int_to_onehot(Matrix<float> y, int num_classes) {
      
      Matrix<float> y_onehot(y.row_count, num_classes);

      for(int i=0; i < y.row_count; i++) {
        y_onehot[i][y[i][0]] = 1;
      }

      return y_onehot;
    }

    std::tuple<Matrix<float>, Matrix<float>> readBatch() {
      Matrix<float> batch_dataset, batch_label;
      // cout << "last read index: " << last_read_index << endl;

      for(int i=0; i < 1; i++) {
        if(last_read_index == dataset->train_indices.size()-1) break;

        cout << "i: " << i << endl;
        // cout << "label size: " << label_reader.read_line_number(dataset->train_indices[last_read_index+1]).size() << endl;

        // batch_dataset.data.push_back(dataset_reader.read_line_number(dataset->train_indices[last_read_index+1]));
        // batch_label.data.push_back(label_reader.read_line_number(dataset->train_indices[last_read_index+1]));

        auto line_1 = dataset_reader.read_next_line();
        cout << "line" << endl;
        if(i==0) {
          for(auto i : line_1) cout << i << " ";
        }
        cout << endl;

        batch_dataset.data.push_back(line_1);
        batch_label.data.push_back(label_reader.read_next_line());

        last_read_index += 1;
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
    std::vector<float> losses;

    Train(NeuralNetMLP* model, Dataset* dataset, int num_epochs, int batch_size=500, float train_test_split_size = 0.2): model(model),
    dataset(dataset), num_epochs(num_epochs), batch_size(batch_size), dataset_reader(dataset->dataset_filepath), label_reader(dataset->labels_filepath) {

      dataset->train_test_split_indices(train_test_split_size);
    }

    void train(float learning_rate = 0.01) {
      Matrix<float> y_onehot;

      dataset_reader.move_to_beginning_of_file();
      label_reader.move_to_beginning_of_file();

      for(int i=0; i < num_epochs; i++) {
        std::cout << "Training epoch " << i << " #####################" << std::endl;

        // cout << "last read index: " << last_read_index << " dtis: " << dataset->train_indices.size() << endl;
        cout << "training set size: " << dataset->train_indices.size() << endl;
        
        // std::tuple<Matrix<float>, Matrix<float>> data = readBatch();
        // y_onehot = dataset->int_to_onehot(std::get<1>(data)[0]);
        // cout << "batch size: " << batch_size << " y_onehot shape: " << y_onehot.shape() << endl;

        while(last_read_index < (int) dataset->train_indices.size()) {
          std::tuple<Matrix<float>, Matrix<float>> data = readBatch();
          cout << "X shape: " << std::get<0>(data).shape() << "y shape: " << std::get<1>(data).shape() << endl;
          y_onehot = int_to_onehot(std::get<1>(data), dataset->num_classes);
          cout << "y onehot shape: " << y_onehot.shape() << endl;

          // std::cout << std::get<0>(data).shape() << std::endl;
          model->forward(std::get<0>(data));
          model->backward(std::get<0>(data), y_onehot, learning_rate);
        }

        last_read_index = -1;
        float loss = model->loss_function(model->output_activations, y_onehot);
        std::cout << "Loss after epoch " << i << ": " << std::to_string(loss) << std::endl;
        losses.push_back(loss);

        dataset->shuffle_indices(dataset->train_indices);
      }
    }
};