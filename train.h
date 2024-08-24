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

    std::tuple<Matrix<double>, Matrix<float>> readBatch() {
      Matrix<double> batch_dataset;
      Matrix<float> batch_label;
      // cout << "last read index: " << last_read_index << endl;

      for(int i=0; i < batch_size; i++) {
        if(last_read_index == dataset->train_indices.size()-1) break;

        // cout << "i: " << i << endl;
        // cout << "label size: " << label_reader.read_line_number(dataset->train_indices[last_read_index+1]).size() << endl;

        // batch_dataset.data.push_back(dataset_reader.read_line_number(dataset->train_indices[last_read_index+1]));
        // batch_label.data.push_back(label_reader.read_line_number(dataset->train_indices[last_read_index+1]));

        batch_dataset.data.push_back(dataset_reader.read_next_line());

        vector<double> y = label_reader.read_next_line();
        vector<float> y_f(y.size());

        std::transform(y.begin(), y.end(), y_f.begin(), [](double val) { return static_cast<float>(val); });

        batch_label.data.push_back(y_f);

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
      Matrix<double> y_onehot;

      dataset_reader.move_to_beginning_of_file();
      label_reader.move_to_beginning_of_file();

      for(int i=0; i < num_epochs; i++) {
        std::cout << "Training epoch " << i << " #####################" << std::endl;

        // cout << "last read index: " << last_read_index << " dtis: " << dataset->train_indices.size() << endl;
        cout << "training set size: " << dataset->train_indices.size() << endl;
        
        // std::tuple<Matrix<float>, Matrix<float>> data = readBatch();
        // y_onehot = dataset->int_to_onehot(std::get<1>(data)[0]);
        // cout << "batch size: " << batch_size << " y_onehot shape: " << y_onehot.shape() << endl;

        while(last_read_index < (int) dataset->train_indices.size()-1) {
          std::tuple<Matrix<double>, Matrix<float>> data = readBatch();
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
        float loss = model->loss_function(model->output_activations, y_onehot);
        // std::cout << "Loss after epoch " << i+1 << ": " << std::to_string(loss) << std::endl;
        losses.push_back(loss);

        dataset->shuffle_indices(dataset->train_indices);
      }
    }
};