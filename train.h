#pragma once

#include "neural_net.h"
#include "dataset.h"
#include "mmap_csv_reader.h"
#include "csv_reader.h"
#include "standard_scaler.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <chrono>

struct Train {
  private:
    Dataset* dataset;
    CSVReader label_reader;
    MMapCSVReader dataset_reader;
    std::unordered_map<int, std::vector<double>> training_data;

    bool scale_data;
    StandardScaler<double> ss;

    int last_read_index = -1;
    float train_test_split_size;

    void map_to_matrix(std::unordered_map<int, std::vector<double>>& training_data, Matrix<double>& mat) {
      for(auto it=training_data.begin(); it != training_data.end(); it++) {
        mat.data.push_back(it->second);
      }

      mat.update_shape();
    }

    void matrix_to_map(Matrix<double>& mat, std::unordered_map<int, std::vector<double>>& training_data) {
      // WARNING: This method assumes that matrix rows are in the same order as the map keys
      int index = 0;

      for(auto it=training_data.begin(); it != training_data.end(); it++) {
        it->second = mat[index];
        index++;
      }
    }

    void scale(std::unordered_map<int, std::vector<double>>& data, bool fit=true) {
      Matrix<double> scaled_data;
      map_to_matrix(data, scaled_data);

      std::cout << "begin transform" << std::endl;
      if(fit) {
        ss.fit(scaled_data);
        std::cout << "fitted" << std::endl;
      }
      ss.transform(scaled_data);
      std::cout << "end transform" << std::endl;

      // refill training data map with scaled data
      matrix_to_map(scaled_data, data);
    }

    void load_training_data() {
      std::cout << "loading training data" << std::endl;
      training_data = dataset_reader.getlines_from_mmap_thread(dataset->train_line_numbers);

      if(scale_data) scale(training_data);

      std::cout << "completed loading training data" << std::endl;
    }

    std::tuple<Matrix<double>, Matrix<float>> readBatch() {
      Matrix<double> batch_dataset;
      Matrix<float> batch_label;

      for(int i=0; i < batch_size; i++) {
        if(last_read_index == dataset->train_line_numbers.size()-1) break;

        int line_number = dataset->train_line_numbers[last_read_index+1];
        batch_dataset.data.push_back(training_data[line_number]);

        if(training_data[line_number].size() == 0) {
          std::cout << "no data on line number: " << line_number;
          throw std::runtime_error("empty data");
        }

        std::vector<float> y = dataset->y[line_number-1];

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
    std::vector<double> losses;

    Train(NeuralNetMLP* model, Dataset* dataset, int num_epochs, int batch_size=500, float train_test_split_size = 0.2, bool scale_data=true): model(model),
    dataset(dataset), num_epochs(num_epochs), batch_size(batch_size), dataset_reader(dataset->dataset_filepath), label_reader(dataset->labels_filepath),
    train_test_split_size(train_test_split_size), scale_data(scale_data) {
      dataset->train_test_split_indices(train_test_split_size);

      auto start = std::chrono::high_resolution_clock::now();
      load_training_data();
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      std::cout << "Time taken for loading data " << duration.count() << " seconds" << endl;
    }

    void train(float learning_rate = 0.01) {
      Matrix<double> y_onehot;

      for(int i=0; i < num_epochs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        dataset->shuffle_vec(dataset->train_line_numbers);

        std::cout << "Training epoch " << i << " #####################" << std::endl;

        while(last_read_index < (int) dataset->train_line_numbers.size()-1) {
          std::tuple<Matrix<double>, Matrix<float>> data = readBatch();
          y_onehot = model->int_to_onehot(std::get<1>(data));

          // auto start_frwrd = std::chrono::high_resolution_clock::now();
          model->forward(std::get<0>(data));
          // auto end_frwrd = std::chrono::high_resolution_clock::now();
          // std::chrono::duration<double> duration_frwrd = end_frwrd - start_frwrd;
          // std::cout << "Time taken for forward propagation: " << duration_frwrd.count() << " seconds" << endl;

          // auto start_bcwrd = std::chrono::high_resolution_clock::now();
          model->backward(std::get<0>(data), y_onehot, learning_rate);
          // auto end_bcwrd = std::chrono::high_resolution_clock::now();
          // std::chrono::duration<double> duration_bcwrd = end_bcwrd - start_bcwrd;
          // std::cout << "Time taken for backward propagation: " << duration_bcwrd.count() << " seconds" << endl;
        }

        last_read_index = -1;

        double loss = model->loss_function(model->output_activations, y_onehot);

        losses.push_back(loss);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Time taken for training epoch " << i + ": " << duration.count() << " seconds" << endl;
      }

      print_train_and_test_accuracy();
    }

    void print_train_and_test_accuracy() {
      Matrix<double> X_train, X_test;
      Matrix<float> y_train, y_test;

      for(int i=0; i < dataset->train_line_numbers.size(); i++) {
        X_train.data.push_back(training_data[dataset->train_line_numbers[i]]);
        y_train.data.push_back({dataset->y[dataset->train_line_numbers[i]-1]});
      }

      X_train.update_shape();
      y_train.update_shape();

      std::unordered_map<int, vector<double>> test_data = dataset_reader.getlines_from_mmap_thread(dataset->test_line_numbers);
      if(scale_data) scale(test_data, false);
      
      map_to_matrix(test_data, X_test);

      for(int j=0; j < dataset->test_line_numbers.size(); j++) {
        y_test.data.push_back({dataset->y[dataset->test_line_numbers[j]-1]});
      }

      y_test.update_shape();

      double train_acc = model->compute_accuracy(X_train, y_train);
      double test_acc = model->compute_accuracy(X_test, y_test);
      cout << "Training accuracy: " << train_acc << endl;
      cout << "Test accuracy: " << test_acc << endl;
    }
};