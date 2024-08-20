#pragma once

#include "csv_reader.h"
#include "matrix.h"

#include <iostream>
#include <tuple>
#include <algorithm>
#include <random>
#include <vector>
#include <set>

struct Dataset {
  private:
    void shuffle_X_y() {
      std::vector<size_t> indices(y.data.size());
      for(size_t i=0; i < indices.size(); i++) indices[i] = i;

      std::random_device rd;
      std::mt19937 g(rd());
      
      std::shuffle(indices.begin(), indices.end(), g);

      Matrix<float> X_copy(X.row_count, X.col_count), y_copy(y.row_count, y.col_count);

      for(int i=0; i < indices.size(); i++) {
        X_copy[i] = X[indices[i]];
        y_copy[i] = y[indices[i]];
      }

      X = X_copy;
      y = y_copy;
    }

    int countUniqueValues(const Matrix<float>& labels) {
      // Create a set to store unique values
      std::set<int> unique_values;

      // Insert each value from the vector into the set
      for (vector<float> row : labels.data) {
        for(float value : row) {
          unique_values.insert(value);
        }
      }

      // The size of the set is the count of unique values
      return unique_values.size();
    }
  
  public:
    char delimiter;
    int num_classes;
    std::string dataset_filepath, labels_filepath;
    Matrix<float> X, y;

    Dataset(const std::string& dataset_filepath, const std::string& labels_filepath, const char& delimiter=','): dataset_filepath(dataset_filepath),
    labels_filepath(labels_filepath), delimiter(delimiter) {
      get_data();
    }

    void get_data() {
      X = CSVReader(dataset_filepath, delimiter).readCSV();
      y = CSVReader(labels_filepath, delimiter).readCSV();

      num_classes = countUniqueValues(y);
    }

    Matrix<float> int_to_onehot(std::vector<float> y) {
      
      Matrix<float> y_onehot(y.size(), num_classes);

      for(int i=0; i < y.size(); i++) {
        y_onehot[i][y[i]] = 1;
      }

      return y_onehot;
    }

    std::tuple<Matrix<float>, Matrix<float>, Matrix<float>, Matrix<float>> train_test_split(float test_size=0.2) {
      shuffle_X_y();

      Matrix<float> X_train, y_train, X_test, y_test;
      size_t split_point = X_train.row_count * (1 - test_size);

      X_train = Matrix<float>(std::vector<std::vector<float>>(X.data.begin(), X.data.begin() + split_point));
      X_test = Matrix<float>(std::vector<std::vector<float>>(X.data.begin() + split_point, X.data.end()));

      y_train = Matrix<float>(std::vector<std::vector<float>>(y.data.begin(), y.data.begin() + split_point));
      y_test = Matrix<float>(std::vector<std::vector<float>>(y.data.begin() + split_point, y.data.end()));

      return std::make_tuple(X_train, y_train, X_test, y_test);
    }
};