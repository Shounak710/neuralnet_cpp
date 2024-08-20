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
    int countUniqueValues(const Matrix<float>& labels) {
      std::set<int> unique_values;

      for (vector<float> row : labels.data) {
        for(float value : row) {
          unique_values.insert(value);
        }
      }

      return unique_values.size();
    }
  
  public:
    char delimiter;
    int num_classes, num_rows, num_features;
    float test_size;
    std::string dataset_filepath, labels_filepath;

    Dataset() {}

    Dataset(const std::string& dataset_filepath, const std::string& labels_filepath, float test_size=0.2, const char& delimiter=','): dataset_filepath(dataset_filepath),
    labels_filepath(labels_filepath), test_size(test_size), delimiter(delimiter) {
      CSVReader dataset_reader(dataset_filepath);
      Matrix<float> y = CSVReader(labels_filepath, delimiter).readCSV();

      num_classes = countUniqueValues(y);
      num_rows = y[0].size();
      num_features = dataset_reader.read_line_number(1).size();
    }

    Matrix<float> int_to_onehot(std::vector<float> y) {
      
      Matrix<float> y_onehot(num_rows, num_classes);

      for(int i=0; i < y.size(); i++) {
        y_onehot[i][y[i]] = 1;
      }

      return y_onehot;
    }

    std::tuple<std::vector<int>, std::vector<int>> train_test_split_indices() {
      std::vector<size_t> indices;

      std::random_device rd;
      std::mt19937 g(rd());

      for(int i=0; i < num_rows; i++) { indices.push_back(i); }
      
      std::shuffle(indices.begin(), indices.end(), g);

      std::vector<int> train_indices, test_indices;
      size_t split_point = num_rows * (1 - test_size);

      train_indices = std::vector<int>(indices.begin(), indices.begin() + split_point);
      test_indices = std::vector<int>(indices.begin() + split_point, indices.end());

      return std::make_tuple(train_indices, test_indices);
    }
};