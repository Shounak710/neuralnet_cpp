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
    std::vector<int> train_indices, test_indices;
    std::string dataset_filepath, labels_filepath;

    Dataset(const std::string& dataset_filepath, const std::string& labels_filepath, const char& delimiter=','): dataset_filepath(dataset_filepath),
    labels_filepath(labels_filepath), delimiter(delimiter) {
      CSVReader dataset_reader(dataset_filepath);
      Matrix<float> y = CSVReader(labels_filepath, delimiter).readCSV();
      
      num_classes = countUniqueValues(y);
      num_rows = y.data.size();
      num_features = dataset_reader.read_line_number(1).size();
    }

    // Matrix<float> int_to_onehot(std::vector<float> y) {
      
    //   Matrix<float> y_onehot(num_rows, num_classes);

    //   for(int i=0; i < y.size(); i++) {
    //     y_onehot[i][y[i]] = 1;
    //   }

    //   return y_onehot;
    // }

    static void shuffle_indices(vector<int>& indices) {
      std::random_device rd;
      std::mt19937 g(rd());

      std::shuffle(indices.begin(), indices.end(), g);
    }

    void train_test_split_indices(float test_size=0.2) {
      std::vector<int> indices;
      for(int i=0; i < num_rows; i++) { indices.push_back(i); }

      shuffle_indices(indices);

      size_t split_point = num_rows * (1 - test_size);

      train_indices = std::vector<int>(indices.begin(), indices.begin() + split_point);
      test_indices = std::vector<int>(indices.begin() + split_point, indices.end());
    }
};