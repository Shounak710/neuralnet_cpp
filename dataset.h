#pragma once

#include "csv_reader.h"
#include "mmap_csv_reader.h"
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
        for(double value : row) {
          unique_values.insert(value);
        }
      }

      return unique_values.size();
    }
  
  public:
    char delimiter;
    int num_classes, num_rows, num_features;
    std::vector<int> train_line_numbers, test_line_numbers;
    std::string dataset_filepath, labels_filepath;
    Matrix<float> y;

    Dataset(const std::string& dataset_filepath, const std::string& labels_filepath, const char& delimiter=','): dataset_filepath(dataset_filepath),
    labels_filepath(labels_filepath), delimiter(delimiter) {
      // CSVReader dataset_reader(dataset_filepath);
      MMapCSVReader dataset_reader(dataset_filepath);
      y = CSVReader(labels_filepath, delimiter).readCSV<float>();
      
      num_classes = countUniqueValues(y);
      num_rows = y.data.size();
      num_features = dataset_reader.read_line_number(1).size();
    }

    template<typename T=int>
    static void shuffle_vec(vector<T>& indices) {
      std::random_device rd;
      std::mt19937 g(rd());

      std::shuffle(indices.begin(), indices.end(), g);
    }

    void train_test_split_indices(float test_size=0.2) {
      std::vector<int> line_numbers;
      for(int i=1; i <= num_rows; i++) { line_numbers.push_back(i); }

      shuffle_vec(line_numbers);

      size_t split_point = num_rows * (1 - test_size);

      train_line_numbers = std::vector<int>(line_numbers.begin(), line_numbers.begin() + split_point);
      test_line_numbers = std::vector<int>(line_numbers.begin() + split_point, line_numbers.end());

      std::sort(train_line_numbers.begin(), train_line_numbers.end());
      std::sort(test_line_numbers.begin(), test_line_numbers.end());
    }
};