#pragma once

#include "matrix.h"

#include <iostream>

template<typename T>
struct StandardScaler {
  Matrix<T> mean, stdev;

  void fit(const Matrix<T>& x) {
    mean = x.col_mean();

    std::vector<std::vector<T>> mean_data;
    for(auto m : mean) {
      mean_data.push_back(std::vector<T>(m, x.row_count));
    }

    stdev = Matrix<T>(mean_data).Tr();
    stdev = x - stdev;
    stdev = stdev.pow(2).col_mean();
    stdev = stdev.pow(0.5);
  }

  void transform(Matrix<T>& x) {
    for(i=0; i < x.col_count; i++) {
      T mos = mean[0][i] / stdev[0][i];

      for(int j=0; j < x.row_count; j++) {
        X[j][i] = (X[j][i] / stdev[0][i]) - mos;
      }
    }
  }
};