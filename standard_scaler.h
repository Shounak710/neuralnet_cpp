#pragma once

#include "matrix.h"

#include <iostream>

template<typename T>
struct StandardScaler {
  Matrix<T> mean, stdev;

  void fit(Matrix<T> x) {
    mean = x.col_mean();

  }
};