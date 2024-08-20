#pragma once

#include "neural_net.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

struct Train {
  int num_epochs;
  NeuralNetMLP* model;

  Train(NeuralNetMLP* model, int num_epochs): model(model), num_epochs(num_epochs) {}

  std::vector<std::vector<float>> readBatch(std::ifstream& file, size_t batch_size) {
    std::vector<std::vector<float>> batch;
    std::string line;
    size_t count = 0;

    while (count < batch_size && std::getline(file, line)) {
      std::stringstream ss(line);
      std::string item;
      std::vector<float> row;

      while (std::getline(ss, item, ',')) {
          row.push_back(std::stof(item));
      }

      batch.push_back(row);
      count++;
    }

    return batch;
  }

  void processBatches(const std::string& filename, size_t batch_size) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return;
    }

    while (!file.eof()) {
      std::vector<std::vector<float>> batch = readBatch(file, batch_size);

      if (batch.empty()) {
        break;
      }

      // Process the batch
      std::cout << "Processing batch of size " << batch.size() << std::endl;
      for (const auto& row : batch) {
          for (float value : row) {
              std::cout << value << " ";
          }
          std::cout << std::endl;
      }
    }

    file.close();
  }
};