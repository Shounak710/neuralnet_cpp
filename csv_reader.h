#pragma once

#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVReader {
  private:
    const char delimiter;

    std::vector<std::string> split(const std::string& str) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

  public:
    CSVReader(char delimiter=','): delimiter(delimiter) {}

    Matrix<float> readCSV(const std::string& filename) {
      std::vector<std::vector<float>> data;
      std::ifstream file(filename);
      std::string line;

      if (!file.is_open()) {
          std::cerr << "Could not open the file: " << filename << std::endl;
          return data;
      }

      std::vector<float> res;

      while (getline(file, line)) {
        std::vector<std::string> row = split(line);
        if(res.size() != row.size()) res.resize(row.size());

        transform(row.begin(), row.end(), res.begin(), [](std::string& s) {
          return std::stof(s);
        });

        data.push_back(res);
      }
      
      file.close();
      return Matrix<float>(data);
    }
};
