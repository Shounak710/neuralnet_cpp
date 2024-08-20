#pragma once

#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVReader {
  private:
    char delimiter;
    std::string filename;
    std::ifstream file;

    inline std::vector<float> line_to_vec(std::string& line) {
      std::vector<float> res;

      std::vector<std::string> row = split(line);
      if(res.size() != row.size()) res.resize(row.size());

      transform(row.begin(), row.end(), res.begin(), [](std::string& s) {
        return std::stof(s);
      });

      return res;
    }

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
    // CSVReader() {}

    CSVReader(const std::string& filename, char delimiter=','): delimiter(delimiter), filename(filename), file(filename) {
      if (!file.is_open()) {
        throw std::runtime_error("Could not open the file: " + filename);
      }
    }

    ~CSVReader() {
      if(file.is_open()) file.close();
    }

    std::vector<float> read_line_number(size_t line_number) {
      std::vector<float> res;
      int count = 0;
      std::string line;

      move_to_beginning_of_file();

      while(count != line_number) {
        getline(file, line);
        count += 1;
      }

      return line_to_vec(line);    
    }

    Matrix<float> readCSV() {
      std::vector<std::vector<float>> data;
      std::string line;

      while (getline(file, line)) {
        data.push_back(line_to_vec(line));
      }

      return Matrix<float>(data);
    }

    inline void move_to_beginning_of_file() {
      file.clear();  // Clear the EOF flag
      file.seekg(0, std::ios::beg);  // Go to the beginning of the file
    }

    inline int get_line_count() {
      int count = 0;

      move_to_beginning_of_file();

      std::string line;

      while (getline(file, line)) count += 1;

      return count;
    }

    inline int get_col_count(const std::string& filename) {
      int size = 0;
      std::string line;

      move_to_beginning_of_file();

      getline(file, line);
      std::vector<std::string> row = split(line);

      return row.size();
    }
};
