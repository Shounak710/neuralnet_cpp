#pragma once

#include "matrix.h"
#include "helper_functions.h"

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

    void go_to_line(unsigned int num){
      move_to_beginning_of_file();

      for(int i=0; i < num - 1; i++){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
      }
    }

  public:
    CSVReader(const std::string& filename, char delimiter=','): delimiter(delimiter), filename(filename), file(filename) {
      if (!file.is_open()) {
        throw std::runtime_error("Could not open the file: " + filename);
      }
    }

    ~CSVReader() {
      if(file.is_open()) file.close();
    }

    std::vector<double> read_line_number(size_t line_number) {
      if(line_number < 1) throw std::runtime_error("Line number should be greater than 0. Received: " + std::to_string(line_number));

      std::string line;

      go_to_line(line_number);
      getline(file, line);

      return split(line, delimiter);
    }

    std::vector<double> read_next_line() {
      std::string line;

      getline(file, line);
      return split(line, delimiter);
    }

    template<typename T = double>
    Matrix<T> readCSV() {
      std::vector<std::vector<T>> data;
      std::string line;

      while (getline(file, line)) {
        data.push_back(split<T>(line, delimiter));
      }

      return Matrix<T>(data);
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
      std::vector<double> row = split(line, delimiter);

      return row.size();
    }
};
