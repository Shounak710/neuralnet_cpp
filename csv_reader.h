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

    inline std::vector<float> split(const std::string& str) {
        std::vector<float> tokens;
        std::string token;
        std::istringstream tokenStream(str);

        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(std::stof(token));
        }
        return tokens;
    }

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

    std::vector<float> read_line_number(size_t line_number) {
      if(line_number < 1) throw std::runtime_error("Line number should be greater than 0. Received: " + to_string(line_number));

      std::string line;

      go_to_line(line_number);
      getline(file, line);

      return split(line);    
    }

    std::vector<float> read_next_line() {
      std::string line;

      getline(file, line);
      return split(line);    
    }

    Matrix<float> readCSV() {
      std::vector<std::vector<float>> data;
      std::string line;

      while (getline(file, line)) {
        data.push_back(split(line));
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
      std::vector<float> row = split(line);

      return row.size();
    }
};
