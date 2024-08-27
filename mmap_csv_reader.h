#pragma once

#include "helper_functions.h"

#include<iostream>
#include<sys/mman.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/stat.h>
#include<cstring>
#include<string>

class MMapCSVReader {
  const char *filepath, delimiter;
  char *data;
  int fd;
  size_t filesize;

  public:
    MMapCSVReader(const std::string filepath, const char delimiter=','): filepath(filepath.c_str()), delimiter(delimiter) {
      int fd = open(this->filepath, O_RDONLY);
      if(fd == -1) throw std::runtime_error("Error opening file: " + std::string(filepath));
      
      struct stat sb;
      if(fstat(fd, &sb) == -1) throw std::runtime_error("Error getting filesize");

      filesize = sb.st_size;

      data = static_cast<char*>(mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
      if(data == MAP_FAILED) throw std::runtime_error("Error mapping file"); 
    }

    ~MMapCSVReader() {
      if(munmap(data, filesize) == -1) {
        std::cerr << "Error unmapping file" << std::endl;
      }

      close(fd);
    }
  
    std::string getline_from_mmap(int line_number) {
      int current_line = 1;
      const char *line_start = data;
      const char *line_end = nullptr;

      for (size_t i = 0; i < filesize; ++i) {
        if (data[i] == '\n' || i == filesize - 1) {
          line_end = &data[i];
          if (current_line == line_number) {
              return std::string(line_start, line_end - line_start);
          }
          current_line++;
          line_start = &data[i + 1];
        }
      }

      // If the line number is greater than the total number of lines
      return "";
    }

    std::vector<std::vector<double>> getlines_from_mmap(std::vector<int> training_line_numbers) {
      std::vector<std::vector<double>> res;
      // std::cout << "training line_numbers size: " << training_line_numbers.size() << std::endl;

      int current_line = 1;
      int curr_finding_index = 0;

      const char *line_start = data;
      const char *line_end = nullptr;

      for (size_t i = 0; i < filesize; ++i) {
        if(curr_finding_index == training_line_numbers.size()) return res;

        if (data[i] == '\n' || i == filesize - 1) {
          line_end = &data[i];

          if (current_line == training_line_numbers[curr_finding_index]) {
            res.push_back(split(std::string(line_start, line_end - line_start), delimiter));
            // std::cout << "Curr line number: " << training_line_numbers[curr_finding_index] << std::endl;
            curr_finding_index++;
          }

          current_line++;
          line_start = &data[i + 1];
        }
      }

      // If the line number is greater than the total number of lines
      return res;
    }

    std::vector<double> read_line_number(size_t line_number) {
      if(line_number < 1) throw std::runtime_error("Line number should be greater than 0. Received: " + std::to_string(line_number));

      std::string line = getline_from_mmap(line_number);

      return split(line, delimiter);
    }
};