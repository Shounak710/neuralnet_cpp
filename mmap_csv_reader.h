#pragma once

#include "helper_functions.h"

#include<iostream>
#include<sys/mman.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/stat.h>
#include<cstring>
#include<string>
#include <mutex>
#include<unordered_map>
#include<thread>

std::mutex feature_data_mutex;

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
  
    // std::string getline_from_mmap(int line_number) {
    //   int current_line = 1;
    //   const char *line_start = data;
    //   const char *line_end = nullptr;

    //   for (size_t i = 0; i < filesize; ++i) {
    //     if (data[i] == '\n' || i == filesize - 1) {
    //       line_end = &data[i];
    //       if (current_line == line_number) {
    //           return std::string(line_start, line_end - line_start);
    //       }
    //       current_line++;
    //       line_start = &data[i + 1];
    //     }
    //   }

    //   // If the line number is greater than the total number of lines
    //   return "";
    // }

    // static std::string getline_from_mmap(int line_number, char*& data, size_t& filesize) {
    //     int current_line = 1;
    //     const char *line_start = data;
    //     const char *line_end = nullptr;

    //     for (size_t i = 0; i < filesize; ++i) {
    //         if (data[i] == '\n' || i == filesize - 1) {
    //             line_end = &data[i];
    //             if (current_line == line_number) {
    //                 return std::string(line_start, line_end - line_start);
    //             }
    //             current_line++;
    //             line_start = &data[i + 1];
    //         }
    //     }
    //     return "";
    // }

    static void getline_from_mmap_thread(int start_idx, int end_idx, const std::vector<int>& training_line_numbers,
    std::unordered_map<int, std::vector<double>>& feature_data, const char* data, const size_t filesize, const char delimiter) {
      int current_line = 1;
      int curr_finding_index = start_idx;

      const char *line_start = data;
      const char *line_end = nullptr;

      for (size_t i = 0; i < filesize; ++i) {
        if(curr_finding_index == end_idx) return;

        if (data[i] == '\n' || i == filesize - 1) {
          line_end = &data[i];

          if (current_line == training_line_numbers[curr_finding_index]) {
            std::vector<double> features = split(std::string(line_start, line_end - line_start), delimiter);
            std::lock_guard<std::mutex> lock(feature_data_mutex);
            feature_data[training_line_numbers[curr_finding_index]] = features;
            // std::cout << "Curr line number: " << training_line_numbers[curr_finding_index] << std::endl;
            curr_finding_index++;
          }

          current_line++;
          line_start = &data[i + 1];
        }
      }
    }

    std::unordered_map<int, std::vector<double>> getlines_from_mmap_thread(const std::vector<int>& training_line_numbers) {
      std::unordered_map<int, std::vector<double>> feature_data;

      int num_threads = std::thread::hardware_concurrency(); // Number of threads supported by the hardware

      std::vector<std::thread> threads;
      int datapts_per_thread = training_line_numbers.size() / num_threads;

      for (int t = 0; t < num_threads; ++t) {
          int start_idx = t * datapts_per_thread;
          int end_idx = (t == num_threads - 1) ? training_line_numbers.size() : start_idx + datapts_per_thread;

          threads.emplace_back(getline_from_mmap_thread, start_idx, end_idx, std::cref(training_line_numbers),
          std::ref(feature_data), std::ref(data), std::ref(filesize), delimiter);
      }

      for (auto& th : threads) {
          th.join();
      }

      return feature_data;
    }

    // std::vector<std::vector<double>> getlines_from_mmap(std::vector<int> training_line_numbers) {
    //   std::vector<std::vector<double>> res;
    //   // std::cout << "training line_numbers size: " << training_line_numbers.size() << std::endl;

    //   int current_line = 1;
    //   int curr_finding_index = 0;

    //   const char *line_start = data;
    //   const char *line_end = nullptr;

    //   for (size_t i = 0; i < filesize; ++i) {
    //     if(curr_finding_index == training_line_numbers.size()) return res;

    //     if (data[i] == '\n' || i == filesize - 1) {
    //       line_end = &data[i];

    //       if (current_line == training_line_numbers[curr_finding_index]) {
    //         res.push_back(split(std::string(line_start, line_end - line_start), delimiter));
    //         // std::cout << "Curr line number: " << training_line_numbers[curr_finding_index] << std::endl;
    //         curr_finding_index++;
    //       }

    //       current_line++;
    //       line_start = &data[i + 1];
    //     }
    //   }

    //   // If the line number is greater than the total number of lines
    //   return res;
    // }

    std::vector<double> read_line_number(int line_number) {
      if(line_number < 1) throw std::runtime_error("Line number should be greater than 0. Received: " + std::to_string(line_number));

      return getlines_from_mmap_thread({line_number}).begin()->second;
    }
};