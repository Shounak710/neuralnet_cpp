#pragma once

#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>

template<typename T = double>
inline std::vector<T> split(const std::string& str, char delimiter=',') {
  std::vector<T> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(std::stof(token));
  }

  return tokens;
}