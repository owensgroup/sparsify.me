/**
 * @file util.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <random>

namespace sparsifyme {
namespace util {

template <typename type_t = float>
type_t get_random(type_t begin = 0.0f, type_t end = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(begin, end);
  return (type_t)dis(gen);
}

template <typename type_t>
type_t ceil_div(type_t x, type_t y) {
  return (x + y - 1) / y;
}

// m,n,k,b
typedef std::tuple<int, int, int, int> mat_sz;

std::vector<mat_sz> read_shapes(std::string filename) {
  std::ifstream shape_file;
  shape_file.open(filename);
  std::vector<mat_sz> shape_data;
  if(!shape_file.is_open()) {
    throw "Unable to open shape CSV file.";
  }
  
  std::string line;
  std::getline(shape_file, line);
  while(std::getline(shape_file, line)) {
    std::istringstream s(line);
    std::string field;
    std::vector<int> line_data;
    while(getline(s, field, ',')) {
      line_data.push_back(std::stoi(field));
    }
    int m = line_data[0];
    int n = line_data[1];
    int k = line_data[2];
    int b = line_data[3];

    shape_data.push_back(std::make_tuple(m,n,k,b));
  }
  return shape_data;
}

}  // namespace util
}  // namespace sparsifyme

#include <sparsify.me/util/timer.hxx>
#include <sparsify.me/util/launch.hxx>
