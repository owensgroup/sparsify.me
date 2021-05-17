#include <random>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#pragma once
namespace sparsifyme {
namespace util {

struct launch_t {
  cudaStream_t stream;
  cudaEvent_t event;
};

enum memory_space_t { device, host };

template <typename type_t, memory_space_t space>
using vector_t =
    std::conditional_t<space == memory_space_t::host,  // condition
                       thrust::host_vector<type_t>,    // host_type
                       thrust::device_vector<type_t>   // device_type
                       >;

float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1);  // rage 0 - 1
  return dis(e);
}

struct timer_t {
  float time;

  timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  // Alias of each other, start the timer.
  void begin() { cudaEventRecord(start_); }
  void start() { this->begin(); }

  float end() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return milliseconds();
  }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  cudaEvent_t start_, stop_;
};

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