#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#pragma once
namespace sparsifyme {
namespace util {

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

}  // namespace util
}  // namespace sparsifyme