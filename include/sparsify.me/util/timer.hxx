#include <random>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @file timer.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
namespace sparsifyme {
namespace util {

struct timer_t {
  float time;

  timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void begin(cudaStream_t stream = 0) { 
    cudaEventRecord(start_, stream);
    cudaEventSynchronize(start_);
  }
  
  float end(cudaStream_t stream = 0) {
    cudaEventRecord(stop_, stream);
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