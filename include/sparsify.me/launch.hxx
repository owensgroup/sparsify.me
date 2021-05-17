/**
 * @file launch.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <thrust/host_vector.h>
#include <sparsify.me/util.hxx>
#include <cusparse.h>

namespace sparsifyme {
namespace util {
struct launch_t {
  cudaStream_t stream;
  cudaEvent_t event;
  cusparseHandle_t handle;
  util::timer_t timer;
  void* buffer;
  std::size_t buffer_size = 0;
};

void create_launch_configs(thrust::host_vector<launch_t>& configs) {
  for (auto& config : configs) {
    cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking);
    cusparseCreate(&config.handle);
    cusparseSetStream(config.handle, config.stream);
  }
}

void destroy_launch_configs(thrust::host_vector<launch_t>& configs) {
  for (auto& config : configs) {
    cusparseDestroy(config.handle);
    cudaFreeAsync(config.buffer, config.stream);
    config.buffer = nullptr;
    buffer_size = 0;
  }
}
}  // namespace util
}  // namespace sparsifyme