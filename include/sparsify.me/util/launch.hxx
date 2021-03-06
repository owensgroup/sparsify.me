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
#include <cusparse.h>
#include <thrust/host_vector.h>
#include <sparsify.me/util/timer.hxx>

namespace sparsifyme {
namespace util {

struct launch_t {
  cudaStream_t stream;
  cudaEvent_t event;
  cusparseHandle_t handle;
  void* buffer;
  std::size_t buffer_size = 0;
};

void create_launch_configs(std::vector<launch_t>& configs) {
  for (auto& config : configs) {
    cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking);
    cusparseCreate(&config.handle);
    cusparseSetStream(config.handle, config.stream);
  }
}

void destroy_launch_configs(std::vector<launch_t>& configs) {
  for (auto& config : configs) {
    cusparseDestroy(config.handle);
    cudaFreeAsync(config.buffer, config.stream);
    config.buffer = nullptr;
    config.buffer_size = 0;
  }
}
}  // namespace util
}  // namespace sparsifyme