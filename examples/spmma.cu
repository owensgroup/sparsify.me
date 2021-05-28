/**
 * @file spmm.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Benchmark batched Matrix-Multiplication for Sparse and Dense inputs.
 * @version 0.1
 * @date 2021-05-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/util/util.hxx>
#include <sparsify.me/spmma.hxx>

int main(int argc, char** argv) {
  using namespace sparsifyme;
  using type_t = float;

  int major_cc, minor_cc;
  cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0);

  if (argc != 5) {
    std::cout << "Invalid # of arguments. Usage: ./spmma m n k b" << std::endl;
    return EXIT_FAILURE;
  }

  if (!(major_cc == 8 && minor_cc == 0)) {
    std::cerr << "\ncusparseLt is supported only on GPU devices with compute "
                 "capability == 8.0, current: "
              << major_cc << "." << minor_cc << std::endl;
    return EXIT_FAILURE;
  }

  // Sizes (m, n, k) and batches
  std::size_t m = std::stoi(argv[1]);
  std::size_t n = std::stoi(argv[2]);
  std::size_t k = std::stoi(argv[3]);
  std::size_t batch_size = std::stoi(argv[4]);

  thrust::host_vector<type_t> h_A(m * k * batch_size);
  thrust::host_vector<type_t> h_B(k * n * batch_size);

  for (auto& a : h_A)
    a = static_cast<type_t>(static_cast<float>(util::get_random()));

  for (auto& b : h_B)
    b = static_cast<type_t>(static_cast<float>(util::get_random()));

  thrust::device_vector<type_t> A = h_A;
  thrust::device_vector<type_t> B = h_B;
  thrust::device_vector<type_t> C(m * n * batch_size);

  auto profiled_times = spmma(A.data().get(), B.data().get(), C.data().get(), m,
                              n, k, batch_size);

  std::cout << "Pruning Time (ms): " << profiled_times[0] << std::endl;
  std::cout << "Compression Time (ms): " << profiled_times[1] << std::endl;
  std::cout << "SpMMA Time (ms): " << profiled_times[2] << std::endl;
}