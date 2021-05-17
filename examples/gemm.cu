/**
 * @file gemm.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Benchmark batched Matrix-Multiplication for Sparse and Dense inputs.
 * @version 0.1
 * @date 2021-05-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/util/util.hxx>

#include <sparsify.me/gemm.hxx>

int main(int argc, char** argv) {
  using type_t = float;

  // Sizes (m, n, k) and batches
  std::size_t m = 4;
  std::size_t n = 4;
  std::size_t k = 2;
  std::size_t batch_size = 4;

  /// Matrix A (pointers to data)
  thrust::host_vector<type_t*> h_A_pointers(batch_size);
  thrust::host_vector<thrust::host_vector<type_t>> h_A_batches(batch_size);

  /// Matrix B (pointers to data)
  thrust::host_vector<type_t*> h_B_pointers(batch_size);
  // NOTE: We just need 1 B-matrix that is duplicated over batches.
  thrust::host_vector<thrust::host_vector<type_t>> h_B_batches(1);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    // Create a batch of A w/ random numbers.
    auto& h_A = h_A_batches[batch];
    h_A.resize(m * k);  // m x k
    for (auto& a_value : h_A)
      a_value = sparsifyme::util::get_random();

    // Push the batch onto a vector of batches.
    h_A_pointers[batch] = h_A_batches[batch].data();

    // Create a batch of B w/ random numbers.
    auto& h_B = h_B_batches[0];
    h_B.resize(k * n);  // k x n
    for (auto& b_value : h_B)
      b_value = sparsifyme::util::get_random();

    // Push the batch onto a vector of batches.
    h_B_pointers[batch] = h_B_batches[0].data();
  }

  // Move the data to GPU.
  // Device:: Matrix A
  thrust::device_vector<type_t*> d_A_pointers;
  thrust::host_vector<thrust::device_vector<type_t>> d_A_batches(batch_size);

  // Device:: Matrix B
  thrust::device_vector<type_t*> d_B_pointers;
  // NOTE: We just need 1 B-matrix that is duplicated over batches.
  thrust::host_vector<thrust::device_vector<type_t>> d_B_batches(1);

  // Device:: Output Matrix C
  thrust::device_vector<type_t*> d_C_pointers;
  thrust::host_vector<thrust::device_vector<type_t>> d_C_batches(batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& d_A = d_A_batches[batch];
    d_A.resize(m * k);                         // m x k
    d_A = h_A_batches[batch];                  // CPU -> GPU
    d_A_pointers.push_back(d_A.data().get());  // Store pointers

    auto& d_B = d_B_batches[0];
    d_B.resize(k * n);                         // k x n
    d_B = h_B_batches[0];                      // CPU -> GPU
    d_B_pointers.push_back(d_B.data().get());  // Store pointers

    auto& d_C = d_C_batches[batch];
    d_C.resize(m * n);                         // m x n
    d_C_pointers.push_back(d_C.data().get());  // Store pointers
  }

  float elapsed = sparsifyme::batched::gemm(
      d_A_pointers.data().get(), d_B_pointers.data().get(),
      d_C_pointers.data().get(), m, n, k, batch_size);

  // Log and output.
  std::cout << "Matrix Sizes (m, n, k, batch) = (" << m << ", " << n << ", "
            << k << ", " << batch_size << ")" << std::endl;
  std::cout << "Time elapsed (ms) = " << elapsed << std::endl;

  std::cout << "A-Matrix = " << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(d_A_batches[batch].begin(), d_A_batches[batch].end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

  std::cout << "B-Matrix = " << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(d_B_batches[0].begin(), d_B_batches[0].end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

  std::cout << "C-Matrix = " << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(d_C_batches[batch].begin(), d_C_batches[batch].end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }
}