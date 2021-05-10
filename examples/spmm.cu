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
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/ell.hxx>
#include <sparsify.me/spmm.hxx>
#include <sparsify.me/util.hxx>

int main(int argc, char** argv) {
  using namespace sparsifyme;
  using type_t = float;

  // Sizes (m, n, k) and batches
  std::size_t m = 4;
  std::size_t n = 3;
  std::size_t k = 4;
  std::size_t batch_size = 2;

  /// Sparse Matrices A (batched)
  thrust::host_vector<ell_t<type_t, util::memory_space_t::host>> h_As(
      batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    // Create a batch of B w/ random numbers.
    auto& h_A = h_As[batch];
    h_A.rows = m;
    h_A.cols = k;
    h_A.block_size = 2;
    std::size_t ell_cols = (h_A.cols / 2);  // (50% sparsity)
    h_A.num_blocks = ell_cols * h_A.rows / (h_A.block_size * h_A.block_size);

    // Prepare data arrays.
    h_A.values.resize((m * k) / 2);  // m x k / 2 (50% sparsity)
    h_A.column_indices.resize(ell_cols);

    for (auto& a_value : h_A.values)
      a_value = util::get_random();

    for (std::size_t col = 0; col < h_A.column_indices.size(); col++)
      h_A.column_indices[col] = col;
  }

  /// Dense Matrix B
  thrust::host_vector<type_t> h_B(k * n);

  // Create A w/ random numbers.
  for (auto& b_value : h_B)
    b_value = util::get_random();

  // Move the data to GPU.
  // Device:: Sparse Matrices A
  thrust::host_vector<ell_t<type_t, util::memory_space_t::device>> d_As(
      batch_size);
  // Device:: Matrix B
  thrust::device_vector<type_t> d_B = h_B;

  // Device:: Output Matrix C
  thrust::host_vector<type_t*> d_C_pointers;
  thrust::host_vector<thrust::device_vector<type_t>> d_C_batches(batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& d_A = d_As[batch];
    auto& h_A = h_As[batch];
    d_A = h_A;

    auto& d_C = d_C_batches[batch];
    d_C.resize(m * n);                         // m x n
    d_C_pointers.push_back(d_C.data().get());  // Store pointers
  }

  float time = batched::spmm(d_As.data(), d_B.data().get(), d_C_pointers.data(),
                             m, n, k, batch_size);

  // Log and output.
  std::cout << "Matrix Sizes (m, n, k, batch) = (" << m << ", " << n << ", "
            << k << ", " << batch_size << ")" << std::endl;
  std::cout << "Time elapsed (ms) = " << time << std::endl;

  std::cout << "A-Matrix = " << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    std::cout << "A-Matrix (Column Idx) = " << std::endl;
    thrust::copy(h_As[batch].column_indices.begin(), h_As[batch].column_indices.end(),
                 std::ostream_iterator<std::size_t>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "A-Matrix (Values) = " << std::endl;
    thrust::copy(h_As[batch].values.begin(), h_As[batch].values.end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

  std::cout << "B-Matrix = " << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(h_B.begin(), h_B.end(),
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