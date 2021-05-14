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
    // Create a batch of A w/ random numbers.
    auto& h_A = h_As[batch];
    h_A.rows = m;
    h_A.cols = k;
    h_A.block_size = 2;  // 2 by 2 blocks.
    h_A.ell_cols = (h_A.cols / h_A.block_size) /
                   2;  // we divide by 2 assuming 50% sparsity.

    // Calculate sizes of values and indices arrays.
    h_A.blocked_rows = util::ceil_div(h_A.rows, h_A.block_size);
    h_A.blocked_cols = util::ceil_div(h_A.ell_cols, h_A.block_size);
    h_A.num_blocks = h_A.blocked_rows * h_A.blocked_cols;

    std::size_t col_idx_size = h_A.num_blocks;
    std::size_t values_size = h_A.ell_cols * h_A.rows;

    // Prepare data arrays.
    h_A.column_indices.resize(col_idx_size);
    h_A.values.resize(values_size);

    // do not care how values are initialized.
    for (auto& a_value : h_A.values)
      a_value = util::get_random();

    // column indices initialize to random columns.
    for (std::size_t r = 0; r < h_A.blocked_rows; ++r) {
      std::size_t fill_size = h_A.blocked_cols;
      std::vector<std::size_t> vec;
      while (vec.size() != fill_size) {
        std::size_t rand_idx = util::get_random(
            (std::size_t)0, (std::size_t)(h_A.cols / h_A.block_size));
        vec.emplace_back(rand_idx);

        // erase duplicates
        std::sort(begin(vec), end(vec));
        auto last = std::unique(begin(vec), end(vec));
        vec.erase(last, end(vec));
      }

      std::sort(begin(vec), end(vec));
      for (std::size_t c = 0; c < h_A.blocked_cols; ++c)
        h_A.column_indices[h_A.blocked_cols * r + c] = vec[c];
    }
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

  float elapsed = batched::spmm(d_As.data(), d_B.data().get(),
                                d_C_pointers.data(), m, n, k, batch_size);

  // Log and output.
  std::cout << "Matrix Sizes (m, n, k, batch) = (" << m << ", " << n << ", "
            << k << ", " << batch_size << ")" << std::endl;
  std::cout << "Time elapsed (ms) = " << elapsed << std::endl;

  std::cout << "A-Matrix" << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    std::cout << "A-Matrix (Column Idx) = ";
    thrust::copy(h_As[batch].column_indices.begin(),
                 h_As[batch].column_indices.end(),
                 std::ostream_iterator<std::size_t>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "A-Matrix (Values) = ";
    thrust::copy(h_As[batch].values.begin(), h_As[batch].values.end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

  std::cout << "B-Matrix" << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(h_B.begin(), h_B.end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

  std::cout << "C-Matrix" << std::endl;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    thrust::copy(d_C_batches[batch].begin(), d_C_batches[batch].end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }
}