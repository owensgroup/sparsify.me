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

#include <sparsify.me/containers/ell.hxx>
#include <sparsify.me/util/util.hxx>

#include <sparsify.me/spmm.hxx>

int main(int argc, char** argv) {
  using namespace sparsifyme;
  using type_t = float;

  if (argc != 5) {
    std::cout << "Invalid # of arguments. Usage: ./spmm m n k b" << std::endl;
    return EXIT_FAILURE;
  }

  // Sizes (m, n, k) and batches
  std::size_t m = std::stoi(argv[1]);
  std::size_t n = std::stoi(argv[2]);
  std::size_t k = std::stoi(argv[3]);
  std::size_t batch_size = std::stoi(argv[4]);

  /// Sparse Matrices A (batched)
  thrust::host_vector<ell_t<type_t, memory_space_t::host>> h_As(batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    // Create a batch of A w/ random numbers.
    auto& h_A = h_As[batch];
    h_A.rows = m;
    h_A.cols = k;
    h_A.block_size = 2;           // 2 by 2 blocks.
    h_A.ell_cols = h_A.cols / 2;  // we divide by 2 assuming 50% sparsity.

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
    for (std::size_t i = 1; i <= h_A.values.size(); ++i)
      h_A.values[i - 1] = static_cast<float>(i);

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
  for (std::size_t i = 1; i <= h_B.size(); ++i)
    h_B[i - 1] = static_cast<float>(i);

  // Move the data to GPU.
  // Device:: Sparse Matrices A
  thrust::host_vector<ell_t<type_t, memory_space_t::device>> d_As(batch_size);

  // Device:: Matrix B
  thrust::device_vector<type_t> d_B = h_B;

  // Device:: Output Matrix C
  thrust::host_vector<type_t*> C_ptrs;
  thrust::host_vector<thrust::device_vector<type_t>> d_C_batches(batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& d_A = d_As[batch];
    auto& h_A = h_As[batch];
    d_A = h_A;

    auto& d_C = d_C_batches[batch];
    d_C.resize(m * n);                   // m x n
    C_ptrs.push_back(d_C.data().get());  // Store pointers
  }

  float elapsed = batched::spmm(d_As.data(), d_B.data().get(), C_ptrs.data(), m,
                                n, k, batch_size);

  std::cout << elapsed << std::endl;
  // // Log and output.
  // std::cout << "Matrix Sizes (m, n, k, batch) = (" << m << ", " << n << ", "
  //           << k << ", " << batch_size << ")" << std::endl;
  // std::cout << "Time elapsed (ms) = " << elapsed << std::endl;

  // for (std::size_t batch = 0; batch < batch_size; ++batch) {
  //   h_As[batch].print();
  // }

  // std::cout << "B-Matrix" << std::endl;
  // for (std::size_t batch = 0; batch < batch_size; ++batch) {
  //   std::cout << "\t";
  //   for (auto& val : h_B)
  //     std::cout << static_cast<float>(val) << " ";
  //   std::cout << std::endl;
  // }

  // std::cout << "C-Matrix" << std::endl;
  // for (std::size_t batch = 0; batch < batch_size; ++batch) {
  //   std::cout << "\t";
  //   auto& d_C = d_C_batches[batch];
  //   thrust::host_vector<type_t> h_C = d_C;
  //   for (auto& val : h_C)
  //     std::cout << static_cast<float>(val) << " ";
  //   std::cout << std::endl;
  // }
}