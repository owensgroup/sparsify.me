/**
 * @file sparsify.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief A simple test case to show how to use the sparsify.me.
 * @version 0.1
 * @date 2021-05-04
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <ctime>
#include <iostream>

#include <sparsify.me/sparsify.hxx>
#include <sparsify.me/util.hxx>

int main(int argc, char** argv) {
  using weight_t = float;

  // Weight Matrix (m x n)
  constexpr std::size_t m = 32;
  constexpr std::size_t n = 32;

  thrust::host_vector<weight_t> h_weights(m * n);

  // Initialize with random weights.
  srand((unsigned)time(0));
  for (auto& weight : h_weights)
    weight = sparsifyme::util::get_random();

  // Move the data to GPU.
  thrust::device_vector<weight_t> d_weights = h_weights;
  auto weights = d_weights.data().get();

  thrust::device_vector<std::size_t> d_mask(m * n);
  auto mask = d_mask.data().get();

  // (blk_m, blk_n) = 2x2 tile, 50% sparsity.
  sparsifyme::sparsify<2, 2>(weights, mask, m, n);

  // Log and output.
  std::cout << "Matrix Size (m, n) = (" << m << ", " << n << ")" << std::endl;
  std::cout << "Weights (sparsified) = " << std::endl;
  thrust::copy(d_weights.begin(), d_weights.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;
}