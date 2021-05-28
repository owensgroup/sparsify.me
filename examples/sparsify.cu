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

#include <sparsify.me/util/util.hxx>

#include <sparsify.me/sparsify.hxx>

int main(int argc, char** argv) {
  using weight_t = float;
  if(argc != 3) {
    std::cout << "Invalid # of args. Usage: ./sparsify m n" << std::endl;
    return EXIT_FAILURE;
  }
  // Weight Matrix (m x n)
  std::size_t m = std::stoi(argv[1]);
  std::size_t n = std::stoi(argv[2]);

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

  sparsifyme::util::timer_t timer;
  timer.begin();
  // (blk_m, blk_n) = 2x2 tile, 50% sparsity.
  sparsifyme::sparsify<2, 2>(weights, mask, m, n);
  float elapsed = timer.end();
  // Log and output.
  // std::cout << "Matrix Size (m, n) = (" << m << ", " << n << ")" << std::endl;
  // std::cout << "Weights (sparsified) = " << std::endl;
  // thrust::copy(d_weights.begin(), d_weights.end(),
  //              std::ostream_iterator<weight_t>(std::cout, " "));
  // std::cout << std::endl;
  std::cout << elapsed << std::endl;
}