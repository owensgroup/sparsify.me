/**
 * @file sparsify.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-04
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <sparsify.me/util.hxx>

#pragma once
namespace sparsifyme {
template <std::size_t BLK_M = 2, std::size_t BLK_N = 2, typename type_t>
void sparsify(type_t* weights,
              std::size_t* mask,
              std::size_t const& m,
              std::size_t const& n,
              float sparsity_factor = 0.5,
              cudaStream_t stream = 0) {
  // Block configuration (2 x 2)
  constexpr std::size_t blk_m = BLK_M;
  constexpr std::size_t blk_n = BLK_N;
  constexpr std::size_t blk_size = blk_m * blk_n;

  // Tile configuration (Number of blocks per m, n)
  std::size_t tile_m = m / blk_m;
  std::size_t tile_n = n / blk_n;

  // Sparsify lambda
  std::size_t number_of_zeros_per_block = floor(blk_size * sparsity_factor);

  auto sparsify = [=] __device__(std::size_t const& blk_idx) {
    // Global idx strided by blk_idx
    auto global_idx = blk_idx * blk_size;
    std::size_t sparsified = 0;

    // Block idx as (m, n)
    // auto blk_m_idx = blk_idx % blk_n;
    // auto blk_n_idx = blk_idx / blk_n;

    // Loop over the (2 x 2) block
    for (std::size_t h = 0; h < blk_m; ++h) {
      for (std::size_t w = 0; w < blk_n; ++w) {
        if (sparsified == number_of_zeros_per_block)
          break;

        // <todo> need a good condition to determine
        // if a value should be sparsified.
        auto idx = global_idx + h + (w * blk_n);
        weights[idx] = (type_t)0;
        mask[idx] = 0;
        sparsified++;
      }
    }

    return 0;
  };

  // Fill the mask with 1s.
  thrust::fill_n(thrust::cuda::par.on(stream), mask, m * n, 1);

  // Perform sparsification.
  thrust::transform(
      thrust::cuda::par.on(stream),                    // CUDA stream
      thrust::make_counting_iterator<std::size_t>(0),  // Begin iterator: 0
      thrust::make_counting_iterator<std::size_t>(
          tile_m * tile_n),             // End iterator: tile_m * tile_n
      thrust::make_discard_iterator(),  // Discard output
      sparsify                          // Unary Operator
  );
}
}  // namespace sparsifyme