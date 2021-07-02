/**
 * @file spmm.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <math.h>
#include <cstdio>

#include <cusparse.h>
#include <nvToolsExt.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <sparsify.me/containers/ell.hxx>
#include <sparsify.me/util/util.hxx>

#pragma once

namespace sparsifyme {
namespace batched {

template <typename type_t>
float spmm(ell_t<type_t, memory_space_t::device>* As,
           type_t* B,
           type_t** Cs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cusparseOperation_t transpose_a = CUSPARSE_OPERATION_NON_TRANSPOSE,
           cusparseOperation_t transpose_b = CUSPARSE_OPERATION_NON_TRANSPOSE,
           float alpha = 1.0f,
           float beta = 0.0f) {
  thrust::host_vector<float> timers(batch_size);
  std::vector<util::launch_t> configs(batch_size);
  util::create_launch_configs(configs);

  thrust::host_vector<cusparseSpMatDescr_t> desc_As(batch_size);
  cusparseDnMatDescr_t desc_B;
  thrust::host_vector<cusparseDnMatDescr_t> desc_Cs(batch_size);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& A = As[batch];
    auto C = Cs[batch];

    auto& desc_A = desc_As[batch];
    auto& desc_C = desc_Cs[batch];
    // Create sparse matrix A in blocked ELL format
    cusparseCreateBlockedEll(&desc_A, A.rows, A.cols, A.block_size, A.ell_cols,
                             A.column_indices.data().get(),
                             A.values.data().get(), CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // Create dense matrix C
    cusparseCreateDnMat(&desc_C, m, n, m, C, CUDA_R_16F, CUSPARSE_ORDER_COL);
  }

  // Create dense matrix B
  cusparseCreateDnMat(&desc_B, k, n, k, B, CUDA_R_16F, CUSPARSE_ORDER_COL);

  // Set-up temporary buffer for each batch of SpMM
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& handle = configs[batch].handle;
    auto& stream = configs[batch].stream;
    auto& desc_A = desc_As[batch];
    auto& desc_C = desc_Cs[batch];

    void* buffer = configs[batch].buffer;
    auto& buffer_size = configs[batch].buffer_size;

    // Allocate an external buffer if needed
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc_A,
                            desc_B, &beta, desc_C, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
    cudaMallocAsync(&buffer, buffer_size, stream);
  }

  // Lazy synchronization to simplify code.
  cudaDeviceSynchronize();

  util::timer_t t;
  t.begin();
  nvtxRangePushA("batched-SpMM");
  // Execute batched-SpMM per each CPU thread
  #pragma omp parallel for num_threads(batch_size)
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& handle = configs[batch].handle;
    auto& stream = configs[batch].stream;
    auto& desc_A = desc_As[batch];
    auto& desc_C = desc_Cs[batch];

    void* buffer = configs[batch].buffer;
    // XXX: uncomment for average time.
    // util::timer_t timer;
    // timer.begin(stream);

    // Execute SpMM
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc_A, desc_B,
                  &beta, desc_C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                  buffer);
    // cudaStreamSynchronize(stream);

    // XXX: uncomment for average time.
    // timers[batch] = timer.end(stream);    
  }

  // Lazy synchronization to simplify code.
  cudaDeviceSynchronize();

  // End batched-SpMM range.
  nvtxRangePop();
  t.end();

  // clean-up.
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto& desc_A = desc_As[batch];
    auto& desc_C = desc_Cs[batch];
    cusparseDestroySpMat(desc_A);
    cusparseDestroyDnMat(desc_C);
  }
  cusparseDestroyDnMat(desc_B);
  util::destroy_launch_configs(configs);

  // XXX: uncomment for average time. 
  // std::cout << "Average Elapsed Time per Batch (ms) = " << thrust::reduce(timers.begin(), timers.end(), (float)0.0f) << std::endl;

  return t.milliseconds();
}  // namespace sparsifyme
}  // namespace batched
}  // namespace sparsifyme