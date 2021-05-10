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
#include <thread>

#include <cusparse.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/ell.hxx>
#include <sparsify.me/util.hxx>

#pragma once

namespace sparsifyme {
namespace batched {
template <typename type_t>
float spmm(ell_t<type_t, util::memory_space_t::device>* As,
           type_t* B,
           type_t** C_ptrs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cusparseOperation_t transpose_a = CUSPARSE_OPERATION_NON_TRANSPOSE,
           cusparseOperation_t transpose_b = CUSPARSE_OPERATION_NON_TRANSPOSE,
           type_t alpha = (type_t)1.0f,
           type_t beta = (type_t)0.0f) {
  cusparseHandle_t master_handle;
  cusparseCreate(&master_handle);
  thrust::host_vector<cusparseHandle_t> handles;

  cudaStream_t master_stream;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  cusparseSetStream(master_handle, master_stream);

  thrust::host_vector<util::launch_t> settings;
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    util::launch_t s;
    cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming);
    settings.push_back(s);

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSetStream(handle, settings[batch].stream);
    handles.push_back(handle);
  }

  thrust::host_vector<cusparseSpMatDescr_t> desc_As(
      batch_size);              // vector of A matrices descriptors
  cusparseDnMatDescr_t desc_B;  // dense B matrix descriptor
  thrust::host_vector<cusparseDnMatDescr_t> desc_Cs(
      batch_size);  // vector of C matrices descriptors

  thrust::host_vector<thrust::device_vector<char>> buffers(batch_size);

  // Matrix descriptor for A.
  cusparseCreateDnMat(&desc_B, k, n, k, B, CUDA_R_16F, CUSPARSE_ORDER_COL);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    auto A = As[batch];
    auto& desc_A = desc_As[batch];

    // This API is absolutely atrocious.
    // Build Blocked-ELL descr based on ell_t.
    cusparseCreateBlockedEll(&desc_A, A.rows, A.cols, A.block_size,
                             A.column_indices.size(),  // ELL Columns
                             A.column_indices.data().get(),
                             A.values.data().get(), CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // Create dense matrix C.
    auto& desc_C = desc_Cs[batch];
    cusparseCreateDnMat(&desc_C, m, n, m, C_ptrs[batch], CUDA_R_16F,
                        CUSPARSE_ORDER_COL);

    std::size_t buffer_size;
    cusparseSpMM_bufferSize(master_handle, transpose_a, transpose_b, &alpha,
                            desc_A, desc_B, &beta, desc_C, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);

    auto& buffer = buffers[batch];
    buffer.resize(buffer_size);
  }

  cudaDeviceSynchronize();

  std::vector<std::thread> threads;
  util::timer_t timer;
  timer.begin();
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    threads.push_back(std::thread([&, batch]() {
      auto handle = handles[batch];
      auto& desc_A = desc_As[batch];
      auto& desc_C = desc_Cs[batch];
      auto& buffer = buffers[batch];

      /**
       * @par Overview
       * cusparseStatus_t
       * cusparseSpMM(cusparseHandle_t     handle,
       *              cusparseOperation_t  opA,
       *              cusparseOperation_t  opB,
       *              const void*          alpha,
       *              cusparseSpMatDescr_t matA,
       *              cusparseDnMatDescr_t matB,
       *              const void*          beta,
       *              cusparseDnMatDescr_t matC,
       *              cudaDataType         computeType,
       *              cusparseSpMMAlg_t    alg,
       *              void*                externalBuffer);
       *
       */
      cusparseSpMM(handle, transpose_a, transpose_b, &alpha, desc_A, desc_B,
                   &beta, desc_C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                   (void*)buffer.data().get());
      cudaEventRecord(settings[batch].event, settings[batch].stream);
    }));
  }

  for (auto& thread : threads)
    thread.join();

  for (std::size_t batch = 0; batch < batch_size; ++batch)
    cudaStreamWaitEvent(master_stream, settings[batch].event, 0);

  cudaStreamSynchronize(master_stream);
  return timer.end();
}
}  // namespace batched
}  // namespace sparsifyme