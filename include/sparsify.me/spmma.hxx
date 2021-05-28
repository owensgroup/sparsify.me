/**
 * @file ampere.hxx
 * @author Teja Aluru (tsaluru@ucdavis.edu)
 * @brief Templated ampere utility functions
 * @version 0.1
 * @date 2021-05-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <cusparse.h>
#include <cusparseLt.h>
#include <sparsify.me/util/util.hxx>
#include <vector>

namespace sparsifyme {
/**
 * @brief Ampere prune and SpMM functions
 */
template <typename type_t>
std::vector<float> spmma(
    type_t* dA,
    type_t* dB,
    type_t* dC,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t batch_size,
    cusparseOperation_t transpose_a = CUSPARSE_OPERATION_NON_TRANSPOSE,
    cusparseOperation_t transpose_b = CUSPARSE_OPERATION_NON_TRANSPOSE,
    float alpha = 1.0f,
    float beta = 0.0f) {
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t desc_A, desc_B, desc_C;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cudaStream_t stream = nullptr;
  auto type = CUDA_R_16F;
  auto compute_type = CUSPARSE_COMPUTE_16F;
  unsigned alignment = 16;

  // Size check.
  if (m % 8 != 0 || n % 8 != 0 || k % 8 != 0) {
    std::cerr << "Invalid matrix sizes for data type __half. Rows and columns "
                 "must be divisible by 8."
              << std::endl;
  }

  cusparseLtInit(&handle);
  type_t* dD;
  dD = dC;

  // Initialize cuSparse Matrix Descriptors
  cusparseLtStructuredDescriptorInit(&handle, &desc_A, m, k, k, alignment, type,
                                     CUSPARSE_ORDER_ROW,
                                     CUSPARSELT_SPARSITY_50_PERCENT);

  cusparseLtDenseDescriptorInit(&handle, &desc_B, k, n, n, alignment, type,
                                CUSPARSE_ORDER_ROW);

  cusparseLtDenseDescriptorInit(&handle, &desc_C, m, n, n, alignment, type,
                                CUSPARSE_ORDER_ROW);

  // Initialize matmul plan/algorithm
  cusparseLtMatmulDescriptorInit(&handle, &matmul, transpose_a, transpose_b,
                                 &desc_A, &desc_B, &desc_C, &desc_C,
                                 compute_type);

  cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
                                   CUSPARSELT_MATMUL_ALG_DEFAULT);

  int alg = 0;
  cusparseLtMatmulAlgSetAttribute(
      &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg));

  size_t workspace_size, compressed_size;
  cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);
  cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size);

  util::timer_t prune_timer;
  prune_timer.begin();
  // Prune the sparse structured Matrix A
  thrust::device_vector<int> valid(1);
  cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE,
                       stream);
  cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, valid.data().get(), stream);
  int is_valid;
  cudaMemcpyAsync(&is_valid, valid.data().get(), sizeof(is_valid),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (is_valid != 0)
    std::cerr << "Incorrect pruning results." << std::endl;
  float prune_time = prune_timer.end();

  util::timer_t compress_timer;
  compress_timer.begin();
  // Prune and Compress Sparse Matrix A
  cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
  thrust::device_vector<type_t> compressed(compressed_size);

  cusparseLtSpMMACompress(&handle, &plan, dA, compressed.data().get(), stream);
  float compress_time = compress_timer.end();

  util::timer_t mul_timer;
  mul_timer.begin();
  // Do matmul
  void* d_workspace = nullptr;
  int num_streams = 0;
  cudaStream_t* streams = nullptr;
  cusparseLtMatmul(&handle, &plan, &alpha, compressed.data().get(), dB, &beta,
                   dC, dD, d_workspace, streams, num_streams);
  float mul_time = mul_timer.end();
  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtDestroy(&handle);
  return {prune_time, compress_time, mul_time};
}

namespace batched {}  // namespace batched
}  // namespace sparsifyme