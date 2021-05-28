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
#include <cusparseLt.h>
#include <sparsify.me/util/util.hxx>
#include <vector>
#pragma once
namespace sparsifyme {
namespace ampere {
/**
 * @brief Ampere prune and SpMM functions
 */
template <typename type_t>
std::vector<float> spmma(type_t** dA,
                        type_t** dB,
                        type_t** dC,
                        std::size_t m,
                        std::size_t n,
                        std::size_t k,
                        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        type_t alpha = (type_t)1.0f,
                        type_t beta = (type_t)0.f);

template <>
std::vector<float> spmma(__half** dA,
                        __half** dB,
                        __half** dC,
                        std::size_t m,
                        std::size_t n,
                        std::size_t k,
                        cusparseOperation_t opA,
                        cusparseOperation_t opB,
                        __half alpha,
                        __half beta)
  {
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t matA, matB, matC;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cudaStream_t stream = nullptr;
  auto type = CUDA_R_16F;
  auto compute_type = CUSPARSE_COMPUTE_16F;
  if(m % 8 != 0 || n % 8 != 0 || k % 8 != 0) {
    throw "Invalid matrix sizes for data type __half. Rows and columns must be divisible by 8.\n";
  }
  cusparseLtInit(&handle);
  __half **dD;
  dD = dC;
  util::timer_t timer;
  // Initialize cuSparse Matrix Descriptors
  cusparseLtStructuredDescriptorInit(
                              &handle,
                              &matA,
                              m,
                              k,
                              k,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW,
                              CUSPARSELT_SPARSITY_50_PERCENT);
  
  cusparseLtDenseDescriptorInit(
                              &handle,
                              &matB,
                              k,
                              n,
                              n,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW);

  cusparseLtDenseDescriptorInit(
                              &handle,
                              &matC,
                              k,
                              n,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW);

  // Initialize matmul plan/algorithm
  cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB,
                                &matA, &matB, &matC, &matC,
                                compute_type);
  
  cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
                                  CUSPARSELT_MATMUL_ALG_DEFAULT);
  
  int alg = 0;
  cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
                                  CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                  &alg, sizeof(alg));

  size_t workspace_size, compressed_size;
  cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);
  cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size);

  timer.begin();
  // Prune the sparse structured Matrix A
  int* d_valid;
  cudaMalloc((void**)&d_valid, sizeof(d_valid));
  cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
  cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream);
  int is_valid;
  cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if(is_valid != 0) {
    throw "Incorrect pruning results.\n";
  }
  float prune_time = timer.end();

  util::timer_t compress_timer;
  compress_timer.begin();
  // Prune and Compress Sparse Matrix A
  __half *dA_compressed;
  cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
  cudaMalloc((void**)&dA_compressed, compressed_size);
  cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream);
  float compress_time = compress_timer.end();

  util::timer_t mul_timer;
  mul_timer.begin();
  // Do matmul
  void*         d_workspace = nullptr;
  int           num_streams = 0;
  cudaStream_t* streams     = nullptr;
  cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams);
  float mul_time = mul_timer.end();
  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtDestroy(&handle);
  return {prune_time, compress_time, mul_time};
}

template <>
std::vector<float> spmma(float** dA,
                  float** dB,
                  float** dC,
                  std::size_t m,
                  std::size_t n,
                  std::size_t k,
                  cusparseOperation_t opA,
                  cusparseOperation_t opB,
                  float alpha,
                  float beta) {
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t matA, matB, matC;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cudaStream_t stream = nullptr;
  auto type = CUDA_R_32F;
  auto compute_type = CUSPARSE_COMPUTE_16F;
  if(m % 4 != 0 || n % 4 != 0 || k % 4 != 0) {
    throw "Invalid matrix sizes for data type float. Rows and columns must be divisible by 8.\n";
  }
  cusparseLtInit(&handle);
  float *dD;
  dD = dC;
  util::timer_t timer;
  // Initialize cuSparse Matrix Descriptors
  cusparseLtStructuredDescriptorInit(
                              &handle,
                              &matA,
                              m,
                              k,
                              k,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW,
                              CUSPARSELT_SPARSITY_50_PERCENT);
  
  cusparseLtDenseDescriptorInit(
                              &handle,
                              &matB,
                              k,
                              n,
                              n,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW);

  cusparseLtDenseDescriptorInit(
                              &handle,
                              &matC,
                              k,
                              n,
                              16,
                              type,
                              CUSPARSE_ORDER_ROW);

  // Initialize matmul plan/algorithm
  cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB,
                                &matA, &matB, &matC, &matC,
                                compute_type);
  
  cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
                                  CUSPARSELT_MATMUL_ALG_DEFAULT);
  
  int alg = 0;
  cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
                                  CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                  &alg, sizeof(alg));

  size_t workspace_size, compressed_size;
  cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);
  cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size);

  timer.begin();
  // Prune the sparse structured Matrix A
  int* d_valid;
  cudaMalloc((void**)&d_valid, sizeof(d_valid));
  cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
  cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream);
  int is_valid;
  cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if(is_valid != 0) {
    throw "Incorrect pruning results.\n";
  }
  float prune_time = timer.end();

  util::timer_t compress_timer;
  compress_timer.begin();
  // Prune and Compress Sparse Matrix A
  __half *dA_compressed;
  cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
  cudaMalloc((void**)&dA_compressed, compressed_size);
  cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream);
  float compress_time = compress_timer.end();

  util::timer_t mul_timer;
  mul_timer.begin();
  // Do matmul
  void*         d_workspace = nullptr;
  int           num_streams = 0;
  cudaStream_t* streams     = nullptr;
  cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams);
  float mul_time = mul_timer.end();
  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtDestroy(&handle);
  return {prune_time, compress_time, mul_time};
}

namespace batched {

}//namespace batched
}//namespace ampere
}//namespace sparsifyme