/**
 * @file gemm.hxx
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

#include <cublas_v2.h>

#include <sparsify.me/util.hxx>

#pragma once
namespace sparsifyme {
namespace batched {
/**
 * @brief Specialize over batched dense gemm.
 */
template <typename type_t>
float gemm(type_t** A_ptrs,
           type_t** B_ptrs,
           type_t** C_ptrs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cublasOperation_t transpose_a = CUBLAS_OP_N,
           cublasOperation_t transpose_b = CUBLAS_OP_N,
           type_t alpha = (type_t)1.0f,
           type_t beta = (type_t)0.0f);

template <>
float gemm(__half** A_ptrs,
           __half** B_ptrs,
           __half** C_ptrs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cublasOperation_t transpose_a,
           cublasOperation_t transpose_b,
           __half alpha,
           __half beta) {
  cublasHandle_t context;
  cublasCreate(&context);

  util::timer_t timer;
  timer.begin();
  /**
   * @brief Call cublas' batched GEMM.
   *
   * @par Overview
   * cublasStatus_t cublasHgemmBatched(cublasHandle_t handle,
   *                                   cublasOperation_t transa,
   *                                   cublasOperation_t transb,
   *                                   int m, int n, int k,
   *                                   const __half *alpha,
   *                                   const __half *Aarray[], int lda,
   *                                   const __half *Barray[], int ldb,
   *                                   const __half *beta,
   *                                   __half *Carray[], int ldc,
   *                                   int batchCount);
   *
   * @par cublasOperation_t
   * | Value       | Meaning                                       |
   * |-------------|-----------------------------------------------|
   * | CUBLAS_OP_N | the non-transpose operation is selected       |
   * | CUBLAS_OP_T | the transpose operation is selected           |
   * | CUBLAS_OP_C | the conjugate transpose operation is selected |
   *
   */
  try {
    cublasStatus_t status =
        cublasHgemmBatched(context, transpose_a, transpose_b, m, n, k, &alpha,
                           A_ptrs, m, B_ptrs, k, &beta, C_ptrs, m, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw(status);
  } catch (cublasStatus_t s) {
    std::cout << "error: cublasHgemmBatched exited with an error: " << s
              << std::endl;
  }
  return timer.end();
}

template <>
float gemm(float** A_ptrs,
           float** B_ptrs,
           float** C_ptrs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cublasOperation_t transpose_a,
           cublasOperation_t transpose_b,
           float alpha,
           float beta) {
  cublasHandle_t context;
  cublasCreate(&context);

  util::timer_t timer;
  timer.begin();
  /**
   * @brief Call cublas' batched GEMM.
   *
   * @par Overview
   * cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
   *                                   cublasOperation_t transa,
   *                                   cublasOperation_t transb,
   *                                   int m, int n, int k,
   *                                   const float *alpha,
   *                                   const float *Aarray[], int lda,
   *                                   const float *Barray[], int ldb,
   *                                   const float *beta,
   *                                   float *Carray[], int ldc,
   *                                   int batchCount);
   *
   * @par cublasOperation_t
   * | Value       | Meaning                                       |
   * |-------------|-----------------------------------------------|
   * | CUBLAS_OP_N | the non-transpose operation is selected       |
   * | CUBLAS_OP_T | the transpose operation is selected           |
   * | CUBLAS_OP_C | the conjugate transpose operation is selected |
   *
   */
  try {
    cublasStatus_t status =
        cublasSgemmBatched(context, transpose_a, transpose_b, m, n, k, &alpha,
                           A_ptrs, m, B_ptrs, k, &beta, C_ptrs, m, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw(status);
  } catch (cublasStatus_t s) {
    std::cout << "error: cublasSgemmBatched exited with an error: " << s
              << std::endl;
  }
  return timer.end();
}

template <>
float gemm(double** A_ptrs,
           double** B_ptrs,
           double** C_ptrs,
           std::size_t m,
           std::size_t n,
           std::size_t k,
           std::size_t batch_size,
           cublasOperation_t transpose_a,
           cublasOperation_t transpose_b,
           double alpha,
           double beta) {
  cublasHandle_t context;
  cublasCreate(&context);

  util::timer_t timer;
  timer.begin();
  /**
   * @brief Call cublas' batched GEMM.
   *
   * @par Overview
   * cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
   *                                   cublasOperation_t transa,
   *                                   cublasOperation_t transb,
   *                                   int m, int n, int k,
   *                                   const double *alpha,
   *                                   const double *Aarray[], int lda,
   *                                   const double *Barray[], int ldb,
   *                                   const double *beta,
   *                                   double *Carray[], int ldc,
   *                                   int batchCount);
   *
   * @par cublasOperation_t
   * | Value       | Meaning                                       |
   * |-------------|-----------------------------------------------|
   * | CUBLAS_OP_N | the non-transpose operation is selected       |
   * | CUBLAS_OP_T | the transpose operation is selected           |
   * | CUBLAS_OP_C | the conjugate transpose operation is selected |
   *
   */
  try {
    cublasStatus_t status =
        cublasDgemmBatched(context, transpose_a, transpose_b, m, n, k, &alpha,
                           A_ptrs, m, B_ptrs, k, &beta, C_ptrs, m, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw(status);
  } catch (cublasStatus_t s) {
    std::cout << "error: cublasDgemmBatched exited with an error: " << s
              << std::endl;
  }
  return timer.end();
}

}  // namespace batched
}  // namespace sparsifyme