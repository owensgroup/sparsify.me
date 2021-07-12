/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#if !defined(CUSPARSELT_HEADER_)
#define CUSPARSELT_HEADER_

#include "cusparse.h"      // cusparseStatus_t

#include <cstddef>         // size_t
#include <driver_types.h>  // cudaStream_t
#include <library_types.h> // cudaDataType
#include <stdint.h>        // uint8_t

//##############################################################################
//# CUSPARSELT VERSION INFORMATION
//##############################################################################

#define CUSPARSELT_VER_MAJOR 0
#define CUSPARSELT_VER_MINOR 1
#define CUSPARSELT_VER_PATCH 0
#define CUSPARSELT_VER_BUILD 1
#define CUSPARSELT_VERSION (CUSPARSELT_VER_MAJOR * 1000 + \
                            CUSPARSELT_VER_MINOR *  100 + \
                            CUSPARSELT_VER_PATCH)

//------------------------------------------------------------------------------

#if !defined(_MSC_VER)
#   define CUSPARSELT_CPP_VERSION __cplusplus
#elif _MSC_FULL_VER >= 190024210 // Visual Studio 2015 Update 3
#   define CUSPARSELT_CPP_VERSION _MSVC_LANG
#else
#   define CUSPARSELT_CPP_VERSION 0
#endif

//------------------------------------------------------------------------------
#if !defined(CUSPARSELT_DISABLE_DEPRECATION)

#   if CUSPARSELT_CPP_VERSION >= 201402L

#       define CUSPARSELT_DEPRECATED(new_func)                                 \
            [[deprecated("please use " #new_func " instead")]]

#   elif defined(_MSC_VER)

#       define CUSPARSELT_DEPRECATED(new_func)                                 \
            __declspec(deprecated("please use " #new_func " instead"))

#   elif defined(__INTEL_COMPILER) || defined(__clang__) ||                    \
         (defined(__GNUC__) &&                                                 \
          (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#       define CUSPARSELT_DEPRECATED(new_func)                                 \
            __attribute__((deprecated("please use " #new_func " instead")))

#   elif defined(__GNUC__) || defined(__xlc__)

#       define CUSPARSELT_DEPRECATED(new_func)                                 \
            __attribute__((deprecated))

#   else

#       define CUSPARSELT_DEPRECATED(new_func)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L

#else // defined(DISABLE_CUSPARSE_DEPRECATED)

#   define CUSPARSELT_DEPRECATED(new_func)

#endif // !defined(CUSPARSELT_DISABLE_DEPRECATION)
//------------------------------------------------------------------------------

#undef CUSPARSELT_CPP_VERSION

// #############################################################################
// # MACRO
// #############################################################################

#if !defined(CUSPARSELT_API)
#    if defined(_WIN32)
#        define CUSPARSELT_API __stdcall
#    else
#        define CUSPARSELT_API
#    endif
#endif

//------------------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

//##############################################################################
//# OPAQUE DATA STRUCTURES
//##############################################################################

typedef struct { uint8_t data[1024]; } cusparseLtHandle_t;

typedef struct { uint8_t data[1024]; } cusparseLtMatDescriptor_t;

typedef struct { uint8_t data[1024]; } cusparseLtMatmulDescriptor_t;

typedef struct { uint8_t data[1024]; } cusparseLtMatmulAlgSelection_t;

typedef struct { uint8_t data[1024]; } cusparseLtMatmulPlan_t;

//##############################################################################
//# INITIALIZATION, DESTROY
//##############################################################################

cusparseStatus_t CUSPARSELT_API
cusparseLtInit(cusparseLtHandle_t* handle);

cusparseStatus_t CUSPARSELT_API
cusparseLtDestroy(const cusparseLtHandle_t* handle);

//##############################################################################
//# MATRIX DESCRIPTOR
//##############################################################################
// Dense Matrix

cusparseStatus_t CUSPARSELT_API
cusparseLtDenseDescriptorInit(const cusparseLtHandle_t*  handle,
                              cusparseLtMatDescriptor_t* matDescr,
                              int64_t                    rows,
                              int64_t                    cols,
                              int64_t                    ld,
                              uint32_t                   alignment,
                              cudaDataType               valueType,
                              cusparseOrder_t            order);

//------------------------------------------------------------------------------
// Structured Matrix

typedef enum {
    CUSPARSELT_SPARSITY_50_PERCENT
} cusparseLtSparsity_t;

cusparseStatus_t CUSPARSELT_API
cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t*  handle,
                                   cusparseLtMatDescriptor_t* matDescr,
                                   int64_t                    rows,
                                   int64_t                    cols,
                                   int64_t                    ld,
                                   uint32_t                   alignment,
                                   cudaDataType               valueType,
                                   cusparseOrder_t            order,
                                   cusparseLtSparsity_t       sparsity);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr);

//##############################################################################
//# MATMUL DESCRIPTOR
//##############################################################################

typedef enum {
    CUSPARSE_COMPUTE_16F,
    CUSPARSE_COMPUTE_32I,
    CUSPARSE_COMPUTE_TF32,
    CUSPARSE_COMPUTE_TF32_FAST
} cusparseComputeType;

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t*        handle,
                               cusparseLtMatmulDescriptor_t*    matMulDescr,
                               cusparseOperation_t              opA,
                               cusparseOperation_t              opB,
                               const cusparseLtMatDescriptor_t* matA,
                               const cusparseLtMatDescriptor_t* matB,
                               const cusparseLtMatDescriptor_t* matC,
                               const cusparseLtMatDescriptor_t* matD,
                               cusparseComputeType              computeType);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t*        handle,
                               cusparseLtMatmulDescriptor_t*    matMulDescr,
                               cusparseOperation_t              opA,
                               cusparseOperation_t              opB,
                               const cusparseLtMatDescriptor_t* matA,
                               const cusparseLtMatDescriptor_t* matB,
                               const cusparseLtMatDescriptor_t* matC,
                               const cusparseLtMatDescriptor_t* matD,
                               cusparseComputeType              computeType);

//##############################################################################
//# ALGORITHM SELECTION
//##############################################################################

typedef enum {
    CUSPARSELT_MATMUL_ALG_DEFAULT
} cusparseLtMatmulAlg_t;

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulAlgSelectionInit(
                            const cusparseLtHandle_t*           handle,
                            cusparseLtMatmulAlgSelection_t*     algSelection,
                            const cusparseLtMatmulDescriptor_t* matmulDescr,
                            cusparseLtMatmulAlg_t               alg);

typedef enum {
    CUSPARSELT_MATMUL_ALG_CONFIG_ID,     // READ/WRITE
    CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, // READ-ONLY
    CUSPARSELT_MATMUL_SEARCH_ITERATIONS  // READ/WRITE
} cusparseLtMatmulAlgAttribute_t;

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t*       handle,
                                cusparseLtMatmulAlgSelection_t* algSelection,
                                cusparseLtMatmulAlgAttribute_t  attribute,
                                const void*                     data,
                                size_t                          dataSize);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulAlgGetAttribute(
                            const cusparseLtHandle_t*             handle,
                            const cusparseLtMatmulAlgSelection_t* algSelection,
                            cusparseLtMatmulAlgAttribute_t        attribute,
                            void*                                 data,
                            size_t                                dataSize);

//##############################################################################
//# MATMUL PLAN
//##############################################################################

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulGetWorkspace(
                        const cusparseLtHandle_t*             handle,
                        const cusparseLtMatmulAlgSelection_t* algSelection,
                        size_t*                               workspaceSize);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulPlanInit(const cusparseLtHandle_t*             handle,
                         cusparseLtMatmulPlan_t*               plan,
                         const cusparseLtMatmulDescriptor_t*   matmulDescr,
                         const cusparseLtMatmulAlgSelection_t* algSelection,
                         size_t                                workspaceSize);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan);

//##############################################################################
//# MATMUL EXECUTION
//##############################################################################

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmul(const cusparseLtHandle_t*     handle,
                 const cusparseLtMatmulPlan_t* plan,
                 const void*                   alpha,
                 const void*                   d_A,
                 const void*                   d_B,
                 const void*                   beta,
                 const void*                   d_C,
                 void*                         d_D,
                 void*                         workspace,
                 cudaStream_t*                 streams,
                 int32_t                       numStreams);

cusparseStatus_t CUSPARSELT_API
cusparseLtMatmulSearch(const cusparseLtHandle_t* handle,
                       cusparseLtMatmulPlan_t*   plan,
                       const void*               alpha,
                       const void*               d_A,
                       const void*               d_B,
                       const void*               beta,
                       const void*               d_C,
                       void*                     d_D,
                       void*                     workspace,
                       cudaStream_t*             streams,
                       int32_t                   numStreams);

//##############################################################################
//# HELPER ROUTINES
//##############################################################################
// PRUNING

typedef enum {
    CUSPARSELT_PRUNE_SPMMA_TILE  = 0,
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1
} cusparseLtPruneAlg_t;

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMAPrune(const cusparseLtHandle_t*           handle,
                     const cusparseLtMatmulDescriptor_t* matmulDescr,
                     const void*                         d_in,
                     void*                               d_out,
                     cusparseLtPruneAlg_t                pruneAlg,
                     cudaStream_t                        stream);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t*           handle,
                          const cusparseLtMatmulDescriptor_t* matmulDescr,
                          const void*                         d_in,
                          int*                                valid,
                          cudaStream_t                        stream);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMAPrune2(const cusparseLtHandle_t*        handle,
                      const cusparseLtMatDescriptor_t* sparseMatDescr,
                      int                              isSparseA,
                      cusparseOperation_t              op,
                      const void*                      d_in,
                      void*                            d_out,
                      cusparseLtPruneAlg_t             pruneAlg,
                      cudaStream_t                     stream);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t*        handle,
                           const cusparseLtMatDescriptor_t* sparseMatDescr,
                           int                              isSparseA,
                           cusparseOperation_t              op,
                           const void*                      d_in,
                           int*                             d_valid,
                           cudaStream_t                     stream);

//------------------------------------------------------------------------------
// COMPRESSION

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMACompressedSize(const cusparseLtHandle_t*     handle,
                              const cusparseLtMatmulPlan_t* plan,
                              size_t*                       compressedSize);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMACompress(const cusparseLtHandle_t*     handle,
                        const cusparseLtMatmulPlan_t* plan,
                        const void*                   d_dense,
                        void*                         d_compressed,
                        cudaStream_t                  stream);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t*        handle,
                               const cusparseLtMatDescriptor_t* sparseMatDescr,
                               size_t*                          compressedSize);

cusparseStatus_t CUSPARSELT_API
cusparseLtSpMMACompress2(const cusparseLtHandle_t*        handle,
                         const cusparseLtMatDescriptor_t* sparseMatDescr,
                         int                              isSparseA,
                         cusparseOperation_t              op,
                         const void*                      d_dense,
                         void*                            d_compressed,
                         cudaStream_t                     stream);

//==============================================================================
//==============================================================================

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(CUSPARSELT_HEADER_)
