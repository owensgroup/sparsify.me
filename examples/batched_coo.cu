#include <cusparse.h>         // cusparseSpMM
#include <iostream>
#include <math.h>

#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <sparsify.me/util/util.hxx>
#include <sparsify.me/util/gen.hxx>
#include <sparsify.me/spmm.hxx>

struct prg
{
    int a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        int operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<int> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

int main(int argc, char** argv) {
    using type_t = float;
    if(argc != 5) {
      std::cout << "Invalid # of args. Usage: ./batched_coo m n k b" << std::endl;
      return EXIT_FAILURE;
    }

    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);
    int b = std::stoi(argv[4]);

    // Host problem definition
    int   A_num_rows   = m;
    int   A_num_cols   = k;
    int   A_nnz        = ceil(m * n * .5);
    int   B_num_rows   = A_num_cols;
    int   B_num_cols   = n;
    int   ldb          = B_num_rows;
    int   ldc          = A_num_rows;
    int   B_size       = ldb * B_num_cols;
    int   C_size       = ldc * B_num_cols;
    int   num_batches = b;
    
    // Singular Sparse Matrix (COO)
    thrust::device_vector<int> dA_rows(A_num_rows);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin,
            index_sequence_begin + A_num_rows,
            dA_rows.begin(),
            prg(0, A_num_rows-1));

    thrust::device_vector<int> dA_cols(A_num_cols);
    thrust::transform(index_sequence_begin,
            index_sequence_begin + A_num_cols,
            dA_cols.begin(),
            prg(0, A_num_cols-1));

    thrust::device_vector<float> dA_values(A_nnz);
    sparsifyme::util::random::uniform_distribution(dA_values, -1.f, 1.f);

    // Batch of Dense Matrices
    thrust::host_vector<thrust::device_vector<float>> dB_batches(num_batches);

    // Output Matrices
    thrust::host_vector<thrust::device_vector<float>> dC_batches(num_batches);
    
    // Batch Pointers
    float *dB, *dC;
    cudaMalloc((void**)&dB, sizeof(float) * num_batches * B_size);
    cudaMalloc((void**)&dC, sizeof(float) * num_batches * C_size);
    
    // Initialize Matrices
    for(int batch = 0; batch < num_batches; batch++) {
      auto& dBt = dB_batches[batch];
      dBt.resize(B_size);
      sparsifyme::util::random::uniform_distribution(dBt, -1.f, 1.f);
      cudaMemcpy(dB + batch * B_size, dBt.data().get(), B_size, cudaMemcpyDeviceToDevice);

      auto& dCt = dC_batches[batch];
      dCt.resize(C_size);
      cudaMemcpy(dC + batch * C_size, dCt.data().get(), C_size, cudaMemcpyDeviceToDevice);
    }
    

    
    float  alpha        = 1.0f;
    float  beta         = 0.0f;
    
    // Call cuSparse Strided Batched SpMM
    float elapsed = sparsifyme::batched::strided_coo(A_num_rows,
                                     A_num_cols,
                                     A_nnz,
                                     B_num_rows,
                                     B_num_cols,
                                     num_batches,
                                     dA_rows.data().get(),
                                     dA_cols.data().get(),
                                     dA_values.data().get(),
                                     dB, &dC, alpha, beta);
    std::cout << elapsed << std::endl;
}