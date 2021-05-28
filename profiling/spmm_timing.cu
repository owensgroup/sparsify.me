#include <cstdio>             
#include <cstdlib>            
#include <sparsify.me/ampere.hxx>

#define CHECK_CUSPARSE(func)                                                   
{                                                                              
    cusparseStatus_t status = (func);                                          
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",        
               __LINE__, cusparseGetErrorString(status), status);              
        return EXIT_FAILURE;                                                   
    }                                                                          
}

int main(int argc, char** argv) {
  using namespace sparsifyme;
  using type_t = float;

  if(argc != 5) {
    throw "Invalid # of args. Usage: ./spmm_timing m n k b";
  }

  int major_cc, minor_cc;
  cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0);
  if (!(major_cc == 8 && minor_cc == 0)) {
      std::printf("\ncusparseLt is supported only on GPU devices with"
                  " compute capability == 8.0, current: %d.%d\n\n",
                   major_cc, minor_cc);
      return EXIT_UNSUPPORTED;
  }

  cudaDataType valueType;

  std::size_t m = std::stoi(argv[2]);
  std::size_t n = std::stoi(argv[3]);
  std::size_t k = std::stoi(argv[4]);


  // Initialize host pointers
  type_t* hA = malloc(m * k * sizeof(type_t));
  type_t* hB = malloc(k * n * sizeof(type_t));
  type_t* hC = malloc(m * n * sizeof(type_t));
  for(int i = 0; i < m * k; i++) {
    hA[i] = static_cast<type_t>(static_cast<float>(std::rand() % 100));
  }
  for(int i = 0; i < m * k; i++) {
    hB[i] = static_cast<type_t>(static_cast<float>(std::rand() % 100));
  }
  float alpha = 1.0f;
  float beta = 0.f;

  // Allocate device pointers and move data to GPU from host pointers
  type_t* dA, dB, dC;
  cudaMalloc((**void)&dA, m*k*sizeof(type_t));
  cudaMalloc((**void)&dB, k*n*sizeof(type_t));
  cudaMalloc((**void)&dC, m*n*sizeof(type_t));

  cudaMemcpy(dA,hA,m*k*sizeof(type_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,k*n*sizeof(type_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dC,hC,m*n*sizeof(type_t), cudaMemcpyHostToDevice);

  // Call Ampere Functions
  auto times = sparsifyme::ampere_spmm(&dA, &dB, &dC, m,n,k);
  // std::cout << "Matrix Sizes (m, n, k) = (" << m << ", " << n << ", " << k << ")" << std::endl;
  std::cout << times[0] << ", " << times[1] << ", " << times[2] << std::endl;

}