# sparsify.me
Simple header-only library for deep neural network sparsification. I set out to achieve the following formal goals:

 1. Given a pointer to a 2D-dense tensor, introduce sparsity to it.
 2. Achieved sparsity could be introduced in a number of ways:
     - **Blocked sparsity**, (NVIDIA's Ampere sparsity model, 2x2 for example or any size of the block),
     - **Unstructured sparsity**, presumably best-accuracy based sparsity pattern,
     - **Random sparsity**, randomly sparsify a random percent of the tensor.

## Enhancements
- Implement `im2col` fused with `sparsify`.
- Implement `SpMM` fused with `im2col` + `sparsify`.
- Python (pytorch) wrappers for these functions.

## `cusparseLt` install for `spmma`

### Ubuntu 20.04
```
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"  
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub  
sudo apt update  
sudo apt-get install libcusparselt0 libcusparselt-dev
```

### Ubuntu 18.04
```
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"  
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub  
sudo apt update  
sudo apt-get install libcusparselt0 libcusparselt-dev
```

## Quickstart
Needs CUDA version 11 or higher and an CUDA-capable GPU.
```bash
git clone https://github.com/neoblizz/sparsify.me.git
cd sparsify.me/examples
make
./bin/sparsify
```