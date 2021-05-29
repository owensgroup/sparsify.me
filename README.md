# sparsify.me
Simple header-only library for deep neural network sparse kernels.

## Getting Started

### Dependency: `cusparseLt` for `spmma`

#### Ubuntu 20.04
```
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"  
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub  
sudo apt update  
sudo apt-get install libcusparselt0 libcusparselt-dev
```

#### Ubuntu 18.04
```
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"  
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub  
sudo apt update  
sudo apt-get install libcusparselt0 libcusparselt-dev
```

### Quickstart
Needs CUDA version 11 or higher and an CUDA-capable GPU.
```bash
git clone https://github.com/neoblizz/sparsify.me.git
cd sparsify.me/examples
make
./bin/sparsify
```
