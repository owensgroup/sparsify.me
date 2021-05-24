CUSPARSELT_DIR=../libs/libcusparse_lt
NVRTC_LIB=/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so

CC=nvcc
IDIR=$(CUSPARSELT_DIR)/include
LDIR=$(CUSPARSELT_DIR)/lib64 $(NVRTC_LIB)
LIBS=-lcusparseLt -lcusparse -ldl

spmma_example:
	$(CC) -o spmm_timing spmma_timing.cu -I$(IDIR) -L$(LDIR) $(LIBS)

clean:
	rm spmm_timing