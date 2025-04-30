# Makefile for building CUDA-based Ed25519

NVCC := nvcc
CC := gcc
CFLAGS := -O3 -std=c99 -fPIC
CUDA_OPTS := -O3 -std=c++11 -arch=sm_70
VENDOR_DIR := vendor/ed25519_ref10
INCLUDES := -I$(VENDOR_DIR)

all: libed25519_cuda.so

# Build shared library with CUDA kernel and ref10 code
libed25519_cuda.so: gpu_ed25519.o ed25519_ref10.o
	$(NVCC) -shared -Xcompiler -fPIC gpu_ed25519.o ed25519_ref10.o -o $@

# Compile CUDA kernel
gpu_ed25519.o: gpu_ed25519.cu $(VENDOR_DIR)/ed25519_ref10.h
	$(NVCC) $(CUDA_OPTS) $(INCLUDES) -c $< -o $@

# Compile ref10 Ed25519 C code
ed25519_ref10.o: $(VENDOR_DIR)/ed25519_ref10.c $(VENDOR_DIR)/ed25519_ref10.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o libed25519_cuda.so
