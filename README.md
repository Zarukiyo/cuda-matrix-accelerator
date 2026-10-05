# cuda-matrix-accelerator
# CUDA Matrix Accelerator

A high-performance C++ benchmarking tool designed to measure the execution latency of matrix multiplication on standard CPUs versus NVIDIA GPU architectures. 

Built to demonstrate parallel thread block orchestration, host-to-device memory management, and hardware acceleration principles using the CUDA API.

## Architecture & Implementation
 **Host (CPU) Execution:** Standard single-threaded C++ nested loop implementation.
 **Device (GPU) Execution:** 2D Grid and Block topology. The workload is distributed across 16x16 thread blocks (256 threads per block) to maximize multiprocessor occupancy.
 **Memory Management:** Utilizes `cudaMalloc` and `cudaMemcpy` for explicit Host-to-Device (H2D) and Device-to-Host (D2H) data transfers.

## Benchmark Results
Tested on an NVIDIA T4 Tensor Core GPU (Google Colab Environment) with a 1024x1024 matrix size.

| Architecture | Execution Time | Performance Delta |
|--------------|----------------|-------------------|
| CPU Baseline | 3359.43 ms     | 1.0x              |
| NVIDIA T4 GPU| 8.2999 ms      | **404.7x Faster** |

## Build & Run Instructions
To compile and execute this benchmark on a Linux machine with `nvcc` installed:

```bash
nvcc -O3 matrix_mul.cu -o matrix_mul
./matrix_mul
