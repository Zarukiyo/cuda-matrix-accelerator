#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// GPU Kernel for Matrix Multiplication
__global__ void matrixMulGPU(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU Baseline for Matrix Multiplication
void matrixMulCPU(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int N = 1024; // 1024x1024 matrix
    size_t bytes = N * N * sizeof(float);

    // Allocate Host (CPU) Memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_CPU = (float*)malloc(bytes);
    float *h_C_GPU = (float*)malloc(bytes);

    // Initialize matrices with arbitrary data
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate Device (GPU) Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define Grid and Block dimensions (16x16 threads per block)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Benchmark GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> durationGPU = endGPU - startGPU;

    cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost);

    // Benchmark CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;

    // Output Results
    std::cout << "Matrix Size: " << N << "x" << N << std::endl;
    std::cout << "CPU Execution Time: " << durationCPU.count() << " ms" << std::endl;
    std::cout << "GPU Execution Time: " << durationGPU.count() << " ms" << std::endl;
    std::cout << "Hardware Speedup: " << durationCPU.count() / durationGPU.count() << "x faster on GPU" << std::endl;

    // Free Memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);

    return 0;
}
