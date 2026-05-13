#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__
void traceKernelGlobal(float* A, float* B, double* result, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (row < N && col < N && row == col) {

        for (int k = 0; k < N; k++) {
            local_sum += A[row * N + k] * B[k * N + col];
        }

        atomicAdd(result, local_sum);
    }
}

int main()
{
    const int N = 8192;
    size_t size = N * N * sizeof(float);

    float* A = new float[N * N];
    float* B = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    float *d_A, *d_B;
    double *d_result;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_result, sizeof(double));

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMemset(d_result, 0, sizeof(double));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    traceKernelGlobal<<<grid, block>>>(d_A, d_B, d_result, N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double result;

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Trace = " << result << std::endl;
    std::cout << "Time = " << ms / 1000.0 << " sec" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    delete[] A;
    delete[] B;

    return 0;
}