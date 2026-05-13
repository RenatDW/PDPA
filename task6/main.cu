#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__device__ unsigned long long fastPowGPU(
    unsigned long long a,
    unsigned int n)
{
    unsigned long long result = 1;

    while (n > 0)
    {
        if (n & 1)
            result *= a;

        a *= a;
        n >>= 1;
    }

    return result;
}

__global__ void kernel(
    unsigned long long* a,
    unsigned int* n,
    unsigned long long* result,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M)
    {
        result[idx] = fastPowGPU(a[idx], n[idx]);
    }
}

int main()
{
    const int M = 100000000;

    size_t sizeA = M * sizeof(unsigned long long);
    size_t sizeN = M * sizeof(unsigned int);

    unsigned long long* h_a =
        (unsigned long long*)malloc(sizeA);

    unsigned int* h_n =
        (unsigned int*)malloc(sizeN);

    unsigned long long* h_result =
        (unsigned long long*)malloc(sizeA);

    for (int i = 0; i < M; i++)
    {
        h_a[i] = 2 + i % 10;
        h_n[i] = 20 + i % 10;
    }

    unsigned long long *d_a, *d_result;
    unsigned int* d_n;

    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_n, sizeN);
    cudaMalloc(&d_result, sizeA);

    cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, sizeN, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    kernel<<<blocks, threads>>>(d_a, d_n, d_result, M);

    cudaEventRecord(stop);

    cudaMemcpy(
        h_result,
        d_result,
        sizeA,
        cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    cout << "CUDA Time: "
         << milliseconds / 1000.0
         << " sec\n";

    cudaFree(d_a);
    cudaFree(d_n);
    cudaFree(d_result);

    free(h_a);
    free(h_n);
    free(h_result);

    return 0;
}