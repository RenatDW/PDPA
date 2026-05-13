#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

namespace util
{
    void fill_array_with_random_nums(std::vector<int>& array)
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, RAND_MAX);

        for (auto& value : array)
        {
            value = dist(gen);
        }
    }
}

// CUDA kernel
__global__ void sum_reduction(const int* array, long long* partial_sums, size_t N)
{
    extern __shared__ long long shared[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Загружаем данные в shared memory
    shared[tid] = (idx < N) ? array[idx] : 0;

    __syncthreads();

    // Reduction внутри блока
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] += shared[tid + s];
        }

        __syncthreads();
    }

    // Первый поток блока записывает результат
    if (tid == 0)
    {
        partial_sums[blockIdx.x] = shared[0];
    }
}

int main()
{
    const size_t N = 1000000000;

    std::vector<int> h_array(N);

    util::fill_array_with_random_nums(h_array);

    int* d_array;
    cudaMalloc(&d_array, N * sizeof(int));

    cudaMemcpy(
        d_array,
        h_array.data(),
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    long long* d_partial_sums;
    cudaMalloc(&d_partial_sums, BLOCKS * sizeof(long long));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sum_reduction<<<BLOCKS, THREADS, THREADS * sizeof(long long)>>>(
        d_array,
        d_partial_sums,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Копируем partial sums обратно
    std::vector<long long> h_partial_sums(BLOCKS);

    cudaMemcpy(
        h_partial_sums.data(),
        d_partial_sums,
        BLOCKS * sizeof(long long),
        cudaMemcpyDeviceToHost
    );

    long long sum = 0;

    for (auto v : h_partial_sums)
    {
        sum += v;
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "CUDA time = " << ms / 1000.0 << " sec\n";
    std::cout << "Sum = " << sum << '\n';

    cudaFree(d_array);
    cudaFree(d_partial_sums);

    return 0;
}