#include <iostream>
#include <omp.h>
#include <cstdlib>

int main()
{
    int N = 1000000000;
    long long x = 0;
    int *array = new int[N];
    for (int i = 0; i < N; ++i)
        array[i] = rand();

    double start_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : x)
    for (int i = 0; i < N; ++i)
    {
        x += array[i];
    }
    double end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    std::cout << "Parallel working time = " << parallel_time << std::endl;
    delete[] array;

    std::cout << "Difference parallel/single core = " << parallel_time << std::endl;
    std::cout << "Sum = " << x << std::endl;

    return 0;
}