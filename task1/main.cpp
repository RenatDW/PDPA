#include <iostream>
#include <omp.h>
#include <cstdlib>

class Util
{
public:
    static void fill_array_with_random_nums(int *array, int N)
    {
        for (int i = 0; i < N; ++i)
        {
            array[i] = rand();
        }
    }

    static void beatiful_output(double parallel_time, long long sum)
    {
        std::cout << "Parallel working time = " << parallel_time << std::endl;
        std::cout << "Sum = " << sum << std::endl;
    }
};

int main()
{
    int N = 1000000000;
    long long sum = 0;
    int *array = new int[N];

    Util::fill_array_with_random_nums(array, N);

    double start_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; ++i)
    {
        sum += array[i];
    }
    double end_time = omp_get_wtime();
    delete[] array;

    Util::beatiful_output(end_time - start_time, sum);
    return 0;
}
