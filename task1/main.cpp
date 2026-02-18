#include <iostream>
#include <random>
#include <omp.h>

namespace util
{
    void fill_array_with_random_nums(std::vector<int> &array)
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, RAND_MAX);

        for (auto &value : array)
        {
            value = dist(gen);
        }
    }

    void beautiful_output(double parallel_time, long long sum)
    {
        std::cout << "Parallel working time = " << parallel_time << " sec\n";
        std::cout << "Sum = " << sum << '\n';
    }
};

int main()
{
    const std::size_t N = 10000000;
    std::vector<int> array(N);
    long long sum = 0;

    util::fill_array_with_random_nums(array);

    double start_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (std::size_t i = 0; i < N; ++i)
    {
        sum += array[i];
    }
    double end_time = omp_get_wtime();

    util::beautiful_output(end_time - start_time, sum);
    return 0;
}
