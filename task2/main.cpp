#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>

// Вывод __int128
void print_int128(__int128 x)
{
    if (x == 0)
    {
        std::cout << "0";
        return;
    }
    if (x < 0)
    {
        std::cout << "-";
        x = -x;
    }
    std::string s;
    while (x > 0)
    {
        s = char('0' + x % 10) + s;
        x /= 10;
    }
    std::cout << s;
}

int main(int argc, char *argv[])
{
    omp_lock_t lock;
    omp_init_lock(&lock);
    int const TILE = 32;

    size_t N = 0;
    if (argc != 2)
    {
        std::cerr << "Использование: " << argv[0] << " <N>\n";
        omp_destroy_lock(&lock);
        return 1;
    }

    char *endPtr = nullptr;
    unsigned long long parsedN = std::strtoull(argv[1], &endPtr, 10);
    if (endPtr == argv[1] || *endPtr != '\0' || parsedN == 0)
    {
        std::cerr << "Некорректное значение N: " << argv[1] << "\n";
        omp_destroy_lock(&lock);
        return 1;
    }
    N = static_cast<size_t>(parsedN);

    const size_t matrixElements = N * N;
    std::vector<int> A(matrixElements);
    std::vector<int> B(matrixElements);

    // Воспроизводимая генерация случайных чисел (фиксированный seed)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-10, 10);
    for (size_t idx = 0; idx < matrixElements; ++idx)
    {
        A[idx] = dist(rng);
        B[idx] = dist(rng);
    }

    // __int128 traceSeq = 0;
    __int128 tracePar = 0;
    const int n = static_cast<int>(N);

    auto index = [N](int i, int j)
    {
        return i * N + j;
    };

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        __int128 threadSum = 0;

#pragma omp for collapse(2) nowait schedule(static)
        for (int ii = 0; ii < n; ii += TILE)
            for (int kk = 0; kk < n; kk += TILE)
            {
                const int iEnd = std::min(ii + TILE, n);
                const int kEnd = std::min(kk + TILE, n);

                for (int i = ii; i < iEnd; i++)
                {
                    for (int k = kk; k < kEnd; k++)
                        threadSum += 1LL * A[index(i, k)] * B[index(k, i)];
                }
            }

        omp_set_lock(&lock);
        tracePar += threadSum;
        omp_unset_lock(&lock);
    }
    double end_time = omp_get_wtime();
    std::cout << "Кол-во потоков = " << omp_get_max_threads() << "\nTrace(A*B) = ";
    print_int128(tracePar);
    std::cout << "\nВремя = " << end_time - start_time << std::endl;
    omp_destroy_lock(&lock);
    return 0;
}
