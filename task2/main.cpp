#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
#include <omp.h>

bool parseArrayLine(const std::string &line, size_t expectedCount, std::vector<int> &out)
{
    std::istringstream iss(line);
    int value = 0;
    out.clear();

    while (iss >> value)
    {
        out.push_back(value);
    }

    return out.size() == expectedCount;
}

bool readVectorsFromFile(const std::string &fileName, size_t &N, std::vector<int> &A, std::vector<int> &B)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cerr << "Не удалось открыть файл: " << fileName << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(file, line))
    {
        std::cerr << "Файл пустой или отсутствует строка с N\n";
        return false;
    }

    std::istringstream nStream(line);
    if (!(nStream >> N) || N == 0)
    {
        std::cerr << "Некорректное значение N в первой строке\n";
        return false;
    }

    const size_t matrixElements = N * N;

    if (!std::getline(file, line) || !parseArrayLine(line, matrixElements, A))
    {
        std::cerr << "Некорректная строка для массива A: ожидалось " << matrixElements << " чисел\n";
        return false;
    }

    if (!std::getline(file, line) || !parseArrayLine(line, matrixElements, B))
    {
        std::cerr << "Некорректная строка для массива B: ожидалось " << matrixElements << " чисел\n";
        return false;
    }

    return true;
}

int main()
{
    omp_lock_t lock;
    omp_init_lock(&lock);
    int const TILE = 32;

    size_t N = 0;
    std::vector<int> A;
    std::vector<int> B;
    if (!readVectorsFromFile("vectors.txt", N, A, B))
    {
        omp_destroy_lock(&lock);
        return 1;
    }

    // long long traceSeq = 0;
    long long tracePar = 0;
    const int n = static_cast<int>(N);

    auto index = [N](int i, int j)
    {
        return i * N + j;
    };
    // // Последовательный подсчет trace(A * B): sum_i sum_k A[i,k] * B[k,i].
    // double start_time = omp_get_wtime();
    // for (int i = 0; i < n; i++)
    //     for (int k = 0; k < n; k++)
    //         traceSeq += 1LL * A[index(i, k)] * B[index(k, i)];
    // double end_time = omp_get_wtime();

    // std::cout << "Последовательно: " << traceSeq << " время : " << end_time - start_time << std::endl;

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        long long threadSum = 0;

#pragma omp for collapse(2) nowait
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
    std::cout << "Кол-во потоков = " << omp_get_max_threads() << "\nTrace(A*B) = " << tracePar << "\nВремя = " << end_time - start_time << std::endl;
    omp_destroy_lock(&lock);
    return 0;
}
