#include <iostream>
#include <vector>

int main()
{
    int const TILE = 32;

    size_t N = 4;
    std::vector<int> A = {12, 87, 34, 56, 23, 91, 5, 78, 44, 19, 62, 37, 80, 7, 50, 28, 66, 11, 73, 39};
    std::vector<int> B = {55, 2, 88, 14, 47, 90, 33, 61, 8, 76, 21, 64, 13, 95, 40, 6, 82, 29, 18, 70};
    std::vector<int> C(N * N, 0); // заполняем нулями
    int sum = 0;

    auto index = [N](int i, int j)
    {
        return i * N + j;
    };
    // последовательный обход
    for (int i = 0; i < (int)N; i++)
    {
        for (int r = 0; r < (int)N; r++)
        {
            sum += A[index(i, r)] * B[index(r, i)];
        };
    };
    std::cout << "Последовательно: " << sum << "\n";

#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < (int)N; ii += TILE)
        for (int jj = 0; jj < (int)N; jj += TILE)
            for (int kk = 0; kk < (int)N; kk += TILE)

                for (int i = ii; i < std::min(ii + TILE, (int)N); i++)
                    for (int j = jj; j < std::min(jj + TILE, (int)N); j++)
                    {
                        double sum = C[index(i, j)];
                        for (int k = kk; k < std::min(kk + TILE, (int)N); k++)
                            sum += A[index(i, k)] * B[index(k, j)];
                        C[index(i, j)] = sum;
                    }

    for (int i = 0; i < (int)N; i++)
    {
        for (int j = 0; j < (int)N; j++)
        {
            std::cout << C[index(i, j)] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "Паралельно: " << sum << "\n";
}