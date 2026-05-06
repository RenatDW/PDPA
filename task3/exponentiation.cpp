#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>

using namespace std;

long long exponentiation_sequential(long long base, long long n) {
    if (n == 0)
        return 1;
    if (n == 1)
        return base;
    
    if (n % 2 == 0) {
        long long half = exponentiation_sequential(base, n / 2);
        return half * half;
    } else {
        return base * exponentiation_sequential(base, n - 1);
    }
}

long long exponentiation_tasks(long long base, long long n) {
    if (n == 0)
        return 1;
    if (n == 1)
        return base;
    
    if (n % 2 == 0) {
        long long half_result = 0;
        
        #pragma omp task shared(half_result) firstprivate(base, n)
        {
            half_result = exponentiation_tasks(base, n / 2);
        }
        
        #pragma omp taskwait
        return half_result * half_result;
    } else {
        long long recursive_result = 0;
        
        #pragma omp task shared(recursive_result) firstprivate(base, n)
        {
            recursive_result = exponentiation_tasks(base, n - 1);
        }
        
        #pragma omp taskwait
        return base * recursive_result;
    }
}

long long exponentiation_tasks_parallel(long long base, long long n) {
    long long result = 0;
    #pragma omp parallel shared(result) firstprivate(base, n)
    {
        #pragma omp single
        {
            result = exponentiation_tasks(base, n);
        }
    }
    return result;
}

void benchmark(long long base, long long n, int num_runs = 5) {
    double seq_time = 0, tasks_time = 0;
    
    for (int run = 0; run < num_runs; run++) {
        auto start = chrono::high_resolution_clock::now();
        exponentiation_sequential(base, n);
        auto end = chrono::high_resolution_clock::now();
        seq_time += chrono::duration<double>(end - start).count();
        
        start = chrono::high_resolution_clock::now();
        exponentiation_tasks_parallel(base, n);
        end = chrono::high_resolution_clock::now();
        tasks_time += chrono::duration<double>(end - start).count();
    }
    
    seq_time /= num_runs;
    tasks_time /= num_runs;
    
    cout << "Sequential:  " << fixed << setprecision(6) << seq_time << "s\n";
    cout << "Tasks:       " << fixed << setprecision(6) << tasks_time << "s\n";
    cout << "Speedup:     " << fixed << setprecision(2) << (seq_time / tasks_time) << "x\n";
}

int main() {
    long long base = 2;
    long long n = 40;
    
    cout << "Fast Exponentiation with OpenMP Tasks\n";
    cout << "=====================================\n";
    cout << "Computing " << base << "^" << n << "\n";
    cout << "Max threads: " << omp_get_max_threads() << "\n\n";
    
    benchmark(base, n, 5);
    
    return 0;
}
