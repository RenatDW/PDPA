#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;

long long exponentiation_sequential(long long base, long long n) {
    if (n == 0) return 1;
    if (n == 1) return base;

    long long left = exponentiation_sequential(base, n / 2);
    long long right = exponentiation_sequential(base, n - n / 2);

    return left * right;
}

long long exponentiation_for_parallel(long long base, long long n, int iterations) {
    long long result = 0;

    #pragma omp parallel for reduction(^:result)
    for (int i = 0; i < iterations; i++) {
        long long r = exponentiation_sequential(base, n);
        result ^= r; // чтобы компилятор не выкинул
    }

    return result;
}

const int THRESHOLD = 20;

long long exponentiation_tasks(long long base, long long n) {
    if (n == 0) return 1;
    if (n == 1) return base;

    // threshold — обязательно
    if (n < THRESHOLD) {
        return exponentiation_sequential(base, n);
    }

    long long left = 0;
    long long right = 0;

    #pragma omp task shared(left) firstprivate(base, n)
    {
        left = exponentiation_tasks(base, n / 2);
    }

    #pragma omp task shared(right) firstprivate(base, n)
    {
        right = exponentiation_tasks(base, n - n / 2);
    }

    #pragma omp taskwait

    return left * right;
}


long long exponentiation_tasks_parallel(long long base, long long n) {
    long long result = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = exponentiation_tasks(base, n);
        }
    }

    return result;
}

struct TimingResult {
    double min_time;
    double max_time;
    double avg_time;
};

TimingResult benchmark_version(long long base, long long n, int num_runs,
                               long long (*func)(long long, long long)) {
    vector<double> times;
    static volatile long long sink = 0;

    for (int run = 0; run < num_runs; run++) {
        auto start = chrono::high_resolution_clock::now();
        long long r = func(base, n);
        sink ^= r;
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double>(end - start).count());
    }
    
    sort(times.begin(), times.end());
    
    double sum = 0;
    for (double t : times) sum += t;
    
    return {times.front(), times.back(), sum / num_runs};
}

void benchmark(long long base, long long n, int num_threads, int num_runs = 3) {
    omp_set_num_threads(num_threads);
    
    TimingResult seq = benchmark_version(base, n, num_runs, exponentiation_sequential);
    TimingResult tasks = benchmark_version(base, n, num_runs, exponentiation_tasks_parallel);
    TimingResult for_loops = benchmark_version(base, n, num_runs, exponentiation_for_parallel);
    
    cout << "Threads: " << num_threads << "\n";
    cout << "-----------\n";
    
    cout << "Sequential:  min=" << fixed << setprecision(6) << seq.min_time 
         << "s  avg=" << seq.avg_time << "s  max=" << seq.max_time << "s\n";
    cout << "Tasks:       min=" << fixed << setprecision(6) << tasks.min_time 
         << "s  avg=" << tasks.avg_time << "s  max=" << tasks.max_time << "s\n";
    cout << "For loops:   min=" << fixed << setprecision(6) << for_loops.min_time 
         << "s  avg=" << for_loops.avg_time << "s  max=" << for_loops.max_time << "s\n";
    
    cout << "Speedup (Tasks vs Seq):  " << fixed << setprecision(2) 
         << (seq.avg_time / tasks.avg_time) << "x\n";
    cout << "Speedup (For vs Seq):    " << fixed << setprecision(2) 
         << (seq.avg_time / for_loops.avg_time) << "x\n\n";
}
        

int main() {
    long long base = 37;
    long long n = 100;
    
    cout << "Fast Exponentiation with OpenMP Tasks\n";
    cout << "=====================================\n";
    cout << "Computing " << base << "^" << n << "\n";
    cout << "Max threads: " << omp_get_max_threads() << "\n\n";
    
    vector<int> thread_counts = {1, 2, 4, 8, 12};
    
    for (int threads : thread_counts) {
        if (threads <= omp_get_max_threads()) {
            benchmark(base, n, threads, 3);
        }
    }
    
    return 0;
}
