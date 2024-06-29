#include <benchmark/benchmark.h>


#include <oc/cuda-arrnd-header.h>

static void BM_cpu_func(benchmark::State& state)
{
    for (auto _ : state) {
        cpu_func();
    }
}
BENCHMARK(BM_cpu_func);

static void BM_gpu_func(benchmark::State& state)
{
    for (auto _ : state) {
        gpu_func();
    }
}
BENCHMARK(BM_gpu_func);


BENCHMARK_MAIN();





