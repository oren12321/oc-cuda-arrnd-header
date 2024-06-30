#include <benchmark/benchmark.h>

#include <vector>
#include <numeric>

#include <oc/cuda-arrnd-header.h>

void compute_continuous_full_hd_frame_indices_cpu()
{
    using namespace std;
    using namespace oc;

    vector<int> dims{4, 1920, 1080};
    vector<int> strides{2073600, 1080, 1};
    int offset = 0;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    cpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), true, size, cpu_indices.data());
}

static void BM_continuous_indices_cpu(benchmark::State& state)
{
    for (auto _ : state) {
        compute_continuous_full_hd_frame_indices_cpu();
    }
}
BENCHMARK(BM_continuous_indices_cpu);

void compute_continuous_full_hd_frame_indices_gpu()
{
    using namespace std;
    using namespace oc;

    vector<int> dims{4, 1920, 1080};
    vector<int> strides{2073600, 1080, 1};
    int offset = 0;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    gpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), true, size, cpu_indices.data());
}

static void BM_continuous_indices_gpu(benchmark::State& state)
{
    for (auto _ : state) {
        compute_continuous_full_hd_frame_indices_gpu();
    }
}
BENCHMARK(BM_continuous_indices_gpu);

void compute_non_continuous_full_hd_frame_indices_cpu()
{
    using namespace std;
    using namespace oc;

    vector<int> dims{4, 1920, 1080};
    vector<int> strides{2073600, 1080, 1};
    int offset = 0;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    cpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), false, size, cpu_indices.data());
}

static void BM_non_continuous_indices_cpu(benchmark::State& state)
{
    for (auto _ : state) {
        compute_non_continuous_full_hd_frame_indices_cpu();
    }
}
BENCHMARK(BM_non_continuous_indices_cpu);

void compute_non_continuous_full_hd_frame_indices_gpu()
{
    using namespace std;
    using namespace oc;

    vector<int> dims{4, 1920, 1080};
    vector<int> strides{2073600, 1080, 1};
    int offset = 0;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    gpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), false, size, cpu_indices.data());
}

static void BM_non_continuous_indices_gpu(benchmark::State& state)
{
    for (auto _ : state) {
        compute_non_continuous_full_hd_frame_indices_gpu();
    }
}
BENCHMARK(BM_non_continuous_indices_gpu);
