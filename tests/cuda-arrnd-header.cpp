#include <gtest/gtest.h>

#include <vector>
#include <numeric>

#include <oc/cuda-arrnd-header.h>


TEST(compute_absolute_indices, continuous_indices)
{
    using namespace std;
    using namespace oc;

    vector<int> dims{3, 1, 2};
    vector<int> strides{2, 2, 1};
    int offset = 0;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    cpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), true, size, cpu_indices.data());

    vector<int> gpu_indices(size);
    gpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), true, size, gpu_indices.data());

    EXPECT_EQ(cpu_indices, (std::vector<int>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(cpu_indices, gpu_indices);
}

TEST(compute_absolute_indices, non_continuous_indices)
{
    using namespace std;
    using namespace oc;

    vector<int> dims{2, 1, 1};
    vector<int> strides{2, 2, 1};
    int offset = 3;

    int size = std::reduce(dims.cbegin(), dims.cend(), 1, std::multiplies<>{});

    vector<int> cpu_indices(size);
    cpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), false, size, cpu_indices.data());

    vector<int> gpu_indices(size);
    gpu_compute_absolute_indices(dims.size(), offset, dims.data(), strides.data(), false, size, gpu_indices.data());

    EXPECT_EQ(cpu_indices, (std::vector<int>{3, 5}));
    EXPECT_EQ(cpu_indices, gpu_indices);
}
