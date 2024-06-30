#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <oc/cuda-arrnd-header.h>

__host__ __device__ int compute_absolute_index(
    int rel_idx, int ndims, int offset, int* dims, int* strides, bool is_continuous)
{
    if (is_continuous) {
        return offset + rel_idx;
    }

    int abs_idx = 0;
    int temp_rel_idx = rel_idx;

    for (int i = ndims - 1; i >= 0; --i) {
        int rel_stride = 1;
        if (dims[i] > 1) {
            rel_stride *= dims[i];
            int sub = temp_rel_idx % dims[i];
            temp_rel_idx /= rel_stride;
            abs_idx += sub * strides[i];
        }
    }

    return offset + abs_idx;
}

__global__ void compute_absolute_indices_kernel(
    int ndims, int offset, int* dims, int* strides, bool is_continuous, int size, int* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        indices[i] = compute_absolute_index(i, ndims, offset, dims, strides, is_continuous);
    }
}

namespace oc {
void cpu_compute_absolute_indices(
    int ndims, int offset, int* dims, int* strides, bool is_continuous, int size, int* indices)
{
    for (int i = 0; i < size; ++i) {
        indices[i] = compute_absolute_index(i, ndims, offset, dims, strides, is_continuous);
    }
}

void gpu_compute_absolute_indices(
    int ndims, int offset, int* dims, int* strides, bool is_continuous, int size, int* indices)
{
    int* d_dims;
    cudaMalloc(&d_dims, ndims * sizeof(int));
    cudaMemcpy(d_dims, dims, ndims * sizeof(int), cudaMemcpyHostToDevice);

    int* d_strides;
    cudaMalloc(&d_strides, ndims * sizeof(int));
    cudaMemcpy(d_strides, strides, ndims * sizeof(int), cudaMemcpyHostToDevice);

    int* d_indices;
    cudaMalloc(&d_indices, size * sizeof(int));

    compute_absolute_indices_kernel<<<size / 256 + 1, 256>>>(
        ndims, offset, d_dims, d_strides, is_continuous, size, d_indices);

    cudaDeviceSynchronize();

    cudaMemcpy(indices, d_indices, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_dims);
    cudaFree(d_strides);
    cudaFree(d_indices);
}
}
