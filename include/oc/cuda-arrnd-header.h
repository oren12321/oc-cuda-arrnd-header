#ifndef CUDA_ARRND_HEADER_H
#define CUDA_ARRND_HEADER_H

#include <vector>

namespace oc {
void cpu_compute_absolute_indices(
    int ndims, int offset, int* dims, int* strides, bool is_continuous, int size, int* indices);
void gpu_compute_absolute_indices(
    int ndims, int offset, int* dims, int* strides, bool is_continuous, int size, int* indices);
}

#endif // CUDA_ARRND_HEADER_H