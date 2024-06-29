#include <oc/cuda_arrnd_header.h>

#define N (1024 * 1024)
#define M (1000)

int cpu_func()
{
    float* data = new float[N];
    for (int i = 0; i < N; i++) {
        data[i] = 1.0f * i / N;
        for (int j = 0; j < M; j++) {
            data[i] = data[i] * data[i] - 0.25f;
        }
    }
    int res = data[0];
    delete[] data;
    return res;
}

__global__ void cudakernel(float* buf)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    buf[i] = 1.0f * i / N;
    for (int j = 0; j < M; j++)
        buf[i] = buf[i] * buf[i] - 0.25f;
}

int gpu_func()
{
    float* data = new float[N];
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudakernel<<<N / 256, 256>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    int res = data[0];
    delete[] data;
    return res;
}