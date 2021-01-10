#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

// __global__ indicates it will called from the host and run on the device
// __device__ is for device/device and __host__ for host/host
__global__ void vectorAdd (float* a, float* b, float* c, int N)
{
    // get the global thread ID
    int TID = blockIdx.x * blockDim.x + threadIdx.x;

    // put a predication to check whether the element for that thread
    // exists in the array
    if (TID < N)
    {
        c[TID] = a[TID] + b[TID];
    } 
}

int main ()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[N];
    float *host_c = new float[N];

    // Allocate memory on the device side
    float *dev_a = new float[N];
    float *dev_b = new float[N];
    float *dev_c = new float[N];

    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    for (int i = 0; i < N; i++)
    {
        host_a[i] = rand() % 100;
        host_b[i] = rand() % 100;
    }

    cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice);

    // number of thread blocks and threads per block
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1)/THREADS;

    void* args[4] = {&dev_a, &dev_b, &dev_c, &N};

    cudaLaunchKernel((const void*) &vectorAdd, BLOCKS, THREADS, (void**) &args);

    cudaDeviceSynchronize();

    cudaMemcpy(host_c, dev_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        assert(host_c[i] == host_a[i] + host_b[i]);
    }

    std::cout << "Program completed!" << std::endl;

    return 0;
}