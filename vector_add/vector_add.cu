#include <cstdlib>
#include <cassert>
#include <iostream>

// __global__ indicates it will called from the host and run on the device
// __device__ is for device/device and __host__ for host/host
__global__ void vectorAdd (float*a, float* b, float* c, int N)
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

    // Allocate memory on the host size
    float *a, *b, *c;
    
    // using CUDA unified memory - we do not need to do memcpy to/from the GPU
    // this can be accessed from the host and device
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // number of thread blocks and threads per block
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1)/THREADS;

    void* args[4] = {&a, &b, &c, &N};

    cudaLaunchKernel((const void*) &vectorAdd, BLOCKS, THREADS, (void**) &args);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }

    std::cout << "Program completed!" << std::endl;

    return 0;
}