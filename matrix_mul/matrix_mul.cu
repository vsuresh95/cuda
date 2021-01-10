#include <cstdlib>
#include <cassert>
#include <iostream>

// __global__ indicates it will called from the host and run on the device
// __device__ is for device/device and __host__ for host/host
__global__ void matrixMul (float*a, float* b, float* c, int N)
{
    // get the global thread ID
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // put a predication to check whether the element for that thread
    // exists in the array
    if (row < N && col < N)
    {
        int temp = 0;
        for (int i = 0; i < N; i++)
        {
            temp += a[row*N + i] * b[i*N + col];
        }

        c[row*N + col] = temp;
    } 
}

int main ()
{
    int N = 1 << 7;
    size_t bytes = N * N * sizeof(float);

    // Allocate memory on the host size
    float *a, *b, *c;
    
    // using CUDA unified memory - we do not need to do memcpy to/from the GPU
    // this can be accessed from the host and device
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    for (int i = 0; i < N * N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // number of thread blocks and threads per block
    const int THREADS_1D = 16;
    const int BLOCKS_1D = (N + THREADS_1D - 1)/THREADS_1D;

    // setup kernel launch parameters
    dim3 THREADS (THREADS_1D, THREADS_1D);
    dim3 BLOCKS (BLOCKS_1D, BLOCKS_1D);

    void* args[4] = {&a, &b, &c, &N};

    cudaLaunchKernel((const void*) &matrixMul, BLOCKS, THREADS, (void**) &args);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int temp = 0;
            for (int k = 0; k < N; k++)
            {
                temp += a[i*N + k] * b[k*N + j];
            }
            assert(c[i*N + j] == temp); 
        }
    }

    std::cout << "Program completed!" << std::endl;

    return 0;
}