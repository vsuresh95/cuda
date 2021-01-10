#include <cstdlib>
#include <cassert>
#include <iostream>

// this will be the size of our 256 * 1 float array
// since we have 256 threads per thread block, we must size our
// shared mamory such that each thread can keep at least one element.
#define TILE_SIZE 256

// __global__ indicates it will called from the host and run on the device
// __device__ is for device/device and __host__ for host/host
__global__ void tiledmatrixMul (float*a, float* b, float* c, int N)
{
    __shared__ float tile_a[TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE];

    // get the global thread ID
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // put a predication to check whether the element for that thread
    // exists in the array
    if (row < N && col < N)
    {
        int temp = 0;
        for (int i = 0; i < N; i++)
        {
            // copy contents to tile_a and tile_b
            tile_a[col] = a[i*N + col];
            tile_b[row] = b[i + row*N];

            __syncthreads_count(N);
            
            temp += tile_a[i] * tile_b[i];

            __syncthreads_count(N);
        }

        c[row*N + col] = temp;
    } 
}

int main ()
{
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(float);

    // Allocate memory on the host side
    float *host_a = new float[N * N];
    float *host_b = new float[N * N];
    float *host_c = new float[N * N];

    // Allocate memory on the device side
    float *dev_a = new float[N * N];
    float *dev_b = new float[N * N];
    float *dev_c = new float[N * N];

    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = rand() % 100;
        host_b[i] = rand() % 100;
    }

    cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice);
   
    // number of thread blocks and threads per block
    const int THREADS_1D = 16;
    const int BLOCKS_1D = (N + THREADS_1D - 1)/THREADS_1D;

    // setup kernel launch parameters
    dim3 THREADS (THREADS_1D, THREADS_1D);
    dim3 BLOCKS (BLOCKS_1D, BLOCKS_1D);

    tiledmatrixMul<<<BLOCKS,THREADS>>>(dev_a, dev_b, dev_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(host_c, dev_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float temp = 0;
            for (int k = 0; k < N; k++)
            {
                temp += host_a[i*N + k] * host_b[k*N + j];
            }
            assert(host_c[i*N + j] == temp); 
        }
    }

    std::cout << "Program completed!" << std::endl;

    return 0;
}