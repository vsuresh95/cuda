#include <iostream>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cassert>
#include <arm_neon.h>

using namespace std;

__global__ void conv1d (float* a, float* b, float* c, int N, int M)
{
	// get the global thread ID
    int TID = blockIdx.x * blockDim.x + threadIdx.x;

	if (TID < (N-M+1))
	{
		float temp = 0;
		for (int i = 0; i < M; i++)
		{
			temp += a[TID + i] * b[i];
		}
		c[TID] = temp;
	}
}

static void cuda_conv1d (benchmark::State& s)
{
	int N = 1 << s.range(0);
	int M = 1 << 5;

	size_t N_bytes = N * sizeof(float);
	size_t M_bytes = M * sizeof(float);
	size_t NM_bytes = (N-M+1) * sizeof(float);

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[M];
    float *host_c = new float[N-M+1];

    // Allocate memory on the device side
    float *dev_a = new float[N];
    float *dev_b = new float[M];
    float *dev_c = new float[N-M+1];

	// allocate memory for all the arrays and initialize with random numbers
	cudaMalloc(&dev_a, N_bytes);
    cudaMalloc(&dev_b, M_bytes);
    cudaMalloc(&dev_c, NM_bytes);

    for (int i = 0; i < N; i++)
    {
        host_a[i] = (rand() % 255) / 3.14;
    }

    for (int i = 0; i < M; i++)
    {
        host_b[i] = (rand() % 255) / 2.87;
    }

    cudaMemcpy(dev_a, host_a, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, M_bytes, cudaMemcpyHostToDevice);

    // number of thread blocks and threads per block
    const int THREADS = 256;
    const int BLOCKS = N/THREADS;

    void* args[5] = {&dev_a, &dev_b, &dev_c, &N, &M};

	while(s.KeepRunning())
	{
 		cudaLaunchKernel((const void*) &conv1d, BLOCKS, THREADS, (void**) &args); 
 		cudaDeviceSynchronize(); 
	}

    cudaMemcpy(host_c, dev_c, NM_bytes, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(cuda_conv1d) -> DenseRange(20, 26) -> Unit(benchmark::kMicrosecond);

static void naive_conv1d (benchmark::State& s)
{
	int N = 1 << s.range(0);
	int M = 1 << 5;

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[M];
    float *host_c = new float[N-M+1];

    for (int i = 0; i < N; i++)
    {
        host_a[i] = (rand() % 255) / 3.14;
	}
	
    for (int i = 0; i < M; i++)
    {
        host_a[i] = (rand() % 255) / 2.87;
	}
	
	while(s.KeepRunning())
	{ 
		for (int i = 0; i < N-M+1; i++)
		{
			float temp = 0;
			for (int j = 0; j < M; j++)
			{
				temp += host_a[i + j] * host_b[j];
			}
			host_c[i] += temp;
		}
		benchmark::DoNotOptimize(host_c);
	}

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(naive_conv1d) -> DenseRange(20, 26) -> Unit(benchmark::kMicrosecond);


static void neon_conv1d (benchmark::State& s)
{
	int N = 1 << s.range(0);
	int M = 1 << 5;

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[M];
    float *host_c = new float[N-M+1];

    for (int i = 0; i < N; i++)
    {
        host_a[i] = (rand() % 255) / 3.14;
	}
	
    for (int i = 0; i < M; i++)
    {
        host_b[i] = (rand() % 255) / 2.87;
	}

	float kernelSum = 0.0f;

    for (int i = 0; i < M; i++)
    {
        kernelSum += host_b[i];
	}

	float32x4_t kernelNeon;
	float32x4_t inputNeon;
	float32x4_t tempNeon;

	while(s.KeepRunning())
	{
		kernelNeon = vld1q_f32(host_b);
		for (int i = 0; i < N-M+1; i+=4)
		{
			float temp = 0;
			for (int j = 0; j < M; j++)
			{
				inputNeon = vld1q_f32(host_a + i + j);
				tempNeon = vmulq_f32(inputNeon, kernelNeon);temp += host_a[i + j] * host_b[j];
			}
			vst1q_f32(host_c + i,  tempNeon);
		}
		benchmark::DoNotOptimize(host_c);
	}

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(neon_conv1d) -> DenseRange(20, 26) -> Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();