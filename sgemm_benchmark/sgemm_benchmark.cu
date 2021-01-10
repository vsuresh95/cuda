#include <iostream>
#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/types.h>

using namespace std;

static void cuda_mul_matrix (benchmark::State& s)
{
	int N = 1 << s.range(0);

	size_t bytes = N * N * sizeof(float);

    // Allocate memory on the host side
    float *host_a = new float[N * N];
    float *host_b = new float[N * N];
    float *host_c = new float[N * N];

    // Allocate memory on the device side
    float *dev_a;
    float *dev_b;
    float *dev_c;

    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

	// generate random numbers using a system clock seed
	curandGenerator_t prng; 
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// use the random number generator to generate values for our device pointers
	curandGenerateUniform(prng, dev_a, N * N);
	curandGenerateUniform(prng, dev_b, N * N);
	curandGenerateUniform(prng, dev_c, N * N);

	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	const float alpha = 2.0f;
	const float beta = 3.0f;
	while(s.KeepRunning())
	{
		cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_a, N, dev_b, N, &beta, dev_c, N);
	}

	cublasGetVector(N * N, sizeof(float), dev_c, 1, host_c, 1);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(cuda_mul_matrix) -> DenseRange(6, 10) -> Unit(benchmark::kMicrosecond);

static void naive_mul_matrix (benchmark::State& s)
{
	int N = 1 << s.range(0);

    // Allocate memory on the host side
    float *host_a = new float[N * N];
    float *host_b = new float[N * N];
    float *host_c = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = rand() % 100;
        host_b[i] = rand() % 100;
        host_c[i] = rand() % 100;
	}
	
	const float alpha = 2.0f;
	const float beta = 3.0f;

	while(s.KeepRunning())
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				int temp;
				for (int k = 0; k < N; k++)
				{
					temp += (alpha * host_a[i*N + k]) * host_b[k*N + j];
				}
				host_c[i*N + j] = temp + (beta * host_c[i*N +j]);
			}
		}
		benchmark::DoNotOptimize(host_c);
	}

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(naive_mul_matrix) -> DenseRange(6, 10) -> Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();