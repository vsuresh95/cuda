#include <iostream>
#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <sys/types.h>

using namespace std;

static void opencl_add_vector (benchmark::State& s)
{
	int N = 1 << s.range(0);

	size_t bytes = N * sizeof(float);

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[N];
    float *host_c = new float[N];

    // Allocate memory on the device side
    float *dev_a = new float[N];
    float *dev_b = new float[N];

    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);

    for (int i = 0; i < N; i++)
    {
        host_a[i] = rand() % 100;
        host_b[i] = rand() % 100;
    }

	// creating a handle
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	cublasSetVector(N, sizeof(float), host_a, 1, dev_a, 1);
	cublasSetVector(N, sizeof(float), host_b, 1, dev_b, 1);

	const float scale = 2.0f;
	while(s.KeepRunning())
	{
		cublasSaxpy_v2(handle, N, &scale, dev_a, 1, dev_b, 1);
	}

	cublasGetVector(N, sizeof(float), dev_b, 1, host_c, 1);

	cudaFree(dev_a);
	cudaFree(dev_b);
	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(opencl_add_vector) -> DenseRange(16, 20) -> Unit(benchmark::kMicrosecond);

static void naive_add_vector (benchmark::State& s)
{
	int N = 1 << s.range(0);

    // Allocate memory on the host side
    float *host_a = new float[N];
    float *host_b = new float[N];
    float *host_c = new float[N];

    for (int i = 0; i < N; i++)
    {
        host_a[i] = rand() % 100;
        host_b[i] = rand() % 100;
	}
	
	const float scale = 2.0f;

	while(s.KeepRunning())
	{
		for (int i = 0; i < N; i++)
		{
			host_c[i] = (scale * host_a[i]) + host_b[i];
		}
		benchmark::DoNotOptimize(host_c);
	}

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
}

BENCHMARK(naive_add_vector) -> DenseRange(16, 20) -> Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();