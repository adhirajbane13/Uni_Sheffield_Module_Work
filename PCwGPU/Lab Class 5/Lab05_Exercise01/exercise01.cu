#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int* a);



__global__ void vectorAdd(int* a, int* b, int* c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}



int main(void) {
	//int *a,*b,*c,*c_ref;
	//int a[N], b[N], c[N], c_ref[N]; // Static arrays with size N; host copies of a, b, c
	int a[N], b[N], c[N];
	int* d_a, * d_b, * d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	random_ints(a);
	random_ints(b);

	//a = (int *)malloc(size); 
	//b = (int *)malloc(size); random_ints(b);
	//c = (int *)malloc(size);
	//c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Launch add() kernel on GPU
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Kernel Execution Time: %f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// Cleanup
	//free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	// Calculate Theoretical Bandwidth
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // Assuming you want to query the first CUDA device

	float theoreticalBW = (float)(prop.memoryClockRate) * 2 * prop.memoryBusWidth / (8 * 1024 * 1024 * 1024); // Convert to GB/s
	printf("Theoretical Bandwidth: %f GB/s\n", theoreticalBW);

	// Calculate Measured Bandwidth
	float RBytes = N * sizeof(int); // Number of bytes read by the kernel
	float WBytes = N * sizeof(int); // Number of bytes written by the kernel

	float measuredBW = (RBytes + WBytes) / (elapsedTime / 1000); // Convert elapsedTime to seconds
	printf("Measured Bandwidth: %f GB/s\n", measuredBW);

	return 0;
}

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int* a)
{
	for (unsigned int i = 0; i < N; i++) {
		a[i] = rand();
	}
}
