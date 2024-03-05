#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//The number of character in the encrypted text
#define N 1024

void checkCUDAError(const char*);
void read_encrypted_file(int*);


/* Exercise 1.1 */
__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void affine_decrypt(int *d_input, int *d_output)
{
	/* Exercise 1.2 */
	int t_ind = threadIdx.x;
	if (t_ind < N) {
		int encrypted_char = d_input[t_ind];
		int decrypted_char = modulo(111 * (encrypted_char - 27), 128);
		d_output[t_ind] = decrypted_char;
	}
}

__global__ void affine_decrypt_multiblock(int *d_input, int *d_output)
{
	/* Exercise 1.8 */
		int t_ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (t_ind < N) {
			int encrypted_char = d_input[t_ind];
			int decrypted_char = modulo(111 * (encrypted_char - 27), 128);
			d_output[t_ind] = decrypted_char;
		}
}


int main(int argc, char *argv[])
{
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;

	size = N * sizeof(int);

	/* allocate the host memory */
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);

	/* Exercise 1.3: allocate device memory */
	//cudaMalloc(???);
	//cudaMalloc(???);
	cudaMalloc((void**)&d_input, size);
	cudaMalloc((void**)&d_output, size);
	checkCUDAError("Memory allocation");

	/* read the encryted text */
	read_encrypted_file(h_input);

	/* Exercise 1.4: copy host input to device input */
	//cudaMemcpy(???);
	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	/* Exercise 1.5: Configure the grid of thread blocks and run the GPU kernel */
	//dim3 blocksPerGrid(???);
	//dim3 threadsPerBlock(???);
	//affine_decrypt(???);
	//dim3 blocksPerGrid(1);
	//dim3 threadsPerBlock(N);
	dim3 blocksPerGrid(8);
	dim3 threadsPerBlock(128);
	affine_decrypt_multiblock <<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);


	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("Kernel execution");

	/* Exercise 1.6: copy the gpu output back to the host */
	//cudaMemcpy(???);
	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

	checkCUDAError("Result transfer to host");

	/* print out the result to screen */
	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");

	/* Exercise 1.7: free device memory */
	//cudaFree(???);
	//cudaFree(???);
	cudaFree(d_input);
	cudaFree(d_output);

	checkCUDAError("Free memory");

	/* free host buffers */
	free(h_input);
	free(h_output);

	return 0;
}


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void read_encrypted_file(int* input)
{
	FILE *f = NULL;
	f = fopen("encrypted01.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find encrypted01.bin file \n");
		exit(1);
	}
	//read encrypted data
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}