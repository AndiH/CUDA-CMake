#include "externalClass.cuh"

void externalClass::squareOnDevice(double *a_h, const int N) {
	double *a_d = new double[N]; // initialize a_d as an array with N double pointer
	size_t size = N * sizeof(double);
	cudaMalloc((void **) &a_d, size);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	square_array <<<n_blocks, block_size>>> (a_d, N);
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
}
