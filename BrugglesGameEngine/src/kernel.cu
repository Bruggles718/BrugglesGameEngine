#include "kernel.cuh"
#include <iostream>

__global__ void VectorAdd(int* a, int* b, int* c) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void WrapperVectorAdd() {
	int a[] = { 1, 2, 3 };
	int b[] = { 4, 5, 6 };
	int c[sizeof(a) / sizeof(int)] = { 0 };

	int* cudaA = 0;
	int* cudaB = 0;
	int* cudaC = 0;

	cudaMalloc(&cudaA, sizeof(a));
	cudaMalloc(&cudaB, sizeof(b));
	cudaMalloc(&cudaC, sizeof(c));

	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

	VectorAdd << <1, sizeof(a) / sizeof(int) >> > (cudaA, cudaB, cudaC);

	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

	for (int i = 0; i < sizeof(c) / sizeof(int); i++) {
		std::cout << c[i] << std::endl;
	}
}