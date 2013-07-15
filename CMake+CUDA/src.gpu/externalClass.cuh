#ifndef EXTERNALCLASS_CUH_
#define EXTERNALCLASS_CUH_

#include <iostream>
#include <stdio.h>


__global__ void square_array(double *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = a[idx] * a[idx];
	printf("GPU PRINT: idx = %d, a = %f\n", idx, a[idx]);
}

class externalClass {

public:
	int GetInt() { return 5; };

	void squareOnDevice(double *a_h, const int N);
};

#endif
