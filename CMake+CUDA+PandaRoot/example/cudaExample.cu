#include <stdio.h>
#include <assert.h>  
#include <cuda.h>  
#include <iostream>


__global__ void square_array(double *a, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
  printf("idx = %d, a = %f\n", idx, a[idx]);
}


extern "C" void someOperation() {
  const int localN = 10;
  double * localFloat = new double[localN];
  for (int i = 0; i < localN; i++)
    localFloat[i] = i;

  double *a_d = new double[localN]; // initialize a_d as an array with N double pointer
  size_t size = localN * sizeof(double);
  cudaMalloc((void **) &a_d, size);
  cudaMemcpy(a_d, localFloat, size, cudaMemcpyHostToDevice);

  int block_size = 4;
  int n_blocks = localN/block_size + (localN%block_size == 0 ? 0:1);
  square_array <<<n_blocks, block_size>>> (a_d, localN);
  cudaMemcpy(localFloat, a_d, size, cudaMemcpyDeviceToHost);
  cudaFree(a_d);

  std::cout << "Host side output: " << std::endl;
  for (int i = 0; i < localN; i++)
    std::cout << localFloat[i] << std::endl;
}
  
extern "C" void DeviceInfo(void) {  
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      if (dev == 0) {
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
          printf("There is no device supporting CUDA.\n");
        else if (deviceCount == 1)
           printf("There is 1 device supporting CUDA\n");
        else
          printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n",
         deviceProp.major);
        printf("  Minor revision number:                         %d\n",
         deviceProp.minor);
        printf("  Total amount of global memory:                 %u bytes\n",
         deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n",
         deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n",
         8 * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %u bytes\n",
         deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n",
         deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
         deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
         deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n",
         deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
         deviceProp.maxThreadsDim[0],
         deviceProp.maxThreadsDim[1],
         deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
         deviceProp.maxGridSize[0],
         deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n",
         deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n",
         deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n",
         deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n",
         deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
      }
      printf("\nTest PASSED\n");
}

