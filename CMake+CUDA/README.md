# CMake Examples
## CUDA with CMake

Fortunately someone already wrote a CMake plugin to enable CUDA support. It's called, in the usual CMake manner, FindCUDA.
There's an [SVN repository for it](https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2Ftrunk%2F). Check out the [FindCUDA.cmake file for an overview](https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2Ftrunk%2FCMake%2Fcuda%2FFindCUDA.cmake&view=markup) of all the macros and variables there are. As far as I see it, FindCUDA ships per default with CMake now.

### Generally
There are two easy and essential statements to get your CUDA code running through CMake. The first one is to actually find the CUDA package (resulting in a possible detection of the CUDA compiler):

```CMake
find_package(CUDA)
```

After the `CUDA` keyword a few more options can be specified. Like the version [of CUDA which is requiried](https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2Ftrunk%2FCMakeLists.txt&view=markup). And stuff.

An actual binary file is produced by 

```CMAKE
CUDA_ADD_EXECUTABLE(binaryFile sourceFile.cu)
```

And that's it.

Usually, the CUDA versions of the CMake macros have the same names as their original twins, except a `CUDA_` in front of the name.

#### CUDA-dependent compilation
Also nice to know:
`CUDA_FOUND` is a variable being true or false -- depending on the availability of the CUDA compiler. It can be used as a steering parameter. Eg. including different directories like in this example.

An alternative approach would be to put a precompiler statement into your source file which tests for `#ifdef __CUDACC__`. Depending on this you could even `#define` some functions as `__global__ void` instead of usual `void` and substitute `for (int i = 0; i < 42; i++) {}` by the usual `int idx = threadIdx.x + blockDim.x * blockIdx.x;`.

Ok, enough with this.

### This Example
If a CUDA compiler is found, this example compiles a squaring kernel `gpuSquareDemo`. It squares the from 0 to 9 and prints out the results (both via `printf` on the device and a `cout` on the host afterwards).

If no CUDA compiler is found, a program named `noCUDA` is built with a sad message.
