# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

set(CUDASRCS 
	main.cu 
)

CUDA_ADD_EXECUTABLE(gpuSquareDemo ${CUDASRCS})


# Some untested, alternative statements of how to add libraries to an CUDA executable

# CUDA_ADD_LIBRARY(externals externalStuff.cu)
# 
# target_link_libraries(gpuSquareDemo externals)
