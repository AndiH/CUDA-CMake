cmake_minimum_required(VERSION 2.8)

project(CUDACMAKETEST)

# set(CMAKE_MODULE_PATH "/usr/share/cmake/Modules" ${CMAKE_MODULE_PATH})

find_package(CUDA)

if (CUDA_FOUND)
	message("CUDA found, using device squaring!")
	add_subdirectory(src.gpu)
else()
	message("CUDA not found, doing something alternatively")
	add_subdirectory(src.cpu)
endif()


set(CMAKE_CXX_FLAG "-g -Wall")
