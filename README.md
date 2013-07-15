# CMake Examples
For my PhD project I need to run GPU code out of our experiment's ROOT installation. That's not that easy.
But it's possible, yay!

## CMake
[CMake](http://www.cmake.org/) is a great tool. And it's a meta tool. CMake lets you generate Makefiles. Automatically, more or less.
There are a few steering keywords. You put them into a CMakeLists.txt and does the rest.

CMake not only simplifies creating Makefiles, it also has some other nice feature. Maybe most important is *out-of-source* build – have your source files in one directory, have the actual binaries in one completely different. Important especially for bigger projects.
Also, check out the official [CMake examples](http://www.cmake.org/cmake/help/examples.html).

## CMake Examples
This repo has examples of increasing complexity on how to use CMake, CMake+CUDA, CMake+CUDA+(Panda)ROOT.

The examples are:

 * Plain C++ library
 * C++ and CUDA combined
 * C++, CUDA, and ROOT / PandaRoot combined
 
See subdirectories for detailed descriptions.
