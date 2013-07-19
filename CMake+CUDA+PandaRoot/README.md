# CMake Examples
## CUDA in PandaRoot (ROOT / FairRoot) with CMake

All the previous work was just to set us up for this final example. How to use CUDA from a ROOT framework.

Let me first tell you a bit about PandaRoot, continue with the general idea and then go on to the actual implementation. You may want to skip the PandaRoot introduction.  
Alright?

Questions, remarks, additions -- submit them as **Issues** or **Pull Requests**. Or write me via some other means.

### PandaRoot
Or experiment ([PANDA](http://www-panda.gsi.de)) uses [CERN's ROOT](http://root.cern.ch) as its basic software framework.  
ROOT is customized by two layers. First, there's [FairRoot](http://fairroot.gsi.de/) which expands ROOT by some inter-experiment classes and structures. It's actually quite successful since there are some experiments using FairRoot which are not part even part of the name-giving FAIR project. Finally, customizing FairRoot for our PANDA needs is [PandaRoot](http://fairroot.gsi.de/?q=node/7).
Inside of PandaRoot, algorithms are developed and benchmarked, Monte Carlo studies are done and different experiment configurations are tested. It's used for pretty much everything.

The PandaRoot framework uses a two-fold approach to programming. Some classes are compiled, steering is done via macros. Everything important and essential is compiled and then called in the right order with right parameters by macros.  
Everything is arranged as `Task`s. `Task`s are the units called by macros (more precise, they are `Add`ed to the running chain). Mimicking a C++ structure, `Task`s need `Init`, `Exec`, and `Finish` member functions which are called by PandaRoot's automatic Task management (`Init` and `Finish` at beginning/end of a bunch of events, `Exec` once per event).

The idea: You code your important algorithms and pack them in classes. Then you make an associated `Task` for it, which kind of interprets between C++ and PandaRoot. The `Task`s are then called in the right way and onto the right data by adding them to a `.C` macro in the right order. 

Getting CUDA to PandaRoot is not that easy.
First of all: There are (at least) two different compilers involved. While all the PandaRoot CPU code uses gcc (or clang), CUDA uses its nvcc. But it doesn't stop there. Because of ROOT's *interactiveability* with macros and even command line input using CINT (or cling), there's lots of overhead to be taken care of.

### External C++ Functions
Sewing CUDA into PandaRoot exploits the feature of C++ of h[aving external defined functions](http://en.wikipedia.org/wiki/External_variable).  
The method has first been implemented by M. Al-Turany (as far as I see it) [for the base classes of FairRoot in 2009](https://subversion.gsi.de/trac/fairroot/browser/fairbase/release/cuda). Unfortunately they were a little hard to understand and for me didn't even work anymore.

The idea:

 * Make a CUDA function which does your device computing. Declare it `extern`. Call it, for example, `function_d()` (d is for device).
 * Make a plain C++ wrapper function for it. Call it, for example, `function()`. The wrapper function calls the device functions, eg. `function_d()`.
 * Your `Task` calls the wrapper `function()`.

So far the C++ side. The CMake side is analog to it:

 * Use CMake+FindCUDA to compile your device `function_d()`, but
    * Don't make a binary / executable file of it,
    * Only make a shared object.
 * Use the usual complicated CMake+ROOT stuff do compile your host `function()` and link the shared object to it.

### C++ and CMake Implementation
Let's go through some aspects of the example and explain the files provided in this repo (marked bold in this description).

#### C++
**example/cudaExample.cu** provides two exemplary functions: `DeviceInfo()` prints out some info about the used graphics card (and is shamelessly *borrowed* from the [here](https://subversion.gsi.de/trac/fairroot/browser/fairbase/release/cuda/cuda_imp/test_lib.cu)), `someOperation()` implements a square operation on the device. It's more or less the `squareOnDevice()` function from the `externalClass.cuh` file of the [CMake+CUDA example of this repository](https://github.com/AndiH/CMake/tree/master/CMake%2BCUDA). `someOperation()` prepares the data and calls the `square_array` kernel. Note the keywords in front of `DeviceInfo()` and `someOperation()`:

```C++
extern "C" void someOperation() {}
```

**PndCudaExampleTask.h/cxx** is the aforementioned task and wrapper class. It provides two methods which are called when `Exec()` is invoked:

 * `void callGpuStuff()` just calls `someOperation()` from `example/cudaExample.cu`
 * `void DeviceInfo_()` does the same for the device info function.

Sorry for the names.

Matching the `.cu` file, right on top of `PndCudaExampleTask.h/cxx` the external functions are again defined:

```C++
extern "C" void DeviceInfo();
extern "C" void someOperation();
```

**PndCudaLinkDef.h** provides info for ROOT's CINT compiler, ignore it just like **run_cudaExampleTask.C** which is a steering macro in the usual PandaRoot macro chain.

#### CMake
**example/CMakeLists.txt** is the file which compiles the CUDA stuff. It's quite extensively commented, but for the sake of it, let's go through some highlights.

 * `find_package(CUDA)` makes sure there's the needed CMake plugin for CUDA
 * `list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)` changes the NVCC compiler flag to use computing capability 2.0 (this is important since our kernel useses `printf`s)
 * `CUDA_ADD_LIBRARY(NAME SOURCES SHARED)` compiles the code but does not create a binary but a shared object file

To keep as much automated as possible I introduced a new variable into the CMake file: `PNDCUDAEXAMPLESONAME`. This is the name of our shared object file. It is set to *PndCudaExample* once, then set to the same value in the parent CMakeLists.txt scope again. Per default it's only available in the current scope. `CUDA_ADD_LIBRARY()` uses this variable to create the .so file, but we need it for a few things more… (just a second)

**CMakeLists.txt** is full of ROOT overhead of our big PandaRoot project. Important are:

 * `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example)` to add the `example/CMakeLists.txt` to the project
 * `add_dependencies(PndCuda ${PNDCUDAEXAMPLESONAME})` to prevent the current C++ CMake project from compiling before the CUDA CMake project is done – because then the .so file to link against wouldn't yet be created. Here you see why it's convenient to introduce a variable for the .so filename.
 * `target_link_libraries(PndCuda ${PNDCUDA_SRC} ${CMAKE_BINARY_DIR}/lib/lib${PNDCUDAEXAMPLESONAME}.so …)` finally links the .so file to our main project. Note the reusage of our .so variable name.


That's it.

Now you can use external CUDA functions from big C++ projects using CMake.

### Final Remarks

Don't forget to add

```C++
if(isLibrary("libPndCuda"))gSystem->Load("libPndCuda");
```

to your `rootlogon.C` loading macro in order to load the newly generated library into ROOT.
