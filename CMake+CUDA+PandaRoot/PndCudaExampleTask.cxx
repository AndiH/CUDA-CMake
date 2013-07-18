// -------------------------------------------------------------------------
// -----                 PndCudaExampleTask source file             -----
// -------------------------------------------------------------------------
// libc includes
#include <iostream>

// Root includes
#include "TROOT.h"
#include "TClonesArray.h"

// framework includes
#include "FairRootManager.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairHit.h"
#include "FairMultiLinkedData.h"
#include "FairEventHeader.h"
#include "PndCudaExampleTask.h"

#include "PndDetectorList.h"
#include <iomanip>

// -----   Default constructor   -------------------------------------------
PndCudaExampleTask::PndCudaExampleTask()
: FairTask("Calls a CUDA function")
{
}
// -------------------------------------------------------------------------

// -----   Destructor   ----------------------------------------------------
PndCudaExampleTask::~PndCudaExampleTask()
{
}

// -----   Public method Init   --------------------------------------------
InitStatus PndCudaExampleTask::Init()
{
  FairRootManager* ioman = FairRootManager::Instance();
  if (!ioman) {
    std::cout << "-E- PndCudaExampleTask::Init: "
    << "RootManager not instantiated!" << std::endl;
    return kFATAL;
  }

  std::cout << "-I- PndCudaExampleTask::Init: Initialization successfull" << std::endl;

  return kSUCCESS;
}

// -------------------------------------------------------------------------
void PndCudaExampleTask::SetParContainers()
{

}


// -----   Public method Exec   --------------------------------------------
void PndCudaExampleTask::Exec(Option_t* opt)
{
  std::cout << "============= PndCudaExampleTask:: START DEVICE INFO: " << std::endl;
  DeviceInfo_();
  std::cout << "============= PndCudaExampleTask:: END DEVICE INFO" << std::endl;
  std::cout << "============= PndCudaExampleTask:: START CUDA KERNEL CALL: " << std::endl;
  callGpuStuff();
  std::cout << "============= PndCudaExampleTask:: END CUDA KERNEL CALL" << std::endl;
  std::cout << "============= End PndCudaExampleTask::Exec" << std::endl;
  std::cout << std::endl;
}

void PndCudaExampleTask::Finish()
{

}

ClassImp(PndCudaExampleTask);
