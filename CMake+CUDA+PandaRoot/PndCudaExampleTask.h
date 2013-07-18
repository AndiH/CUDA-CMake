/*

 Test Task to try out some cuda in FairRoot
 Andreas Herten, June 2013

 */
#ifndef PNDCUDAEXAMPLETASK_H_
#define PNDCUDAEXAMPLETASK_H_

// framework includes
#include "FairTask.h"
#include "PndDetectorList.h"

#include <vector>
#include <map>

// Actual GPU device-using function have to be declared as extern
extern "C" void DeviceInfo();
extern "C" void someOperation();

class TClonesArray;

class PndCudaExampleTask : public FairTask
{
 public:

  /** Default constructor **/
	PndCudaExampleTask();

  /** Destructor **/
  virtual ~PndCudaExampleTask();


  /** Virtual method Init **/
  virtual void SetParContainers();
  virtual InitStatus Init();


  /** Virtual method Exec **/
  virtual void Exec(Option_t* opt);

  virtual void Finish();

protected:
  // For both external functions local representatives have to exist
	void callGpuStuff() {someOperation();};
  void DeviceInfo_() {return DeviceInfo();}

 private:
  
  void Register();

  void Reset();

  void ProduceHits();

  ClassDef(PndCudaExampleTask,1);

};

#endif
