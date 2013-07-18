void run_ana_complete(TString FileName="test"){
  // ========================================================================

  gROOT->LoadMacro("$VMCWORKDIR/gconfig/rootlogon.C");
  rootlogon();

	  Int_t iVerbose = 0;

	  PndFileNameCreator creator(FileName.Data());
	  std::cout << 	creator.GetParFileName() 	<< std::endl;
	  std::cout <<	creator.GetSimFileName()		<< std::endl;
	  std::cout <<	creator.GetDigiFileName()		<< std::endl;
	  std::cout <<	creator.GetRecoFileName()		<< std::endl;
	  std::cout <<	creator.GetCustomFileName("blubb")<< std::endl;


	  TString MCFile 	= creator.GetSimFileName();
	  TString parFile 	= creator.GetParFileName() ;
	  Int_t nEvents =0;
	  // ----  Load libraries   -------------------------------------------------

	  gROOT->Macro("$VMCWORKDIR/gconfig/rootlogon.C");
	  // ------------------------------------------------------------------------
	  // Output file

	  TString DigiFile = creator.GetDigiFileName();
	  TString RecoFile = creator.GetRecoFileName();

	  TString outFile = creator.GetCustomFileName("anareco");

	  std::cout << "MCFile  : " << MCFile.Data()<< std::endl;
	  std::cout << "DigiFile: " << DigiFile.Data()<< std::endl;
	  std::cout << "RecoFile: " << RecoFile.Data()<< std::endl;

	  gSystem->Load("libMCMatch");
	  gSystem->Load("libMCMatchExamples");


  // -----   Timer   --------------------------------------------------------
  TStopwatch timer;
  timer.Start();

  // -----   Reconstruction run   -------------------------------------------
  FairRunAna *fRun= new FairRunAna();

  fRun->SetInputFile(MCFile);
  fRun->AddFriend(DigiFile);
  fRun->AddFriend(RecoFile);

  fRun->SetOutputFile(outFile);

  // -----  Parameter database   --------------------------------------------
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();

  FairParRootFileIo* parInput1 = new FairParRootFileIo(kTRUE);
  parInput1->open(parFile.Data(),"UPDATE");

  rtdb->setFirstInput(parInput1);
  Bool_t kParameterMerged=kTRUE;
  rtdb->setOutput(parInput1);
  rtdb->print();

  // ------------------------------------------------------------------------

//  PndSttMvdGemTrackingIdeal* trackStt = new PndSttMvdGemTrackingIdeal();
//  trackStt->SetRelativeMomentumSmearing(0.05);
//  trackStt->SetVertexSmearing(0.05, 0.05, 0.05);
//  trackStt->SetTrackingEfficiency(1.);
//  trackStt->SetTrackOutput("SttMvdGemIdealTrack");
//  fRun->AddTask(trackStt);


// PndSttMvdGemTracking Ideal vor Pnd MCMatchCreator findet PndMCIdealTrackFinder mehr
 // dahinter weniger


  // PndMCMatchCreatorTask* mcMatch = new PndMCMatchCreatorTask();
  // mcMatch->SetVerbose(0);
  // fRun->AddTask(mcMatch);



  // PndMCIdealTrackFinder* idealtrackfinder = new PndMCIdealTrackFinder();
  // fRun->AddTask(idealtrackfinder);

  PndCudaExampleTask* cudaDummy = new PndCudaExampleTask();
  fRun->AddTask(cudaDummy);

  fRun->Init();
  fRun->Run(0,nEvents);

  // -----   Finish   -------------------------------------------------------
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
  cout << "Output file is "    << outFile << endl;
  cout << "Parameter file is " << parFile << endl;
  cout << "Real time " << rtime << " s, CPU time " << ctime << " s" << endl;
  cout << endl;
  // ------------------------------------------------------------------------
}
