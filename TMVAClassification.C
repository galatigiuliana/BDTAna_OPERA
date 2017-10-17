/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides examples for the training and testing of the
/// TMVA classifiers.
///
/// As input data is used a toy-MC sample consisting of four Gaussian-distributed
/// and linearly correlated input variables.
/// The methods to be used can be switched on and off by means of booleans, or
/// via the prompt command, for example:
///
///     root -l ./TMVAClassification.C\(\"Fisher,Likelihood\"\)
///
/// (note that the backslashes are mandatory)
/// If no method given, a default set of classifiers is used.
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// invoked through a GUI that will appear at the end of the run of this macro.
/// Launch the GUI via the command:
///
///     root -l ./TMVAGui.C
///
/// You can also compile and run the example with the following commands
///
///     make
///     ./TMVAClassification <Methods>
///
/// where: `<Methods> = "method1 method2"` are the TMVA classifier names
/// example:
///
///     ./TMVAClassification Fisher LikelihoodPCA BDT
///
/// If no method given, a default set is of classifiers is used
///
/// - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Root Macro: TMVAClassification
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker


#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int TMVAClassification( TString myMethodList = "" )
{
    
    int channel=-99;
    cout << "Select channel: (1 = tau->1h (ALL), 2 = tau->mu (ALL), 3 = tau->3h (ALL), 4 = tau->e (ALL))" << endl;
    cin >> channel;
    
    float nexp_S_1h =2.96;
    float nexp_B_1h =1.43;
    
    float nexp_S_mu =1.15;
    float nexp_B_mu =0.024;
    
    float nexp_S_3h =1.83;
    float nexp_B_3h =0.52;
    
    float nexp_S_e =0.84;
    float nexp_B_e =0.035;
    
    float charmfraction[5]={0, 0.1080, 0.3192, 0.8344, 1};
    float tauDISfraction[5]={0, 0.9563, 0.9060, 0.8231, 0.9374};
    //    float charmfraction[5]={0, 0.1080, 0.7419, 0.8344, 1};
    //    float tauDISfraction[5]={0, 0.9563, 0.9090, 0.8231, 0.9374};
    
    int goldensilver=1;
    
    //1h ->TAGLIO SUGGERITO: -0.110769 sig: 0.922484%, bkg: 0.814547%
    //mu ->TAGLIO SUGGERITO: -0.261667 sig: 0.995654%, bkg: 0.3404%
    //3h ->TAGLIO SUGGERITO: -0.233846 sig: 0.97734%, bkg: 0.203334%
    
    TString fname, fname_S_DIS, fname_S_QE, fname_bkg1, fname_bkg2;
    
    if (channel==1) {
        //fname = "./datarootfiles/bdt_kinematics_1h.root";
        fname_S_DIS = "./datarootfiles/bdt_kinematics_1_0_0.root";
        fname_S_QE = "./datarootfiles/bdt_kinematics_1_1_0.root";
        fname_bkg1 = "./datarootfiles/bdt_kinematics_5_0_0.root";
        fname_bkg2 = "./datarootfiles/bdt_kinematics_21_0.root";
    }
    else if (channel==2||channel==2.5) {
        //fname = "./datarootfiles/bdt_kinematics_mu.root";
        fname_S_DIS = "./datarootfiles/bdt_kinematics_2_0_0.root";
        fname_S_QE = "./datarootfiles/bdt_kinematics_2_1_0.root";
        fname_bkg1 = "./datarootfiles/bdt_kinematics_6_0_0.root";
        fname_bkg2 = "./datarootfiles/bdt_kinematics_20_0_0.root";
    }
    else if (channel==3) {
        //fname = "./datarootfiles/bdt_kinematics_3h.root";
        fname_S_DIS = "./datarootfiles/bdt_kinematics_3_0_0.root";
        fname_S_QE = "./datarootfiles/bdt_kinematics_3_1_0.root";
        fname_bkg1 = "./datarootfiles/bdt_kinematics_7_0_0.root";
        fname_bkg2 = "./datarootfiles/bdt_kinematics_22_0.root";
    }
    else if (channel==4) {
        //fname = "./datarootfiles/bdt_kinematics_e.root";
        fname_S_DIS = "./datarootfiles/bdt_kinematics_4_0_0.root";
        fname_S_QE = "./datarootfiles/bdt_kinematics_4_1_0.root";
        fname_bkg1 = "./datarootfiles/bdt_kinematics_8_0_0.root";
    }
    
    //TFile *input = TFile::Open(fname);
    TFile *input_S_DIS = TFile::Open( fname_S_DIS );
    TFile *input_S_QE = TFile::Open( fname_S_QE );
    TFile *input_bkg1 = TFile::Open( fname_bkg1 );
    TFile *input_bkg2;
    if(channel!=4) input_bkg2 = TFile::Open( fname_bkg2 );
    
    
    
    // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
    // if you use your private .rootrc, or run from a different directory, please copy the
    // corresponding lines from .rootrc
    
    // Methods to be processed can be given as an argument; use format:
    //
    //     mylinux~> root -l TMVAClassification.C\(\"myMethod1,myMethod2,myMethod3\"\)
    
    //---------------------------------------------------------------
    // This loads the library
    TMVA::Tools::Instance();
    
    // Default MVA methods to be trained + tested
    std::map<std::string,int> Use;
    
    
    
    //
    // Boosted Decision Trees
    Use["BDT"]             = 1; // uses Adaptive Boost
    Use["BDTG"]            = 0; // uses Gradient Boost
    Use["BDTB"]            = 0; // uses Bagging
    Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
    Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
    //
    // Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
    Use["RuleFit"]         = 0;
    
    // Neural Networks (all are feed-forward Multilayer Perceptrons)
    Use["MLP"]             = 0; // Recommended ANN
    Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
    Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
    Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
    Use["TMlpANN"]         = 0; // ROOT's own ANN
    Use["DNN"]             = 0;     // Deep Neural Network
    Use["DNN_GPU"]         = 0; // CUDA-accelerated DNN training.
    Use["DNN_CPU"]         = 0; // Multi-core accelerated DNN.
    
    
    // ---------------------------------------------------------------
    
    std::cout << std::endl;
    std::cout << "==> Start TMVAClassification" << std::endl;
    
    // Select methods (don't look at this code - not of interest)
    if (myMethodList != "") {
        for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;
        
        std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
        for (UInt_t i=0; i<mlist.size(); i++) {
            std::string regMethod(mlist[i]);
            
            if (Use.find(regMethod) == Use.end()) {
                std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
                for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
                std::cout << std::endl;
                return 1;
            }
            Use[regMethod] = 1;
        }
    }
    
    // --------------------------------------------------------------------------------------------------
    
    // Here the preparation phase begins
    
    // Read training and test data
    // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
    //   TString fname = "./tmva_class_example.root";
    //
    //   if (gSystem->AccessPathName( fname ))  // file does not exist in local directory
    //      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_class_example.root");
    //
    //   TFile *input = TFile::Open( fname );
    //
    //   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;
    
    //   // Register the training and test trees
    //   TTree *signalTree     = (TTree*)input->Get("TreeS");
    //   TTree *background     = (TTree*)input->Get("TreeB");
    
    // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
    //TString outfileName( "TMVA.root" );
    char outfileName[40];
    sprintf(outfileName, "TMVA_%d.root", channel);
    TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
    // Create the factory object. Later you can choose the methods
    // whose performance you'd like to investigate. The factory is
    // the only TMVA object you have to interact with
    //
    // The first argument is the base of the name of all the
    // weightfiles in the directory weight/
    //
    // The second argument is the output file for the training results
    // All TMVA output can be suppressed by removing the "!" (not) in
    // front of the "Silent" argument in the option string
    TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
    
    TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
    // If you wish to modify default settings
    // (please check "src/Config.h" to see all available global options)
    //
    //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
    //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";
    
    // Define the input variables that shall be used for the MVA training
    // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
    // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
    dataloader->AddVariable( "zdec", "zdec", "#mum", 'F' );
    //dataloader->AddVariable( "decay_length", "decay_length", "#mum", 'F' );
    dataloader->AddVariable( "kink", "kink", "rad", 'F' );
    dataloader->AddVariable( "p2ry", "p2ry", "GeV/c", 'F' );
    if (channel==1) {
        dataloader->AddVariable( "ptmiss", "ptmiss", "GeV/c", 'F' );
        dataloader->AddVariable( "phi", "phi", "rad", 'F' );
        dataloader->AddVariable( "gammadecvtx", "gammadecvtx", "GeV/c", 'I' );
    }
    if (channel==2) {
        dataloader->AddVariable( "charge2ry", "charge", "Charge", 'I' );
    }
    if (channel==3) {
        dataloader->AddVariable( "ptmiss", "ptmiss", "GeV/c", 'F' );
        dataloader->AddVariable( "phi", "phi", "rad", 'F' );
        dataloader->AddVariable( "Minv", "Minv", "GeV/c", 'F' );
        //dataloader->AddVariable( "Minvmin", "Minvmin", "GeV/c", 'F' );
    }
    if (channel!=3) {
        dataloader->AddVariable( "pt2ry", "pt2ry", "GeV/c", 'F' );
    }
    
    
    // You can add so-called "Spectator variables", which are not used in the MVA training,
    // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
    // input variables, the response values of all trained MVAs, and the spectator variables
    
    dataloader->AddSpectator( "enu",  "Nu_Energy", "GeV", 'F' );
    dataloader->AddSpectator( "OscillationP",  "OscillationP", "", 'F' );
    dataloader->AddSpectator( "channel",  "channel", "", 'I' );
    
    //   // global event weights per tree (see below for setting event-wise weights)
    //   Double_t signalWeight     = 1.0;
    //   Double_t backgroundWeight = 1.0;
    //
    //   // You can add an arbitrary number of signal or background trees
    //   dataloader->AddSignalTree    ( signalTree,     signalWeight );
    //   dataloader->AddBackgroundTree( background, backgroundWeight );
    
    // To give different trees for training and testing, do as follows:
    //
    //     dataloader->AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" );
    //     dataloader->AddSignalTree( signalTestTree,     signalTestWeight,  "Test" );
    
    // Use the following code instead of the above two or four lines to add signal and background
    // training and test events "by hand"
    // NOTE that in this case one should not give expressions (such as "var1+var2") in the input
    //      variable definition, but simply compute the expression before adding the event
    // ```cpp
    // // --- begin ----------------------------------------------------------
    // std::vector<Double_t> vars( 4 ); // vector has size of number of input variables
    // Float_t  treevars[4], weight;
    //
    // // Signal
    // for (UInt_t ivar=0; ivar<4; ivar++) signalTree->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
    // for (UInt_t i=0; i<signalTree->GetEntries(); i++) {
    //    signalTree->GetEntry(i);
    //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
    //    // add training and test events; here: first half is training, second is testing
    //    // note that the weight can also be event-wise
    //    if (i < signalTree->GetEntries()/2.0) dataloader->AddSignalTrainingEvent( vars, signalWeight );
    //    else                              dataloader->AddSignalTestEvent    ( vars, signalWeight );
    // }
    //
    // // Background (has event weights)
    // background->SetBranchAddress( "weight", &weight );
    // for (UInt_t ivar=0; ivar<4; ivar++) background->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
    // for (UInt_t i=0; i<background->GetEntries(); i++) {
    //    background->GetEntry(i);
    //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
    //    // add training and test events; here: first half is training, second is testing
    //    // note that the weight can also be event-wise
    //    if (i < background->GetEntries()/2) dataloader->AddBackgroundTrainingEvent( vars, backgroundWeight*weight );
    //    else                                dataloader->AddBackgroundTestEvent    ( vars, backgroundWeight*weight );
    // }
    // // --- end ------------------------------------------------------------
    // ```
    // End of tree registration
    
    
    // --- Register the training and test trees
    // You can add an arbitrary number of signal or background trees
    TTree *signal_DIS     = (TTree*)input_S_DIS->Get("tau_DIS");
    TTree *signal_QE     = (TTree*)input_S_QE->Get("tau_QE");
    
    TTree *background1 = (TTree*)input_bkg1->Get("bkg");
    TTree *background2;
    if(channel!=4) background2 = (TTree*)input_bkg2->Get("bkg2");
    
    //peso le varie componenti
    
    //segnale = DIS + QE
    TH1F *h89_MINBIAS_TFD_S_DIS = (TH1F*)input_S_DIS->Get("h89_MINBIAS_TFD");
    TH1F *h89_MINBIAS_TFD_S_QE = (TH1F*)input_S_QE->Get("h89_MINBIAS_TFD");
    
    Double_t signalWeight_DIS     = tauDISfraction[channel]/h89_MINBIAS_TFD_S_DIS->Integral();;
    Double_t signalWeight_QE     = (1-tauDISfraction[channel])/h89_MINBIAS_TFD_S_QE->Integral();
    
    cout <<"\t\t PESI SIG: " << signalWeight_DIS << "\t" << signalWeight_QE << endl;
    
    //fondo = charm (1) + had reint o LAS (2)
    TH1F *h89_MINBIAS_TFD_bkg1 = (TH1F*)input_bkg1->Get("h89_MINBIAS_TFD");
    TH1F *h89_MINBIAS_TFD_bkg2;
    if(channel!=4) h89_MINBIAS_TFD_bkg2 = (TH1F*)input_bkg2->Get("h89_MINBIAS_TFD");
    
    Double_t backgroundWeight1 = charmfraction[channel]/h89_MINBIAS_TFD_bkg1->Integral();
    Double_t backgroundWeight2;
    if(channel!=4) backgroundWeight2 = (1-charmfraction[channel])/h89_MINBIAS_TFD_bkg2->Integral();
    
    
    if(channel!=4) cout <<"\t\t PESI BKG: " << backgroundWeight1 << "\t" << backgroundWeight2 << endl;
    
    // global event weights per tree (see below for setting event-wise weights)
    dataloader->AddSignalTree    ( signal_DIS,     signalWeight_DIS     );
    dataloader->AddSignalTree    ( signal_QE,     signalWeight_QE     );
    dataloader->AddBackgroundTree( background1, backgroundWeight1 );
    if(channel!=4) dataloader->AddBackgroundTree( background2, backgroundWeight2 );
    
    
    // Set individual event weights (the variables must exist in the original TTree)
    // -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
    // -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`
    dataloader->SetSignalWeightExpression    ("OscillationP");
    dataloader->SetBackgroundWeightExpression("OscillationP");
    
    // Apply additional cuts on the signal and background samples (can be different)
    TCut mycuts = "phi!=-99&&ptmiss!=-99&&p2ry<100";//&&pt2ry>0.2"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1"; //ptcut
    TCut mycutb = "phi!=-99&&ptmiss!=-99&&p2ry<100";//&&pt2ry>0.2"; // for example: TCut mycutb = "abs(var1)<0.5"; //ptcut
    
    if(channel==2) TCut mycuts = "phi!=-99&&ptmiss!=-99&&p2ry<100";//&&pt2ry>0.2"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1"; //ptcut
    if(channel==2)TCut mycutb = "phi!=-99&&ptmiss!=-99&&p2ry<100";//&&pt2ry>0.2"; // for example: TCut mycutb = "abs(var1)<0.5"; //ptcut
    
    // Tell the dataloader how to use the training and testing events
    //
    // If no numbers of events are given, half of the events in the tree are used
    // for training, and the other half for testing:
    //
    //    dataloader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
    //
    // To also specify the number of testing events, use:
    //
    //    dataloader->PrepareTrainingAndTestTree( mycut,
    //         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
    if (channel==1) dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=0:nTrain_Background=5400:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=None:!V" );
    else if (channel==3) dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3000:nTrain_Background=5000:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=None:!V" );
    else dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=None:!V" );
    
    // ### Book MVA methods
    //
    // Please lookup the various method configuration options in the corresponding cxx files, eg:
    // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
    // it is possible to preset ranges in the option string in which the cut optimisation should be done:
    // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable
    
    
    
    // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
    if (Use["MLP"])
        factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
    
    if (Use["MLPBFGS"])
        factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );
    
    if (Use["MLPBNN"])
        factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators
    
    
    
    // CF(Clermont-Ferrand)ANN
    if (Use["CFMlpANN"])
        factory->BookMethod( dataloader, TMVA::Types::kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=2000:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...
    
    // Tmlp(Root)ANN
    if (Use["TMlpANN"])
        factory->BookMethod( dataloader, TMVA::Types::kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ); // n_cycles:#nodes:#nodes:...
    
    
    // Boosted Decision Trees
    if (Use["BDTG"]) // Gradient Boost
        factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG",
                            "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );
    
    if (Use["BDT"])  {// Adaptive Boost
        //      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT",
        //                           "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
        if (channel==1) {//
            //factory->BookMethod( TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=350:MinNodeSize=10%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" ); //NEW
            factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT","!H:!V:NTrees=314:MinNodeSize=10%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=-1" ); //MinNodeSize=2.5% fondo peerfetto 240 //OLD
        }
        else if (channel==2) {//400
            factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
                                "!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");
        }
        else if (channel==3) {//200 0.101 0.009
            factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
                                "!H:!V:NTrees=321:MinNodeSize=5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" ); //321 //324 0.001 0.115
            //OLD                    "!H:!V:NTrees=350:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
        }
        else
            factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
                                "!H:!V:NTrees=138:MinNodeSize=15%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
    }
    
    if (Use["BDTB"]) // Bagging
        factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTB",
                            "!H:!V:NTrees=400:BoostType=Bagging:SeparationType=GiniIndex:nCuts=20" );
    
    if (Use["BDTD"]) // Decorrelation + Adaptive Boost
        factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTD",
                            "!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate" );
    
    if (Use["BDTF"])  // Allow Using Fisher discriminant in node splitting for (strong) linearly correlated variables
        factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTF",
                            "!H:!V:NTrees=50:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20" );
    
    // RuleFit -- TMVA implementation of Friedman's method
    if (Use["RuleFit"])
        factory->BookMethod( dataloader, TMVA::Types::kRuleFit, "RuleFit",
                            "H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );
    
    // For an example of the category classifier usage, see: TMVAClassificationCategory
    //
    // --------------------------------------------------------------------------------------------------
    //  Now you can optimize the setting (configuration) of the MVAs using the set of training events
    // STILL EXPERIMENTAL and only implemented for BDT's !
    //
    //     factory->OptimizeAllMethods("SigEffAt001","Scan");
    //     factory->OptimizeAllMethods("ROCIntegral","FitGA");
    //
    // --------------------------------------------------------------------------------------------------
    
    // Now you can tell the factory to train, test, and evaluate the MVAs
    //
    // Train MVAs using the set of training events
    factory->TrainAllMethods();
    
    // Evaluate all MVAs using the set of test events
    factory->TestAllMethods();
    
    // Evaluate and compare performance of all configured MVAs
    factory->EvaluateAllMethods();
    
    // --------------------------------------------------------------
    
    // Save the output
    outputFile->Close();
    
    std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
    std::cout << "==> TMVAClassification is done!" << std::endl;
    
    
    gStyle->SetOptStat("nemr");
    //gStyle->SetOptStat(0);
    
    
    const int sin2T23=1;
    const float Dm23 = 0.0025;
    const float L = 730;
    const float proton_mass=0.938;
    const Float_t PImass = 139.57018*pow(10,-3); //GeV
    const Float_t TAUmass = 1776.86*pow(10,-3); //GeV
    
    
    TCanvas *c = new TCanvas("c", "BDT Plot", 1000,600);
    TCanvas *c1 = new TCanvas("c1", "Kin Variables", 3000,1400);
    TCanvas *c2 = new TCanvas("c2","Eff&Pur", 900,400);
    c2->Divide(2,1);
    //TCanvas *c3 = new TCanvas("c3","Eff*Pur", 900,600);
    
    //TCanvas *cMLP,*c2MLP,*cMLPBFGS,*c2MLPBFGS,*cTMlpANN,*c2TMlpANN;
    //    if (Use["MLP"]) {
    //        cMLP = new TCanvas("cMLP", "MLP Plot", 1000,600);
    //        c2MLP = new TCanvas("c2MLP","Eff&Pur MLP", 900,400);
    //        c2MLP->Divide(2,1);
    //    }
    //    if (Use["MLPBFGS"]){
    //        cMLPBFGS = new TCanvas("cMLPBFGS", "MLPBFGS Plot", 1000,600);
    //        c2MLPBFGS = new TCanvas("c2MLPBFGS","Eff&Pur MLPBFGS", 900,400);
    //        c2MLPBFGS->Divide(2,1);
    //    }
    //    if (Use["TMlpANN"]) {
    //        cTMlpANN = new TCanvas("cTMlpANN", "TMlpANN Plot", 1000,600);
    //        c2TMlpANN = new TCanvas("c2TMlpANN","Eff&Pur TMlpANN", 900,400);
    //        c2TMlpANN->Divide(2,1);
    //    }
    
    
    TH1F *h_zdec_S, *h_zdec_B;
    if (channel==2) h_zdec_S = new TH1F("h_zdec_S","z_{dec}; #mum; entries",29,-1000,2600);
    else if(channel==4) h_zdec_S = new TH1F("h_zdec_S","z_{dec}; #mum; entries",25,-1000,2600);
    else h_zdec_S = new TH1F("h_zdec_S","z_{dec}; #mum; entries",27,-1000,2600);
    h_zdec_S->SetLineColor(kBlue+1);
    h_zdec_S->SetLineWidth(2);
    h_zdec_S->SetFillColor(kBlue+1);
    h_zdec_S->Sumw2();
    if (channel==2) h_zdec_B = new TH1F("h_zdec_B","z_{dec}; #mum; entries",29,-1000,2600);
    else if (channel==4)h_zdec_B = new TH1F("h_zdec_B","z_{dec}; #mum; entries",25,-1000,2600);
    else h_zdec_B = new TH1F("h_zdec_B","z_{dec}; #mum; entries",27,-1000,2600);
    h_zdec_B->SetLineColor(kRed);
    h_zdec_B->SetFillColor(kRed);
    h_zdec_B->SetFillStyle(3005);
    h_zdec_B->SetLineWidth(2);
    h_zdec_B->Sumw2();
    
    TH1F *h_decay_length_S = new TH1F("h_decay_length_S","decay_length; #mum; entries",30,0,4000);
    h_decay_length_S->SetLineColor(kBlue+1);
    h_decay_length_S->SetLineWidth(2);
    h_decay_length_S->SetFillColor(kBlue+1);
    h_decay_length_S->Sumw2();
    TH1F *h_decay_length_B = new TH1F("h_decay_length_B","decay_length; #mum; entries",30,0,4000);
    h_decay_length_B->SetLineColor(kRed);
    h_decay_length_B->SetFillColor(kRed);
    h_decay_length_B->SetFillStyle(3005);
    h_decay_length_B->SetLineWidth(2);
    h_decay_length_B->Sumw2();
    
    TH1F *h_kink_S, *h_kink_B;
    if (channel==4) h_kink_S = new TH1F("h_kink_S","#theta_{kink}; rad; entries",40,0,1);
    else if (channel==2) h_kink_S = new TH1F("h_kink_S","#theta_{kink}; rad; entries",40,0,0.6);
    else if (channel==1) h_kink_S = new TH1F("h_kink_S","#theta_{kink}; rad; entries",50,0,0.6);
    else h_kink_S = new TH1F("h_kink_S","#theta_{kink}; rad; entries",70,0,0.6);
    h_kink_S->SetLineColor(kBlue+1);
    h_kink_S->SetLineWidth(2);
    h_kink_S->SetFillColor(kBlue+1);
    h_kink_S->Sumw2();
    if (channel==4) h_kink_B = new TH1F("h_kink_B","#theta_{kink}; rad; entries",40,0,1);
    else if (channel==2) h_kink_B = new TH1F("h_kink_B","#theta_{kink}; rad; entries",40,0,0.6);
    else if (channel==1) h_kink_B = new TH1F("h_kink_B","#theta_{kink}; rad; entries",40,0,0.6);
    else h_kink_B = new TH1F("h_kink_B","#theta_{kink}; rad; entries",70,0,0.6);
    h_kink_B->SetLineColor(kRed);
    h_kink_B->SetFillColor(kRed);
    h_kink_B->SetFillStyle(3005);
    h_kink_B->SetLineWidth(2);
    h_kink_B->Sumw2();
    
    TH1F *h_p2ry_S, *h_p2ry_B;
    if (channel==4) h_p2ry_S = new TH1F("h_p2ry_S","p_{2ry}; GeV/c; entries",20,0,20);
    else if(channel==2) h_p2ry_S = new TH1F("h_p2ry_S","p_{2ry}; GeV/c; entries",35,0,20);
    else h_p2ry_S = new TH1F("h_p2ry_S","p_{2ry}; GeV/c; entries",60,0,30);
    h_p2ry_S->SetLineColor(kBlue+1);
    h_p2ry_S->SetLineWidth(2);
    h_p2ry_S->SetFillColor(kBlue+1);
    h_p2ry_S->Sumw2();
    if (channel==4) h_p2ry_B = new TH1F("h_p2ry_B","p_{2ry}; GeV/c; entries",20,0,20);
    else if(channel==2) h_p2ry_B = new TH1F("h_p2ry_B","p_{2ry}; GeV/c; entries",35,0,20);
    else h_p2ry_B = new TH1F("h_p2ry_B","p_{2ry}; GeV/c; entries",60,0,30);
    h_p2ry_B->SetLineColor(kRed);
    h_p2ry_B->SetFillColor(kRed);
    h_p2ry_B->SetFillStyle(3005);
    h_p2ry_B->SetLineWidth(2);
    h_p2ry_B->Sumw2();
    
    TH1F *h_ptmiss_S = new TH1F("h_ptmiss_S","p^{T}_{miss}; GeV/c; entries",25,0,4);
    h_ptmiss_S->SetLineColor(kBlue+1);
    h_ptmiss_S->SetLineWidth(2);
    h_ptmiss_S->SetFillColor(kBlue+1);
    h_ptmiss_S->Sumw2();
    TH1F *h_ptmiss_B = new TH1F("h_ptmiss_B","p^{T}_{miss}; GeV/c; entries",25,0,4);
    h_ptmiss_B->SetLineColor(kRed);
    h_ptmiss_B->SetFillColor(kRed);
    h_ptmiss_B->SetFillStyle(3005);
    h_ptmiss_B->SetLineWidth(2);
    h_ptmiss_B->Sumw2();
    
    TH1F *h_phi_S = new TH1F("h_phi_S","#phi_{lH}; degrees; entries",30,0,180);
    h_phi_S->SetLineColor(kBlue+1);
    h_phi_S->SetLineWidth(2);
    h_phi_S->SetFillColor(kBlue+1);
    h_phi_S->Sumw2();
    TH1F *h_phi_B = new TH1F("h_phi_B","#phi_{lH}; degrees; entries",30,0,180);
    h_phi_B->SetLineColor(kRed);
    h_phi_B->SetFillColor(kRed);
    h_phi_B->SetFillStyle(3005);
    h_phi_B->SetLineWidth(2);
    h_phi_B->Sumw2();
    
    TH1F *h_gammadecvtx_S = new TH1F("h_gammadecvtx_S","#gamma at decay vertex; #gamma; entries",11,-0.5,10.5);
    h_gammadecvtx_S->SetLineColor(kBlue+1);
    h_gammadecvtx_S->SetLineWidth(2);
    h_gammadecvtx_S->SetFillColor(kBlue+1);
    h_gammadecvtx_S->Sumw2();
    TH1F *h_gammadecvtx_B = new TH1F("h_gammadecvtx_B","#gamma at decay vertex; #gamma; entries",11,-0.5,10.5);
    h_gammadecvtx_B->SetLineColor(kRed);
    h_gammadecvtx_B->SetFillColor(kRed);
    h_gammadecvtx_B->SetFillStyle(3005);
    h_gammadecvtx_B->SetLineWidth(2);
    h_gammadecvtx_B->Sumw2();
    
    TH1F *h_Minv_S = new TH1F("h_Minv_S","Invariant Mass; GeV/c; entries",40,0,4);
    h_Minv_S->SetLineColor(kBlue+1);
    h_Minv_S->SetLineWidth(2);
    h_Minv_S->SetFillColor(kBlue+1);
    h_Minv_S->Sumw2();
    TH1F *h_Minv_B = new TH1F("h_Minv_B","Invariant Mass; GeV/c; entries",40,0,4);
    h_Minv_B->SetLineColor(kRed);
    h_Minv_B->SetFillColor(kRed);
    h_Minv_B->SetFillStyle(3005);
    h_Minv_B->SetLineWidth(2);
    h_Minv_B->Sumw2();
    
    TH1F *h_Minvmin_S = new TH1F("h_Minvmin_S","Minimum Invariant mass; GeV/c; entries",40,0,4);
    h_Minvmin_S->SetLineColor(kBlue+1);
    h_Minvmin_S->SetLineWidth(2);
    h_Minvmin_S->SetFillColor(kBlue+1);
    h_Minvmin_S->Sumw2();
    TH1F *h_Minvmin_B = new TH1F("h_Minvmin_B","Minimum Invariant mass; GeV/c; entries",40,0,4);
    h_Minvmin_B->SetLineColor(kRed);
    h_Minvmin_B->SetFillColor(kRed);
    h_Minvmin_B->SetFillStyle(3005);
    h_Minvmin_B->SetLineWidth(2);
    h_Minvmin_B->Sumw2();
    
    TH1F *h_pt2ry_S, *h_pt2ry_B;
    if (channel==2) h_pt2ry_S = new TH1F("h_pt2ry_S","p^{T}_{2ry}; GeV/c; entries",40,0,1.5);
    else h_pt2ry_S = new TH1F("h_pt2ry_S","p^{T}_{2ry}; GeV/c; entries",35,0,3);
    h_pt2ry_S->SetLineColor(kBlue+1);
    h_pt2ry_S->SetLineWidth(2);
    h_pt2ry_S->SetFillColor(kBlue+1);
    h_pt2ry_S->Sumw2();
    if (channel==2) h_pt2ry_B = new TH1F("h_pt2ry_B","p^{T}_{2ry}; GeV/c; entries",40,0,1.5);
    else h_pt2ry_B = new TH1F("h_pt2ry_B","p^{T}_{2ry}; GeV/c; entries",35,0,3);
    h_pt2ry_B->SetLineColor(kRed);
    h_pt2ry_B->SetFillColor(kRed);
    h_pt2ry_B->SetFillStyle(3005);
    h_pt2ry_B->SetLineWidth(2);
    h_pt2ry_B->Sumw2();
    
    TH1F *h_charge_S = new TH1F("h_charge_S","#mu charge; GeV/c; entries",3,-1,2);
    h_charge_S->SetLineColor(kBlue+1);
    h_charge_S->SetLineWidth(2);
    h_charge_S->SetFillColor(kBlue+1);
    h_charge_S->Sumw2();
    TH1F *h_charge_B = new TH1F("h_charge_B","#mu charge; GeV/c; entries",3,-1,2);
    h_charge_B->SetLineColor(kRed);
    h_charge_B->SetFillColor(kRed);
    h_charge_B->SetFillStyle(3005);
    h_charge_B->SetLineWidth(2);
    h_charge_B->Sumw2();
    
    TH1F *h_ch_S = new TH1F("h_ch_S","h_ch_S; ; entries",26,-1,25);
    h_ch_S->SetLineColor(kBlue+1);
    h_ch_S->SetLineWidth(2);
    h_ch_S->SetFillColor(kBlue+1);
    h_ch_S->Sumw2();
    TH1F *h_ch_B = new TH1F("h_ch_B","h_ch_B; ; entries",26,-1,25);
    h_ch_B->SetLineColor(kRed);
    h_ch_B->SetFillColor(kRed);
    h_ch_B->SetFillStyle(3005);
    h_ch_B->SetLineWidth(2);
    h_ch_B->Sumw2();
    
    
    TH1F *h_bdt_S, *h_bdt_B;
    if (channel==2||channel==4) h_bdt_S = new TH1F("h_bdt_S","BDT; BDT response; entries",60,-0.8,0.9);
    else h_bdt_S = new TH1F("h_bdt_S","BDT; BDT response; entries",65,-0.8,0.8);
    h_bdt_S->SetLineColor(kBlue+1);
    h_bdt_S->SetLineWidth(2);
    h_bdt_S->SetFillColor(kBlue+1);
    h_bdt_S->Sumw2();
    if (channel==2||channel==4) h_bdt_B = new TH1F("h_bdt_B","BDT; BDT response; entries",60,-0.8,0.9);
    else h_bdt_B = new TH1F("h_bdt_B","BDT; BDT response; entries",65,-0.8,0.8);
    h_bdt_B->SetLineColor(kRed);
    h_bdt_B->SetFillColor(kRed);
    h_bdt_B->SetFillStyle(3005);
    h_bdt_B->SetLineWidth(2);
    h_bdt_B->Sumw2();
    
    
    
    //SCOMMENTARE DA QUI SE SI USANO MLP,MLPBFGS,TMlpANN PER FARE I PLOT
    //    if (Use["MLP"]){
    //        TH1F *h_MLP_S = new TH1F("h_MLP_S","MLP; MLP response; entries",100,-5,5);
    //        h_MLP_S->SetLineColor(kBlue+1);
    //        h_MLP_S->SetLineWidth(2);
    //        h_MLP_S->SetFillColor(kBlue+1);
    //        h_MLP_S->Sumw2();
    //        TH1F *h_MLP_B = new TH1F("h_MLP_B","MLP; MLP response; entries",100,-5,5);
    //        h_MLP_B->SetLineColor(kRed);
    //        h_MLP_B->SetFillColor(kRed);
    //        h_MLP_B->SetFillStyle(3005);
    //        h_MLP_B->SetLineWidth(2);
    //        h_MLP_B->Sumw2();
    //    }
    //
    //    if (Use["MLPBFGS"]){
    //        if(channel==4)
    //            TH1F *h_MLPBFGS_S = new TH1F("h_MLPBFGS_S","MLPBFGS; MLPBFGS response; entries",100,-1,3);
    //        else
    //            TH1F *h_MLPBFGS_S = new TH1F("h_MLPBFGS_S","MLPBFGS; MLPBFGS response; entries",100,-1,3);
    //        h_MLPBFGS_S->SetLineColor(kBlue+1);
    //        h_MLPBFGS_S->SetLineWidth(2);
    //        h_MLPBFGS_S->SetFillColor(kBlue+1);
    //        h_MLPBFGS_S->Sumw2();
    //        if(channel==4)
    //            TH1F *h_MLPBFGS_B = new TH1F("h_MLPBFGS_B","MLPBFGS; MLPBFGS response; entries",100,-1,3);
    //        else
    //            TH1F *h_MLPBFGS_B = new TH1F("h_MLPBFGS_B","MLPBFGS; MLPBFGS response; entries",100,-1,3);
    //        h_MLPBFGS_B->SetLineColor(kRed);
    //        h_MLPBFGS_B->SetFillColor(kRed);
    //        h_MLPBFGS_B->SetFillStyle(3005);
    //        h_MLPBFGS_B->SetLineWidth(2);
    //        h_MLPBFGS_B->Sumw2();
    //    }
    //
    //    if (Use["TMlpANN"]){
    //        TH1F *h_TMlpANN_S = new TH1F("h_TMlpANN_S","TMlpANN; TMlpANN response; entries",100,-1,2);
    //        h_TMlpANN_S->SetLineColor(kBlue+1);
    //        h_TMlpANN_S->SetLineWidth(2);
    //        h_TMlpANN_S->SetFillColor(kBlue+1);
    //        h_TMlpANN_S->Sumw2();
    //        TH1F *h_TMlpANN_B = new TH1F("h_TMlpANN_B","TMlpANN; TMlpANN response; entries",100,-1,2);
    //        h_TMlpANN_B->SetLineColor(kRed);
    //        h_TMlpANN_B->SetFillColor(kRed);
    //        h_TMlpANN_B->SetFillStyle(3005);
    //        h_TMlpANN_B->SetLineWidth(2);
    //        h_TMlpANN_B->Sumw2();
    //    }
    
    //FIN QUI
    
    
    TMVA::Reader reader;
    Float_t kink, p2ry, pt2ry, zdec, Nu_energy, OscillationP, charge, ptmiss, phi, Minv, Minvmin, yBjorken, nchargedvis1ry, decay_length, psum, gammadecvtx;
    int charge_int, gammadecvtx_int;
    float ch=0;
    
    reader.AddVariable("zdec", &zdec);
    //reader.AddVariable("decay_length", &decay_length);
    reader.AddVariable("kink", &kink);
    reader.AddVariable("p2ry", &p2ry);
    if (channel==1) {
        reader.AddVariable("ptmiss", &ptmiss);
        reader.AddVariable("phi", &phi);
        reader.AddVariable("gammadecvtx", &gammadecvtx);
    }
    if (channel==2) {
        reader.AddVariable("charge2ry", &charge);
    }
    if (channel==3) {
        reader.AddVariable("ptmiss", &ptmiss);
        reader.AddVariable("phi", &phi);
        reader.AddVariable("Minv", &Minv);
        //reader.AddVariable("Minvmin", &Minvmin);
    }
    if (channel!=3) {
        reader.AddVariable("pt2ry", &pt2ry);
    }
    reader.AddSpectator("enu", &Nu_energy);
    reader.AddSpectator("OscillationP", &OscillationP);
    reader.AddSpectator("channel", &ch);
    
    //reader.BookMVA("Likelihood", "weights/TMVAClassification_Likelihood.weights.xml");
//    reader.BookMVA("BDT", "dataset/weights/TMVAClassification_BDT.weights.xml");
//    if (Use["MLP"]) reader.BookMVA("MLP", "dataset/weights/TMVAClassification_MLP.weights.xml");
//    if (Use["MLPBFGS"]) reader.BookMVA("MLPBFGS", "dataset/weights/TMVAClassification_MLPBFGS.weights.xml");
//    if (Use["TMlpANN"]) reader.BookMVA("TMlpANN", "dataset/weights/TMVAClassification_TMlpANN.weights.xml");
    
    TString dir    = "dataset/weights/";
    TString prefix = "TMVAClassification";
    // Book method(s)
    for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
        if (it->second) {
            TString methodName = TString(it->first) + TString(" method");
            TString weightfile = dir + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
            reader.BookMVA( methodName, weightfile );
        }
    }
    double bdteval;
    double MLPeval,MLPBFGSeval,TMlpANNeval;
    
    signal_DIS->SetBranchAddress("zdec", &zdec);
    signal_QE->SetBranchAddress("zdec", &zdec);
    
    //signal_DIS->SetBranchAddress("decay_length", &decay_length);
    //signal_QE->SetBranchAddress("decay_length", &decay_length);
    
    signal_DIS->SetBranchAddress("kink", &kink);
    signal_QE->SetBranchAddress("kink", &kink);
    
    signal_DIS->SetBranchAddress("p2ry", &p2ry);
    signal_QE->SetBranchAddress("p2ry", &p2ry);
    
    if (channel==1) {
        signal_DIS->SetBranchAddress("ptmiss", &ptmiss);
        signal_QE->SetBranchAddress("ptmiss", &ptmiss);
        
        signal_DIS->SetBranchAddress("phi", &phi);
        signal_QE->SetBranchAddress("phi", &phi);
        
        signal_DIS->SetBranchAddress("gammadecvtx", &gammadecvtx_int);
        signal_QE->SetBranchAddress("gammadecvtx", &gammadecvtx_int);
        
    }
    if (channel==2) {
        signal_DIS->SetBranchAddress("charge2ry", &charge_int);
        signal_QE->SetBranchAddress("charge2ry", &charge_int);
        
    }
    if (channel==3) {
        signal_DIS->SetBranchAddress("ptmiss", &ptmiss);
        signal_QE->SetBranchAddress("ptmiss", &ptmiss);
        
        signal_DIS->SetBranchAddress("phi", &phi);
        signal_QE->SetBranchAddress("phi", &phi);
        
        signal_DIS->SetBranchAddress("Minv", &Minv);
        signal_QE->SetBranchAddress("Minv", &Minv);
        
        signal_DIS->SetBranchAddress("Minvmin", &Minvmin);
        signal_QE->SetBranchAddress("Minvmin", &Minvmin);
    }
    if (channel!=3) {
        signal_DIS->SetBranchAddress("pt2ry", &pt2ry);
        signal_QE->SetBranchAddress("pt2ry", &pt2ry);
        
    }
    signal_DIS->SetBranchAddress("enu", &Nu_energy);
    signal_QE->SetBranchAddress("enu", &Nu_energy);
    
    signal_DIS->SetBranchAddress("OscillationP", &OscillationP);
    signal_QE->SetBranchAddress("OscillationP", &OscillationP);
    
    
    
    for (Long64_t j=0; j<signal_DIS->GetEntries(); j++) {
        signal_DIS->GetEntry(j);
        charge=charge_int;
        gammadecvtx=gammadecvtx_int;
        //cout << "S\t" << kink << "\t" << p2ry << "\t" << pt2ry << "\t" << zdec << "\t" << charge << "\t" << Nu_energy << endl;
        //Float_t OscillationP = sin2T23*TMath::Sin(1.27*Dm23*Dm23*L/(Nu_energy))*TMath::Sin(1.27*Dm23*Dm23*L/(Nu_energy));
        
        OscillationP*=signalWeight_DIS;
        bdteval  = reader.EvaluateMVA("BDT method");
//        if (Use["MLP"])  MLPeval  = reader.EvaluateMVA("MLP");
//        if (Use["MLPBFGS"])  MLPBFGSeval  = reader.EvaluateMVA("MLPBFGS");
//        if (Use["TMlpANN"])  TMlpANNeval  = reader.EvaluateMVA("TMlpANN");
        
        //cout << "S\t" << bdteval << endl;
        h_zdec_S->Fill(zdec,OscillationP);
        h_decay_length_S->Fill(decay_length,OscillationP);
        h_kink_S->Fill(kink,OscillationP);
        h_p2ry_S->Fill(p2ry,OscillationP);
        if (channel==1) {
            h_ptmiss_S->Fill(ptmiss,OscillationP);
            h_phi_S->Fill(phi,OscillationP);
            h_gammadecvtx_S->Fill(gammadecvtx,OscillationP);
        }
        if (channel==2) {
            h_charge_S->Fill(charge_int,OscillationP);
        }
        if (channel==3) {
            h_ptmiss_S->Fill(ptmiss,OscillationP);
            h_phi_S->Fill(phi,OscillationP);
            h_Minv_S->Fill(Minv,OscillationP);
            h_Minvmin_S->Fill(Minvmin,OscillationP);
        }
        if (channel!=3) {
            //if (pt2ry>0.2) { //ptcut
            h_pt2ry_S->Fill(pt2ry,OscillationP);
            //}
        }
        
        h_bdt_S->Fill(bdteval,OscillationP);
        //        if (Use["MLP"])  h_MLP_S->Fill(MLPeval,OscillationP);
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Fill(MLPBFGSeval,OscillationP);
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Fill(TMlpANNeval,OscillationP);
        
    }
    
    
    for (Long64_t j=0; j<signal_QE->GetEntries(); j++) {
        signal_QE->GetEntry(j);
        charge=charge_int;
        gammadecvtx=gammadecvtx_int;
        OscillationP*=signalWeight_QE;
        //cout << "S\t" << kink << "\t" << p2ry << "\t" << pt2ry << "\t" << zdec << "\t" << charge << "\t" << Nu_energy << endl;
        //Float_t OscillationP = sin2T23*TMath::Sin(1.27*Dm23*Dm23*L/(Nu_energy))*TMath::Sin(1.27*Dm23*Dm23*L/(Nu_energy));
        bdteval  = reader.EvaluateMVA("BDT method");
//        if (Use["MLP"])  MLPeval  = reader.EvaluateMVA("MLP");
//        if (Use["MLPBFGS"])  MLPBFGSeval  = reader.EvaluateMVA("MLPBFGS");
//        if (Use["TMlpANN"])  TMlpANNeval  = reader.EvaluateMVA("TMlpANN");
        
        //cout << "S\t" << bdteval << endl;
        h_zdec_S->Fill(zdec,OscillationP);
        h_decay_length_S->Fill(decay_length,OscillationP);
        h_kink_S->Fill(kink,OscillationP);
        h_p2ry_S->Fill(p2ry,OscillationP);
        if (channel==1) {
            h_ptmiss_S->Fill(ptmiss,OscillationP);
            h_phi_S->Fill(phi,OscillationP);
            h_gammadecvtx_S->Fill(gammadecvtx,OscillationP);
        }
        if (channel==2) {
            h_charge_S->Fill(charge_int,OscillationP);
        }
        if (channel==3) {
            h_ptmiss_S->Fill(ptmiss,OscillationP);
            h_phi_S->Fill(phi,OscillationP);
            h_Minv_S->Fill(Minv,OscillationP);
            h_Minvmin_S->Fill(Minvmin,OscillationP);
        }
        if (channel!=3) {
            //if (pt2ry>0.2) { //ptcut
            h_pt2ry_S->Fill(pt2ry,OscillationP);
            //}
        }
        
        h_bdt_S->Fill(bdteval,OscillationP);
        //        if (Use["MLP"])  h_MLP_S->Fill(MLPeval,OscillationP);
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Fill(MLPBFGSeval,OscillationP);
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Fill(TMlpANNeval,OscillationP);
    }
    
    
    background1->SetBranchAddress("zdec", &zdec);
    if (channel!=4) background2->SetBranchAddress("zdec", &zdec);
    
    //background1->SetBranchAddress("decay_length", &decay_length);
    //background2->SetBranchAddress("decay_length", &decay_length);
    
    background1->SetBranchAddress("p2ry", &p2ry);
    if (channel!=4) background2->SetBranchAddress("p2ry", &p2ry);
    
    background1->SetBranchAddress("kink", &kink);
    if (channel!=4) background2->SetBranchAddress("kink", &kink);
    
    if (channel==1) {
        background1->SetBranchAddress("ptmiss", &ptmiss);
        background2->SetBranchAddress("ptmiss", &ptmiss);
        
        background1->SetBranchAddress("phi", &phi);
        background2->SetBranchAddress("phi", &phi);
        
        background1->SetBranchAddress("gammadecvtx", &gammadecvtx_int);
        background2->SetBranchAddress("gammadecvtx", &gammadecvtx_int);
    }
    if (channel==2) {
        background1->SetBranchAddress("charge2ry", &charge_int);
        background2->SetBranchAddress("charge2ry", &charge_int);
    }
    if (channel==3) {
        background1->SetBranchAddress("ptmiss", &ptmiss);
        background2->SetBranchAddress("ptmiss", &ptmiss);
        
        background1->SetBranchAddress("phi", &phi);
        background2->SetBranchAddress("phi", &phi);
        
        background1->SetBranchAddress("Minv", &Minv);
        background2->SetBranchAddress("Minv", &Minv);
        
        background1->SetBranchAddress("Minvmin", &Minvmin);
        background2->SetBranchAddress("Minvmin", &Minvmin);
    }
    if (channel!=3) {
        background1->SetBranchAddress("pt2ry", &pt2ry);
        if (channel!=4) background2->SetBranchAddress("pt2ry", &pt2ry);
    }
    background1->SetBranchAddress("enu", &Nu_energy);
    if (channel!=4) background2->SetBranchAddress("enu", &Nu_energy);
    
    background1->SetBranchAddress("OscillationP", &OscillationP);
    if (channel!=4) background2->SetBranchAddress("OscillationP", &OscillationP);
    
    background1->SetBranchAddress("channel", &ch);
    if (channel!=4) background2->SetBranchAddress("channel", &ch);
    
    
    for (Long64_t j=0; j<background1->GetEntries(); j++) {
        background1->GetEntry(j);
        charge=charge_int;
        gammadecvtx=gammadecvtx_int;
        OscillationP*=backgroundWeight1;
        
        //cout << "B\t"<< kink << "\t" << p2ry << "\t" << pt2ry << "\t" << zdec << "\t" << charge << "\t" << Nu_energy << endl;
        
        bdteval  = reader.EvaluateMVA("BDT method");
//        if (Use["MLP"])  MLPeval  = reader.EvaluateMVA("MLP");
//        if (Use["MLPBFGS"])  MLPBFGSeval  = reader.EvaluateMVA("MLPBFGS");
//        if (Use["TMlpANN"])  TMlpANNeval  = reader.EvaluateMVA("TMlpANN");
        
        //cout << "B\t" << bdteval << endl;
        h_zdec_B->Fill(zdec,OscillationP);
        h_decay_length_B->Fill(decay_length,OscillationP);
        h_kink_B->Fill(kink,OscillationP);
        h_p2ry_B->Fill(p2ry,OscillationP);
        if (channel==1) {
            h_ptmiss_B->Fill(ptmiss,OscillationP);
            h_phi_B->Fill(phi,OscillationP);
            h_gammadecvtx_B->Fill(gammadecvtx,OscillationP);
        }
        if (channel==2) {
            h_charge_B->Fill(charge_int,OscillationP);
            h_ch_B->Fill(ch, OscillationP);
        }
        if (channel==3) {
            h_ptmiss_B->Fill(ptmiss,OscillationP);
            h_phi_B->Fill(phi,OscillationP);
            h_Minv_B->Fill(Minv,OscillationP);
            h_Minvmin_B->Fill(Minvmin,OscillationP);
        }
        if (channel!=3) {
            //if (pt2ry>0.2) { //ptcut
            h_pt2ry_B->Fill(pt2ry,OscillationP);
            //}
        }
        
        h_bdt_B->Fill(bdteval,OscillationP);
        //        if (Use["MLP"])  h_MLP_B->Fill(MLPeval,OscillationP);
        //        if (Use["MLPBFGS"])  h_MLPBFGS_B->Fill(MLPBFGSeval,OscillationP);
        //        if (Use["TMlpANN"])  h_TMlpANN_B->Fill(TMlpANNeval,OscillationP);
    }
    
    if (channel!=4) {
        for (Long64_t j=0; j<background2->GetEntries(); j++) {
            background2->GetEntry(j);
            charge=charge_int;
            gammadecvtx=gammadecvtx_int;
            OscillationP*=backgroundWeight2;
            
            //cout << "B\t"<< kink << "\t" << p2ry << "\t" << pt2ry << "\t" << zdec << "\t" << charge << "\t" << Nu_energy << endl;
            
            bdteval  = reader.EvaluateMVA("BDT method");
//            if (Use["MLP"])  MLPeval  = reader.EvaluateMVA("MLP");
//            if (Use["MLPBFGS"])  MLPBFGSeval  = reader.EvaluateMVA("MLPBFGS");
//            if (Use["TMlpANN"])  TMlpANNeval  = reader.EvaluateMVA("TMlpANN");
            
            //cout << "B\t" << bdteval << endl;
            h_zdec_B->Fill(zdec,OscillationP);
            h_decay_length_B->Fill(decay_length,OscillationP);
            h_kink_B->Fill(kink,OscillationP);
            h_p2ry_B->Fill(p2ry,OscillationP);
            if (channel==1) {
                h_ptmiss_B->Fill(ptmiss,OscillationP);
                h_phi_B->Fill(phi,OscillationP);
                h_gammadecvtx_B->Fill(gammadecvtx,OscillationP);
            }
            if (channel==2) {
                h_charge_B->Fill(charge_int,OscillationP);
                h_ch_B->Fill(ch, OscillationP);
            }
            if (channel==3) {
                h_ptmiss_B->Fill(ptmiss,OscillationP);
                h_phi_B->Fill(phi,OscillationP);
                h_Minv_B->Fill(Minv,OscillationP);
                h_Minvmin_B->Fill(Minvmin,OscillationP);
            }
            if (channel!=3) {
                //if (pt2ry>0.2) { //ptcut
                h_pt2ry_B->Fill(pt2ry,OscillationP);
                //}
            }
            
            h_bdt_B->Fill(bdteval,OscillationP);
            //            if (Use["MLP"])  h_MLP_B->Fill(MLPeval,OscillationP);
            //            if (Use["MLPBFGS"])  h_MLPBFGS_B->Fill(MLPBFGSeval,OscillationP);
            //            if (Use["TMlpANN"])  h_TMlpANN_B->Fill(TMlpANNeval,OscillationP);
            
        }
        
    }
    
    if (channel==1) {
        c1->Divide(4,2);
    }
    if (channel==2) {
        c1->Divide(3,2);
    }
    if (channel==3) {
        c1->Divide(3,2);
    }
    if (channel==4) {
        c1->Divide(2,2);
    }
    
    
    c1->cd(1);
    //    h_decay_length_S->Scale(h_decay_length_B->Integral()/h_decay_length_S->Integral());
    //    h_decay_length_S->Draw("");
    //    h_decay_length_B->Draw("same");
    
    h_zdec_S->Scale(1/h_zdec_S->Integral());
    h_zdec_B->Scale(1/h_zdec_B->Integral());
    if(channel==4) h_zdec_B->Draw("HISTOsames");
    h_zdec_S->Draw("HISTOsames");
    h_zdec_B->Draw("HISTOsames");
    h_zdec_S->GetYaxis()->SetTitleOffset(1.5);
    h_zdec_B->GetYaxis()->SetTitleOffset(1.5);
    c1->Update();
    
    c1->cd(2);
    h_kink_S->Scale(1/h_kink_S->Integral());
    h_kink_B->Scale(1/h_kink_B->Integral());
    if(channel!=3) h_kink_B->Draw("HISTO");
    h_kink_S->Draw("HISTOsames");
    h_kink_B->Draw("HISTOsames");
    h_kink_S->GetYaxis()->SetTitleOffset(1.5);
    h_kink_B->GetYaxis()->SetTitleOffset(1.5);
    c1->Update();
    
    TLegend *legend_k_ = new TLegend(.75,.70,.99,.95);
    if (channel==1) {
        legend_k_->AddEntry(h_kink_S,"signal: #tau #rightarrow 1h");
        legend_k_->AddEntry(h_kink_B,"bkg: charm #rightarrow 1h and 1-prong Had.reint.");
    }
    else if (channel==2) {
        legend_k_->AddEntry(h_kink_S,"signal: #tau #rightarrow #mu");
        legend_k_->AddEntry(h_kink_B,"bkg: charm #rightarrow #mu and LAS");
    }
    else if (channel==3) {
        legend_k_->AddEntry(h_kink_S,"signal: #tau #rightarrow 3h");
        legend_k_->AddEntry(h_kink_B,"bkg: charm #rightarrow 3h and 3-prong Had.reint.");
    }
    else if (channel==4) {
        legend_k_->AddEntry(h_kink_S,"signal: #tau #rightarrow e");
        legend_k_->AddEntry(h_kink_B,"bkg: charm #rightarrow e");
    }
    //legend_k->Draw("same");
    
    c1->cd(3);
    h_p2ry_S->Scale(1/h_p2ry_S->Integral());
    h_p2ry_B->Scale(1/h_p2ry_B->Integral());
    //h_p2ry_S->SetMaximum(550);
    h_p2ry_B->Draw("HISTOsames");
    h_p2ry_S->Draw("HISTOsames");
    h_p2ry_B->Draw("HISTOsames");
    h_p2ry_S->GetYaxis()->SetTitleOffset(1.5);
    h_p2ry_B->GetYaxis()->SetTitleOffset(1.5);
    c1->Update();
    
    
    if (channel==1) {
        c1->cd(4);
        h_ptmiss_S->Scale(1/h_ptmiss_S->Integral());
        h_ptmiss_B->Scale(1/h_ptmiss_B->Integral());
        //h_ptmiss_B->Draw("HISTO");
        h_ptmiss_S->Draw("HISTO");
        h_ptmiss_B->Draw("HISTOsames");
        h_ptmiss_S->GetYaxis()->SetTitleOffset(1.5);
        h_ptmiss_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        c1->cd(5);
        h_phi_S->Scale(1/h_phi_S->Integral());
        h_phi_B->Scale(1/h_phi_B->Integral());
        h_phi_S->Draw("HISTO");
        h_phi_B->Draw("HISTOsames");
        h_phi_S->GetYaxis()->SetTitleOffset(1.5);
        h_phi_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        c1->cd(6);
        h_gammadecvtx_S->Scale(1/h_gammadecvtx_S->Integral());
        h_gammadecvtx_B->Scale(1/h_gammadecvtx_B->Integral());
        //h_gammadecvtx_S->Draw("HISTO");
        h_gammadecvtx_B->Draw("HISTO");
        h_gammadecvtx_S->Draw("HISTOsames");
        h_gammadecvtx_B->Draw("HISTOsames");
        h_gammadecvtx_S->GetYaxis()->SetTitleOffset(1.5);
        h_gammadecvtx_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
    }
    
    if (channel==2) {
        c1->cd(5);
        h_charge_S->Scale(1/h_charge_S->Integral());
        h_charge_B->Scale(1/h_charge_B->Integral());
        h_charge_S->Draw("HISTO");
        h_charge_B->Draw("HISTOsames");
        h_charge_S->GetYaxis()->SetTitleOffset(1.5);
        h_charge_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        //        c1->cd(6);
        //        h_ch_B->Scale(1/h_ch_B->Integral());
        //        h_ch_B->Draw("HISTO");
        //        c1->Update();
    }
    
    if (channel==3) {
        c1->cd(4);
        h_ptmiss_S->Scale(1/h_ptmiss_S->Integral());
        h_ptmiss_B->Scale(1/h_ptmiss_B->Integral());
        //h_ptmiss_B->Draw("HISTO");
        h_ptmiss_S->Draw("HISTO");
        h_ptmiss_B->Draw("HISTOsames");
        h_ptmiss_S->GetYaxis()->SetTitleOffset(1.5);
        h_ptmiss_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        
        c1->cd(5);
        h_phi_S->Scale(1/h_phi_S->Integral());
        h_phi_B->Scale(1/h_phi_B->Integral());
        h_phi_S->Draw("HISTO");
        h_phi_B->Draw("HISTOsames");
        h_phi_S->GetYaxis()->SetTitleOffset(1.5);
        h_phi_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        c1->cd(6);
        h_Minv_S->Scale(1/h_Minv_S->Integral());
        h_Minv_B->Scale(1/h_Minv_B->Integral());
        h_Minv_S->Draw("HISTO");
        h_Minv_B->Draw("HISTOsames");
        h_Minv_S->GetYaxis()->SetTitleOffset(1.5);
        h_Minv_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
        
        //        c1->cd(7);
        //        h_Minvmin_S->Scale(1/h_Minvmin_S->Integral());
        //        h_Minvmin_B->Scale(1/h_Minvmin_B->Integral());
        //        h_Minvmin_S->Draw("HISTO");
        //        h_Minvmin_B->Draw("HISTOsames");
        //        c1->Update();
        
    }
    
    if (channel!=3) {
        if (channel==1) {
            c1->cd(7);
        }
        if (channel==2||channel==4) {
            c1->cd(4);
        }
        
        h_pt2ry_S->Scale(1/h_pt2ry_S->Integral());
        h_pt2ry_B->Scale(1/h_pt2ry_B->Integral());
        //h_pt2ry_S->SetMaximum(600);
        h_pt2ry_B->Draw("HISTO");
        h_pt2ry_S->Draw("HISTOsames");
        h_pt2ry_B->Draw("HISTOsames");
        h_pt2ry_S->GetYaxis()->SetTitleOffset(1.5);
        h_pt2ry_B->GetYaxis()->SetTitleOffset(1.5);
        c1->Update();
    }
    
    //TAU EVENTS
    TLine *zdec1, *kink1, *p2ry1, *ptmiss1, *phi1, *gammadecvtx1, *pt2ry1;
    TLine *zdec2, *kink2, *p2ry2, *ptmiss2, *phi2, *gammadecvtx2, *pt2ry2, *Minv2;
    TLine *zdec3, *kink3, *p2ry3, *ptmiss3, *phi3, *gammadecvtx3, *pt2ry3, *charge3;
    TLine *zdec4, *kink4, *p2ry4, *ptmiss4, *phi4, *gammadecvtx4, *pt2ry4;
    TLine *zdec5, *kink5, *p2ry5, *ptmiss5, *phi5, *gammadecvtx5, *pt2ry5;
    TLine *zdecBER, *kinkBER, *p2ryBER, *ptmissBER, *phiBER, *gammadecvtxBER, *pt2ryBER;
    TLine *zdecPDBO, *kinkPDBO, *p2ryPDBO, *ptmissPDBO, *phiPDBO, *gammadecvtxPDBO, *pt2ryPDBO;
    TLine *zdecNAG2, *kinkNAG2, *p2ryNAG2, *ptmissNAG2, *phiNAG2, *gammadecvtxNAG2, *pt2ryNAG2;
    TLine *zdecBARI, *kinkBARI, *p2ryBARI, *ptmissBARI, *phiBARI, *gammadecvtxBARI, *pt2ryBARI, *MinvBARI;
    TLine *zdecNAG4, *kinkNAG4, *p2ryNAG4, *ptmissNAG4, *phiNAG4, *gammadecvtxNAG4, *pt2ryNAG4, *MinvNAG4;
    
    double bdtev1, bdtev2, bdtev3, bdtev4, bdtev5, bdtevBER, bdtevBARI, bdtevPDBO, bdtevNAG2, bdtevNAG4;
    
    if (channel==1) {
        //ev1 9234119599 //1st tau
        //0mu
        kink = 0.041; //+-0.002 rad
        decay_length = 1335;//+-35 micron
        zdec =  435; //+- 35 micron
        p2ry = 12; //+6.2 -3 GeV/c
        psum = 24.3; // (-2.7 +3.9) GeV/c
        phi = 172.55; //+2 -31
        ptmiss = 0.57; // +0.32 -0.17
        pt2ry = 0.47; // +0.24 -0.12
        nchargedvis1ry = 7;
        gammadecvtx=2;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtev1  = reader.EvaluateMVA("BDT method");
//        if (Use["MLP"])  MLPev1  = reader.EvaluateMVA("MLP");
//        if (Use["MLPBFGS"])  MLPBFGSev1  = reader.EvaluateMVA("MLPBFGS");
//        if (Use["TMlpANN"])  TMlpANNev1  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdec1 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdec1->SetLineWidth(2);
        if (goldensilver==1) zdec1->SetLineColor(kYellow+1);
        else zdec1->SetLineColor(kGreen+2);
        zdec1->Draw();
        c1->Update();
        c1->cd(2);
        kink1 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kink1->SetLineWidth(2);
        if (goldensilver==1) kink1->SetLineColor(kYellow+1);
        else kink1->SetLineColor(kGreen+2);
        kink1->Draw();
        c1->Update();
        c1->cd(3);
        p2ry1 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ry1->SetLineWidth(2);
        if (goldensilver==1) p2ry1->SetLineColor(kYellow+1);
        else p2ry1->SetLineColor(kGreen+2);
        p2ry1->Draw();
        c1->Update();
        c1->cd(4);
        ptmiss1 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmiss1->SetLineWidth(2);
        if (goldensilver==1) ptmiss1->SetLineColor(kYellow+1);
        else ptmiss1->SetLineColor(kGreen+2);
        ptmiss1->Draw();
        c1->Update();
        c1->cd(5);
        phi1 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phi1->SetLineWidth(2);
        if (goldensilver==1) phi1->SetLineColor(kYellow+1);
        else phi1->SetLineColor(kGreen+2);
        phi1->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtx1 = new TLine(gammadecvtx,c1->cd(6)->GetUymin(),gammadecvtx,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtx1->SetLineWidth(2);
        if (goldensilver==1) gammadecvtx1->SetLineColor(kYellow+1);
        else gammadecvtx1->SetLineColor(kGreen+2);
        gammadecvtx1->Draw();
        c1->Update();
        c1->cd(7);
        pt2ry1 = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ry1->SetLineWidth(2);
        if (goldensilver==1) pt2ry1->SetLineColor(kYellow+1);
        else pt2ry1->SetLineColor(kGreen+2);
        pt2ry1->Draw();
        c1->Update();
        
        
        //ev4
        //0mu
        nchargedvis1ry = 4;
        kink = 0.137; //+-0.004 rad
        zdec = 406; // +-30 micron
        decay_length = 1090;//+-30 micron
        p2ry = 6.0;// +2.2 - 1.2 GeV/c
        psum = 14.4; //  (-2.7 +3.9) GeV/c
        phi = 166; // +2 -31
        ptmiss = 0.55; // +0.30 -0.20
        pt2ry = 0.82; // +0.3 -0.16
        gammadecvtx=0;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtev4 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPev4  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSev4  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNev4  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdec4 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdec4->SetLineWidth(2);
        if (goldensilver==1) zdec4->SetLineColor(kYellow+1);
        else zdec4->SetLineColor(kGreen+3);
        zdec4->Draw();
        c1->Update();
        c1->cd(2);
        kink4 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kink4->SetLineWidth(2);
        if (goldensilver==1) kink4->SetLineColor(kYellow+1);
        else kink4->SetLineColor(kGreen+3);
        kink4->Draw();
        c1->Update();
        c1->cd(3);
        p2ry4 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ry4->SetLineWidth(2);
        if (goldensilver==1) p2ry4->SetLineColor(kYellow+1);
        else p2ry4->SetLineColor(kGreen+3);
        p2ry4->Draw();
        c1->Update();
        c1->cd(4);
        ptmiss4 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmiss4->SetLineWidth(2);
        if (goldensilver==1) ptmiss4->SetLineColor(kYellow+1);
        else ptmiss4->SetLineColor(kGreen+3);
        ptmiss4->Draw();
        c1->Update();
        c1->cd(5);
        phi4 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phi4->SetLineWidth(2);
        if (goldensilver==1) phi4->SetLineColor(kYellow+1);
        else phi4->SetLineColor(kGreen+3);
        phi4->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtx4 = new TLine(gammadecvtx,c1->cd(6)->GetUymin(),gammadecvtx,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtx4->SetLineWidth(2);
        if (goldensilver==1) gammadecvtx4->SetLineColor(kYellow+1);
        else gammadecvtx4->SetLineColor(kGreen+3);
        gammadecvtx4->Draw();
        c1->Update();
        c1->cd(7);
        pt2ry4 = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ry4->SetLineWidth(2);
        if (goldensilver==1) pt2ry4->SetLineColor(kYellow+1);
        else pt2ry4->SetLineColor(kGreen+3);
        pt2ry4->Draw();
        c1->Update();
        
        
        //ev 5
        //0mu
        nchargedvis1ry = 1;
        decay_length = 960; //+-30 micron
        kink = 0.090; //+-0.002 rad
        //IP: 83+-5 micron
        zdec = 630; // +-30 micron
        p2ry = 11.0; // +14 -4 GeV/c
        psum = 12; // +14 -4 GeV/c
        phi = 151; //+-1
        ptmiss = 0.3; //+-0.1
        pt2ry = 1.0; // +1.2 -0.4
        gammadecvtx=2;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtev5 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPev5  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSev5  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNev5  = reader.EvaluateMVA("TMlpANN");
        
        c1->cd(1);
        zdec5 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdec5->SetLineWidth(2);
        if (goldensilver==1) zdec5->SetLineColor(kYellow+1);
        else zdec5->SetLineColor(kGreen+4);
        zdec5->Draw();
        c1->Update();
        c1->cd(2);
        kink5 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kink5->SetLineWidth(2);
        if (goldensilver==1) kink5->SetLineColor(kYellow+1);
        else kink5->SetLineColor(kGreen+4);
        kink5->Draw();
        c1->Update();
        c1->cd(3);
        p2ry5 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ry5->SetLineWidth(2);
        if (goldensilver==1) p2ry5->SetLineColor(kYellow+1);
        else p2ry5->SetLineColor(kGreen+4);
        p2ry5->Draw();
        c1->Update();
        c1->cd(4);
        ptmiss5 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmiss5->SetLineWidth(2);
        if (goldensilver==1) ptmiss5->SetLineColor(kYellow+1);
        else ptmiss5->SetLineColor(kGreen+4);
        ptmiss5->Draw();
        c1->Update();
        c1->cd(5);
        phi5 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phi5->SetLineWidth(2);
        if (goldensilver==1) phi5->SetLineColor(kYellow+1);
        else phi5->SetLineColor(kGreen+4);
        phi5->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtx5 = new TLine(gammadecvtx+0.1,c1->cd(6)->GetUymin(),gammadecvtx+0.1,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtx5->SetLineWidth(2);
        if (goldensilver==1) gammadecvtx5->SetLineColor(kYellow+1);
        else gammadecvtx5->SetLineColor(kGreen+4);
        gammadecvtx5->Draw();
        c1->Update();
        c1->cd(7);
        pt2ry5 = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ry5->SetLineWidth(2);
        if (goldensilver==1) pt2ry5->SetLineColor(kYellow+1);
        else pt2ry5->SetLineColor(kGreen+4);
        pt2ry5->Draw();
        c1->Update();
        
        
        // ev Marginale BERNA: 11172035775
        //0mu
        nchargedvis1ry = 1;
        decay_length = 1100; //+- micron
        kink = 0.097; //+- rad
        //IP:  micron
        zdec = 652; // +- micron
        p2ry = 2.6; // [5.7,9.2] GeV/c
        psum = 26.5; // + GeV/c
        phi = 139; //+-
        ptmiss = 1.29; //[0.79,1.16]
        pt2ry = 0.25; // [0.56,0.90]
        gammadecvtx= 0;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtevBER = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPevBER  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSevBER  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNevBER  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdecBER = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdecBER->SetLineWidth(2);
        if (goldensilver==1) zdecBER->SetLineColor(kGray+1);
        else zdecBER->SetLineColor(kYellow+1);
        zdecBER->Draw();
        c1->Update();
        c1->cd(2);
        kinkBER = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kinkBER->SetLineWidth(2);
        if (goldensilver==1) kinkBER->SetLineColor(kGray+1);
        else kinkBER->SetLineColor(kYellow+1);
        kinkBER->Draw();
        c1->Update();
        c1->cd(3);
        p2ryBER = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ryBER->SetLineWidth(2);
        if (goldensilver==1) p2ryBER->SetLineColor(kGray+1);
        else p2ryBER->SetLineColor(kYellow+1);
        p2ryBER->Draw();
        c1->Update();
        c1->cd(4);
        ptmissBER = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmissBER->SetLineWidth(2);
        if (goldensilver==1) ptmissBER->SetLineColor(kGray+1);
        else ptmissBER->SetLineColor(kYellow+1);
        ptmissBER->Draw();
        c1->Update();
        c1->cd(5);
        phiBER = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phiBER->SetLineWidth(2);
        if (goldensilver==1) phiBER->SetLineColor(kGray+1);
        else phiBER->SetLineColor(kYellow+1);
        phiBER->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtxBER = new TLine(gammadecvtx+0.13,c1->cd(6)->GetUymin(),gammadecvtx+0.13,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtxBER->SetLineWidth(2);
        if (goldensilver==1) gammadecvtxBER->SetLineColor(kGray+1);
        else gammadecvtxBER->SetLineColor(kYellow+1);
        gammadecvtxBER->Draw();
        c1->Update();
        c1->cd(7);
        pt2ryBER = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ryBER->SetLineWidth(2);
        if (goldensilver==1) pt2ryBER->SetLineColor(kGray+1);
        else pt2ryBER->SetLineColor(kYellow+2);
        pt2ryBER->Draw();
        c1->Update();
        
        
        //EV MARGINALE PD_BO 11143018505
        //0mu
        kink = 0.090; //+- rad
        decay_length = 1160;//+- micron
        zdec =  429.6; //+-  micron //!581.8
        p2ry = 2.7;//[2.1–3.7] GeV/c
        psum = 23.193; //  GeV/c
        phi = 151.77; //+
        ptmiss = 0.876; // +
        pt2ry = 0.24; //  [0.20–0.35] Gev/C
        nchargedvis1ry = 2;
        gammadecvtx=1;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtevPDBO  = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPevPDBO  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSevPDBO  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNevPDBO  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdecPDBO = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdecPDBO->SetLineWidth(2);
        if (goldensilver==1) zdecPDBO->SetLineColor(kGray+1);
        else zdecPDBO->SetLineColor(kOrange+2);
        zdecPDBO->Draw();
        c1->Update();
        c1->cd(2);
        kinkPDBO = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kinkPDBO->SetLineWidth(2);
        if (goldensilver==1) kinkPDBO->SetLineColor(kGray+1);
        else kinkPDBO->SetLineColor(kOrange+2);
        kinkPDBO->Draw();
        c1->Update();
        c1->cd(3);
        p2ryPDBO = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ryPDBO->SetLineWidth(2);
        if (goldensilver==1) p2ryPDBO->SetLineColor(kGray+1);
        else p2ryPDBO->SetLineColor(kOrange+2);
        p2ryPDBO->Draw();
        c1->Update();
        c1->cd(4);
        ptmissPDBO = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmissPDBO->SetLineWidth(2);
        if (goldensilver==1) ptmissPDBO->SetLineColor(kGray+1);
        else ptmissPDBO->SetLineColor(kOrange+2);
        ptmissPDBO->Draw();
        c1->Update();
        c1->cd(5);
        phiPDBO = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phiPDBO->SetLineWidth(2);
        if (goldensilver==1) phiPDBO->SetLineColor(kGray+1);
        else phiPDBO->SetLineColor(kOrange+2);
        phiPDBO->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtxPDBO = new TLine(gammadecvtx-0.1,c1->cd(6)->GetUymin(),gammadecvtx-0.1,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtxPDBO->SetLineWidth(2);
        if (goldensilver==1) gammadecvtxPDBO->SetLineColor(kGray+1);
        else gammadecvtxPDBO->SetLineColor(kOrange+2);
        gammadecvtxPDBO->Draw();
        c1->Update();
        c1->cd(7);
        pt2ryPDBO = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ryPDBO->SetLineWidth(2);
        if (goldensilver==1) pt2ryPDBO->SetLineColor(kGray+1);
        else pt2ryPDBO->SetLineColor(kOrange+2);
        pt2ryPDBO->Draw();
        c1->Update();
        
        
        // ev Marginale NAG2: 9190097972
        //0mu
        nchargedvis1ry = 7;
        decay_length = 822; //+- micron
        kink = 0.146; //+- rad
        //IP:  micron
        zdec = 10+293; // +- micron
        p2ry = 2.24; // [1.63,3.58] GeV/c
        psum = 9.6; // + GeV/c
        phi = 146; //+-
        ptmiss = 0.46; //
        pt2ry = 0.33; //
        gammadecvtx= 0;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtevNAG2 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPevNAG2  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSevNAG2  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNevNAG2  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdecNAG2 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdecNAG2->SetLineWidth(2);
        if (goldensilver==1) zdecNAG2->SetLineColor(kGray+1);
        else zdecNAG2->SetLineColor(kRed);
        zdecNAG2->Draw();
        c1->Update();
        c1->cd(2);
        kinkNAG2 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kinkNAG2->SetLineWidth(2);
        if (goldensilver==1) kinkNAG2->SetLineColor(kGray+1);
        else kinkNAG2->SetLineColor(kRed);
        kinkNAG2->Draw();
        c1->Update();
        c1->cd(3);
        p2ryNAG2 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ryNAG2->SetLineWidth(2);
        if (goldensilver==1) p2ryNAG2->SetLineColor(kGray+1);
        else p2ryNAG2->SetLineColor(kRed);
        p2ryNAG2->Draw();
        c1->Update();
        c1->cd(4);
        ptmissNAG2 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmissNAG2->SetLineWidth(2);
        if (goldensilver==1) ptmissNAG2->SetLineColor(kGray+1);
        else ptmissNAG2->SetLineColor(kRed);
        ptmissNAG2->Draw();
        c1->Update();
        c1->cd(5);
        phiNAG2 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phiNAG2->SetLineWidth(2);
        if (goldensilver==1) phiNAG2->SetLineColor(kGray+1);
        else phiNAG2->SetLineColor(kRed);
        phiNAG2->Draw();
        c1->Update();
        c1->cd(6);
        gammadecvtxNAG2 = new TLine(gammadecvtx-0.13,c1->cd(6)->GetUymin(),gammadecvtx-0.13,(c1->cd(6)->GetUymax()/2.5));
        gammadecvtxNAG2->SetLineWidth(2);
        if (goldensilver==1) gammadecvtxNAG2->SetLineColor(kGray+1);
        else gammadecvtxNAG2->SetLineColor(kRed);
        gammadecvtxNAG2->Draw();
        c1->Update();
        c1->cd(7);
        pt2ryNAG2 = new TLine(pt2ry,c1->cd(7)->GetUymin(),pt2ry,(c1->cd(7)->GetUymax()/2.5));
        pt2ryNAG2->SetLineWidth(2);
        if (goldensilver==1) pt2ryNAG2->SetLineColor(kGray+1);
        else pt2ryNAG2->SetLineColor(kRed);
        pt2ryNAG2->Draw();
        c1->Update();
        
        
    }
    
    if (channel==3) {
        //ev2
        //0mu
        nchargedvis1ry = 2;
        kink = 0.0874;//+-0.0015 rad
        zdec =  1446; //+- 10 micron
        decay_length = 1466;//+-10 micron
        p2ry = 8.4; // +-1.7 GeV/c
        psum = 12.7 ; //(-2.3 +1.7) GeV/c
        phi = 167.8 ; //+1.1
        ptmiss = 0.31; // +-0.11
        pt2ry = -99;
        Minvmin = 0.96; // +- 0.13
        Minv = 0.80; //+-0.12
        gammadecvtx=0;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtev2 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPev2  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSev2  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNev2  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdec2 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdec2->SetLineWidth(2);
        if (goldensilver==1) zdec2->SetLineColor(kYellow+1);
        else zdec2->SetLineColor(kGreen+2);
        zdec2->Draw();
        c1->Update();
        c1->cd(2);
        kink2 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kink2->SetLineWidth(2);
        if (goldensilver==1) kink2->SetLineColor(kYellow+1);
        else kink2->SetLineColor(kGreen+2);
        kink2->Draw();
        c1->Update();
        c1->cd(3);
        p2ry2 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ry2->SetLineWidth(2);
        if (goldensilver==1) p2ry2->SetLineColor(kYellow+1);
        else p2ry2->SetLineColor(kGreen+2);
        p2ry2->Draw();
        c1->Update();
        c1->cd(4);
        ptmiss2 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmiss2->SetLineWidth(2);
        if (goldensilver==1) ptmiss2->SetLineColor(kYellow+1);
        else ptmiss2->SetLineColor(kGreen+2);
        ptmiss2->Draw();
        c1->Update();
        c1->cd(5);
        phi2 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phi2->SetLineWidth(2);
        if (goldensilver==1) phi2->SetLineColor(kYellow+1);
        else phi2->SetLineColor(kGreen+2);
        phi2->Draw();
        c1->Update();
        c1->cd(6);
        Minv2 = new TLine(Minv,c1->cd(6)->GetUymin(),Minv,(c1->cd(6)->GetUymax()/2.5));
        Minv2->SetLineWidth(2);
        if (goldensilver==1) Minv2->SetLineColor(kYellow+1);
        else Minv2->SetLineColor(kGreen+2);
        Minv2->Draw();
        c1->Update();
        //        c1->cd(7);
        //        Minvmin2 = new TLine(Minvmin,c1->cd(7)->GetUymin(),Minvmin,(c1->cd(7)->GetUymax()/2.5));
        //        Minvmin2->SetLineWidth(2);
        //        Minvmin2->SetLineColor(kGreen+2);
        //        Minvmin2->Draw();
        //        c1->Update();
        
        
        //evBARI 10123059807
        //0mu
        nchargedvis1ry = 4;
        kink =  0.231; // misura bari: 12 gradi//+- rad
        zdec = -647.602; //+-  micron
        decay_length = 140.449;//+- micron
        p2ry = 6.7; // +- GeV/c
        psum = 16.9 ; // GeV/c
        phi =  82; //+
        ptmiss = 0.6; // +-.
        pt2ry = -99;
        Minvmin = 2.0; // +-
        Minv = 1.2; //+-
        gammadecvtx=0;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtevBARI = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPevBARI  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSevBARI  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNevBARI  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdecBARI = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdecBARI->SetLineWidth(2);
        if (goldensilver==1) zdecBARI->SetLineColor(kGray+1);
        else zdecBARI->SetLineColor(kMagenta+1);
        zdecBARI->Draw();
        c1->Update();
        c1->cd(2);
        kinkBARI = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kinkBARI->SetLineWidth(2);
        if (goldensilver==1) kinkBARI->SetLineColor(kGray+1);
        else kinkBARI->SetLineColor(kMagenta+1);
        kinkBARI->Draw();
        c1->Update();
        c1->cd(3);
        p2ryBARI = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ryBARI->SetLineWidth(2);
        if (goldensilver==1) p2ryBARI->SetLineColor(kGray+1);
        else p2ryBARI->SetLineColor(kMagenta+2);
        p2ryBARI->Draw();
        c1->Update();
        c1->cd(4);
        ptmissBARI = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmissBARI->SetLineWidth(2);
        if (goldensilver==1) ptmissBARI->SetLineColor(kGray+1);
        else ptmissBARI->SetLineColor(kMagenta+1);
        ptmissBARI->Draw();
        c1->Update();
        c1->cd(5);
        phiBARI = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phiBARI->SetLineWidth(2);
        if (goldensilver==1) phiBARI->SetLineColor(kGray+1);
        else phiBARI->SetLineColor(kMagenta+1);
        phiBARI->Draw();
        c1->Update();
        c1->cd(6);
        MinvBARI = new TLine(Minv,c1->cd(6)->GetUymin(),Minv,(c1->cd(6)->GetUymax()/2.5));
        MinvBARI->SetLineWidth(2);
        if (goldensilver==1) MinvBARI->SetLineColor(kGray+1);
        else MinvBARI->SetLineColor(kMagenta+1);
        MinvBARI->Draw();
        c1->Update();
        
        
        
        //evNAG4 11213015702
        //0mu
        nchargedvis1ry = 5;
        kink =  0.083;
        zdec = 407; //+-  micron //decadimento in base //CHECK
        decay_length = 256;//+- micron
        p2ry = 6.34; // maggiore di GeV/c +- GeV/c
        psum = 6.78; // maggiore di GeV/c
        phi =  47.07; //
        ptmiss = 0.50; // +-.
        pt2ry = 0.4;
        Minvmin = 1.42; // +-
        Minv = 0.94; //+-
        gammadecvtx=2; //?
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum +((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtevNAG4 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPevNAG4  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSevNAG4  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNevNAG4  = reader.EvaluateMVA("TMlpANN");
        
        
        c1->cd(1);
        zdecNAG4 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdecNAG4->SetLineWidth(2);
        if (goldensilver==1) zdecNAG4->SetLineColor(kGray+1);
        else zdecNAG4->SetLineColor(kRed-3);
        zdecNAG4->Draw();
        c1->Update();
        c1->cd(2);
        kinkNAG4 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kinkNAG4->SetLineWidth(2);
        if (goldensilver==1) kinkNAG4->SetLineColor(kGray+1);
        else kinkNAG4->SetLineColor(kRed-3);
        kinkNAG4->Draw();
        c1->Update();
        c1->cd(3);
        p2ryNAG4 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ryNAG4->SetLineWidth(2);
        if (goldensilver==1) p2ryNAG4->SetLineColor(kGray+1);
        else p2ryNAG4->SetLineColor(kRed-3);
        p2ryNAG4->Draw();
        c1->Update();
        c1->cd(4);
        ptmissNAG4 = new TLine(ptmiss,c1->cd(4)->GetUymin(),ptmiss,(c1->cd(4)->GetUymax()/2.5));
        ptmissNAG4->SetLineWidth(2);
        if (goldensilver==1) ptmissNAG4->SetLineColor(kGray+1);
        else ptmissNAG4->SetLineColor(kRed-3);
        ptmissNAG4->Draw();
        c1->Update();
        c1->cd(5);
        phiNAG4 = new TLine(phi,c1->cd(5)->GetUymin(),phi,(c1->cd(5)->GetUymax()/2.5));
        phiNAG4->SetLineWidth(2);
        if (goldensilver==1) phiNAG4->SetLineColor(kGray+1);
        else phiNAG4->SetLineColor(kRed-3);
        phiNAG4->Draw();
        c1->Update();
        c1->cd(6);
        MinvNAG4 = new TLine(Minv,c1->cd(6)->GetUymin(),Minv,(c1->cd(6)->GetUymax()/2.5));
        MinvNAG4->SetLineWidth(2);
        if (goldensilver==1) MinvNAG4->SetLineColor(kGray+1);
        else MinvNAG4->SetLineColor(kRed-3);
        MinvNAG4->Draw();
        c1->Update();
        //        c1->cd(7);
        //        MinvminNAG4 = new TLine(Minvmin,c1->cd(7)->GetUymin(),Minvmin,(c1->cd(7)->GetUymax()/2.5));
        //        MinvminNAG4->SetLineWidth(2);
        //        MinvminNAG4->SetLineColor(kRed-3);
        //        MinvminNAG4->Draw();
        //        c1->Update();
        
        
    }
    
    if (channel==2) {
        //ev3
        //1mu
        nchargedvis1ry = 2;
        kink = 0.245;//+-0.005 rad
        zdec = 151; // +-10 micron
        decay_length = 376; //+-10 micron
        p2ry = 2.8; // +-0.2 GeV/c
        psum = 6.8; //  (-0.6 +0.9) GeV/c
        pt2ry = 0.690; // +-0.05 GeV/c
        ptmiss = 0.670;
        charge=-1;
        yBjorken = (psum*psum+((nchargedvis1ry-1)*PImass*PImass))/((psum*psum+((nchargedvis1ry-1)*PImass*PImass))+TAUmass/kink);
        
        bdtev3 = reader.EvaluateMVA("BDT method");
        //        if (Use["MLP"])  double MLPev3  = reader.EvaluateMVA("MLP");
        //        if (Use["MLPBFGS"])  double MLPBFGSev3  = reader.EvaluateMVA("MLPBFGS");
        //        if (Use["TMlpANN"])  double TMlpANNev3  = reader.EvaluateMVA("TMlpANN");
        //
        
        c1->cd(1);
        zdec3 = new TLine(zdec,c1->cd(1)->GetUymin(),zdec,(c1->cd(1)->GetUymax()/2.5));
        zdec3->SetLineWidth(2);
        if (goldensilver==1) zdec3->SetLineColor(kYellow+1);
        else zdec3->SetLineColor(kGreen+2);
        zdec3->Draw();
        c1->Update();
        c1->cd(2);
        kink3 = new TLine(kink,c1->cd(2)->GetUymin(),kink,(c1->cd(2)->GetUymax()/2.5));
        kink3->SetLineWidth(2);
        if (goldensilver==1) kink3->SetLineColor(kYellow+1);
        else kink3->SetLineColor(kGreen+2);
        kink3->Draw();
        c1->Update();
        c1->cd(3);
        p2ry3 = new TLine(p2ry,c1->cd(3)->GetUymin(),p2ry,(c1->cd(3)->GetUymax()/2.5));
        p2ry3->SetLineWidth(2);
        if (goldensilver==1) p2ry3->SetLineColor(kYellow+1);
        else p2ry3->SetLineColor(kGreen+2);
        p2ry3->Draw();
        c1->Update();
        c1->cd(4);
        pt2ry3 = new TLine(pt2ry,c1->cd(4)->GetUymin(),pt2ry,(c1->cd(4)->GetUymax()/2.5));
        pt2ry3->SetLineWidth(2);
        if (goldensilver==1) pt2ry3->SetLineColor(kYellow+1);
        else pt2ry3->SetLineColor(kGreen+2);
        pt2ry3->Draw();
        c1->Update();
        c1->cd(5);
        charge3 = new TLine(-0.5,c1->cd(5)->GetUymin(),-0.5,(c1->cd(5)->GetUymax()/2.5));
        charge3->SetLineWidth(2);
        if (goldensilver==1) charge3->SetLineColor(kYellow+1);
        else charge3->SetLineColor(kGreen+2);
        charge3->Draw();
        c1->Update();
        
    }
    
    gStyle->SetTextSize(2);
    
    //h_bdt_S->Scale(h_bdt_B->Integral()/h_bdt_S->Integral());
    if (channel==1){
        h_bdt_S->Scale(nexp_S_1h/h_bdt_S->Integral()); //pt cut 0.1
        h_bdt_B->Scale(nexp_B_1h/h_bdt_B->Integral()); //pt cut 0.1
        //        h_bdt_S->Scale(2.14/h_bdt_S->Integral()); //ptcut 0.2
        //        h_bdt_B->Scale(1.1/h_bdt_B->Integral()); //ptcut 0.2
        
        //        if (Use["MLP"])  h_MLP_S->Scale(nexp_S_1h/h_MLP_S->Integral());
        //        if (Use["MLP"])  h_MLP_B->Scale(nexp_B_1h/h_MLP_B->Integral());
        //
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Scale(nexp_S_1h/h_MLPBFGS_S->Integral());
        //        if (Use["MLPBFGS"])  h_MLPBFGS_B->Scale(nexp_B_1h/h_MLPBFGS_B->Integral());
        //
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Scale(nexp_S_1h/h_TMlpANN_S->Integral());
        //        if (Use["TMlpANN"])  h_TMlpANN_B->Scale(nexp_B_1h/h_TMlpANN_B->Integral());
        //
    }
    if (channel==2){
        h_bdt_S->Scale(nexp_S_mu/h_bdt_S->Integral());
        h_bdt_B->Scale(nexp_B_mu/h_bdt_B->Integral());
        
        //        if (Use["MLP"])  h_MLP_S->Scale(nexp_S_mu/h_MLP_S->Integral());
        //        if (Use["MLP"])  h_MLP_B->Scale(nexp_B_mu/h_MLP_B->Integral());
        //
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Scale(nexp_S_mu/h_MLPBFGS_S->Integral());
        //        if (Use["MLPBFGS"])  h_MLPBFGS_B->Scale(nexp_B_mu/h_MLPBFGS_B->Integral());
        //
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Scale(nexp_S_mu/h_TMlpANN_S->Integral());
        //        if (Use["TMlpANN"])  h_TMlpANN_B->Scale(nexp_B_mu/h_TMlpANN_B->Integral());
        
    }
    if (channel==3){
        h_bdt_S->Scale(nexp_S_3h/h_bdt_S->Integral());
        h_bdt_B->Scale(nexp_B_3h/h_bdt_B->Integral());
        
        //        if (Use["MLP"])  h_MLP_S->Scale(nexp_S_3h/h_MLP_S->Integral());
        //        if (Use["MLP"])  h_MLP_B->Scale(nexp_B_3h/h_MLP_B->Integral());
        //
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Scale(nexp_S_3h/h_MLPBFGS_S->Integral());
        //        if (Use["MLPBFGS"])  h_MLPBFGS_B->Scale(nexp_B_3h/h_MLPBFGS_B->Integral());
        //
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Scale(nexp_S_3h/h_TMlpANN_S->Integral());
        //        if (Use["TMlpANN"])  h_TMlpANN_B->Scale(nexp_B_3h/h_TMlpANN_B->Integral());
        
    }
    if (channel==4){
        h_bdt_S->Scale(nexp_S_e/h_bdt_S->Integral());
        h_bdt_B->Scale(nexp_B_e/h_bdt_B->Integral());
        
        //        if (Use["MLP"])  h_MLP_S->Scale(nexp_S_e/h_MLP_S->Integral());
        //        if (Use["MLP"])  h_MLP_B->Scale(nexp_B_e/h_MLP_B->Integral());
        //
        //        if (Use["MLPBFGS"])  h_MLPBFGS_S->Scale(nexp_S_e/h_MLPBFGS_S->Integral());
        //        if (Use["MLPBFGS"])  h_MLPBFGS_B->Scale(nexp_B_e/h_MLPBFGS_B->Integral());
        //
        //        if (Use["TMlpANN"])  h_TMlpANN_S->Scale(nexp_S_e/h_TMlpANN_S->Integral());
        //        if (Use["TMlpANN"])  h_TMlpANN_B->Scale(nexp_B_e/h_TMlpANN_B->Integral());
    }
    
    c->cd();
    //h_bdt_S->SetMaximum(50);
    h_bdt_S->Draw("HISTOsames");
    h_bdt_B->Draw("HISTOsames");
    h_bdt_S->GetYaxis()->SetTitleOffset(1.5);
    h_bdt_B->GetYaxis()->SetTitleOffset(1.5);
    
    //    if (Use["MLP"]) cMLP->cd();
    //    if (Use["MLP"])  h_MLP_S->Draw("HISTOsames");
    //    if (Use["MLP"])  h_MLP_B->Draw("HISTOsames");
    //    if (Use["MLP"])  h_MLP_S->GetYaxis()->SetTitleOffset(1.5);
    //    if (Use["MLP"])  h_MLP_B->GetYaxis()->SetTitleOffset(1.5);
    //
    //    if (Use["MLPBFGS"]) cMLPBFGS->cd();
    //    if (Use["MLPBFGS"])  h_MLPBFGS_S->Draw("HISTOsames");
    //    if (Use["MLPBFGS"])  h_MLPBFGS_B->Draw("HISTOsames");
    //    if (Use["MLPBFGS"])  h_MLPBFGS_S->GetYaxis()->SetTitleOffset(1.5);
    //    if (Use["MLPBFGS"])  h_MLPBFGS_B->GetYaxis()->SetTitleOffset(1.5);
    //
    //    if (Use["TMlpANN"]) cTMlpANN->cd();
    //    if (Use["TMlpANN"])  h_TMlpANN_S->Draw("HISTOsames");
    //    if (Use["TMlpANN"])  h_TMlpANN_B->Draw("HISTOsames");
    //    if (Use["TMlpANN"])  h_TMlpANN_S->GetYaxis()->SetTitleOffset(1.5);
    //    if (Use["TMlpANN"])  h_TMlpANN_B->GetYaxis()->SetTitleOffset(1.5);
    //
    
    c->Update();
    //    if (Use["MLP"]) cMLP->Update();
    //    if (Use["MLPBFGS"]) cMLPBFGS->Update();
    //    if (Use["TMlpANN"]) cTMlpANN->Update();
    
    
    if (channel==2) {
        double xdown_fitS = -0.80;
        double xup_fitS = -0.09;
        TF1 *f_bdt_S = new TF1("f_bdt_S","expo",xdown_fitS,xup_fitS);
        f_bdt_S->FixParameter(0, -5.6721);
        f_bdt_S->FixParameter(1, 13.4821);
        
        double xdown_fitB1 = -0.345;
        double xup_fitB1 = -0.20;
        TF1 *f_bdt_B1 = new TF1("f_bdt_B1","expo",xdown_fitB1,xup_fitB1);
        f_bdt_B1->FixParameter(0, -27.9322);
        f_bdt_B1->FixParameter(1,-60.0313);
        
        double xdown_fitB2 = xup_fitB1;
        double xup_fitB2 = -0.09;
        TF1 *f_bdt_B2 = new TF1("f_bdt_B2","expo",xdown_fitB2,xup_fitB2);
        f_bdt_B2->FixParameter(0, -0.521906);
        f_bdt_B2->FixParameter(1,69.2506);
        
        double xdown_fitB3 = 0.19;
        double xup_fitB3 = 0.9;
        TF1 *f_bdt_B3 = new TF1("f_bdt_B3","expo",xdown_fitB3,xup_fitB3);
        f_bdt_B3->FixParameter(0, -7.05866);
        f_bdt_B3->FixParameter(1,-7.82666);
        
        //        f_bdt_S->Draw("same");
        //        f_bdt_B1->Draw("same");
        //        f_bdt_B2->Draw("same");
        //        f_bdt_B3->Draw("same");
        c->Update();
    }
    
    //CALCOLO TAGLIO BDT
    
    float cut[100]={0};
    float tau_integral[100]={-9999};
    float charm_integral[100]={-9999};
    float tau_Efficiency[100]={-9999};
    float tau_Purity[100]={-9999};
    float zero[100]={0};
    float err_Eff[100]={0};
    float err_Pur[100]={0};
    
    cout << "start: " << h_bdt_S->GetBinLowEdge(0) << endl;
    
    int nbinjj=h_bdt_S->FindLastBinAbove(0,1);
    int bmax = h_bdt_S->GetNbinsX();//FindLastBinAbove(0,1);
    int bmin =0;
    TAxis *axis = h_bdt_S->GetXaxis();
    for (int jj=0; jj<nbinjj; jj++) {
        
        bmin = jj;//axis->FindBin(cut[jj]);
        cut[jj]=h_bdt_S->GetBinLowEdge(jj);//+(float(jj)*1.2/h_bdt_S->GetNbinsX());
        
        tau_integral[jj] = h_bdt_S->Integral(bmin,bmax);
        charm_integral[jj] = h_bdt_B->Integral(bmin,bmax);
        
        tau_Efficiency[jj] = tau_integral[jj]/tau_integral[0];
        tau_Purity[jj] = tau_integral[jj]/(tau_integral[jj]+charm_integral[jj]);
        
        err_Eff[jj]=0;//TMath::Sqrt((tau_Efficiency[jj]*(1-tau_Efficiency[jj]))/(tau_integral[0]));
        err_Pur[jj]=0;//TMath::Sqrt((tau_Purity[jj]*(1-tau_Purity[jj]))/(tau_integral[0]));
        
        
        //        cout << "Taglio a " << cut[jj] << "\tBin: " << bmin << "\t" << bmax << endl;
        //        cout << "\tIntegrale tau = " << tau_integral[jj] << "\tEfficienza: " << tau_Efficiency[jj] << "\tPurezza: " << tau_Purity[jj] << endl;
        //
        //
        //        h_bdt_S->GetMaximumBin();
        //        h_bdt_B->GetMaximumBin();
        
    }
    
    TGraphErrors *gr_Efficiency = new TGraphErrors(nbinjj, cut, tau_Efficiency, zero, err_Eff);
    TGraphErrors *gr_Purity = new TGraphErrors(nbinjj, cut, tau_Purity, zero, err_Pur);
    
    Float_t EP[100];
    float temp=0;
    int ind=0;
    float suggestedcut=0;
    for (int jjj=0; jjj<nbinjj; jjj++) {
        EP[jjj]=tau_Efficiency[jjj]*tau_Purity[jjj];
        //cout << jjj << "\t" << cut[jjj] << "\t E*p: " << EP[jjj] << endl;
        if (EP[jjj]>=temp) {
            suggestedcut=cut[jjj];
            temp=EP[jjj];
            ind=jjj;
        }
    }
    
    cout << "TAGLIO SUGGERITO: " << suggestedcut << " sig: " << tau_Efficiency[ind]*100 << "%, bkg: " << 100-(charm_integral[ind]/charm_integral[0])*100 << "%" << endl;
    
    
    TGraphErrors *gr_Max = new TGraphErrors(nbinjj, cut, EP, zero, zero);
    
    c2->cd(1);
    
    gr_Efficiency->SetTitle("Efficiency and Purity vs cut");
    gr_Efficiency->GetXaxis()->SetTitle("BDT response");
    gr_Purity->GetXaxis()->SetTitle("BDT response");
    
    //    gr_Efficiency->SetMarkerColor(kBlue);
    //    gr_Efficiency->SetMarkerStyle(22);
    gr_Efficiency->SetLineColor(kBlue);
    gr_Efficiency->SetLineWidth(2);
    gr_Efficiency->Draw("AC");
    
    //    gr_Purity->SetMarkerColor(kRed);
    //    gr_Purity->SetMarkerStyle(23);
    gr_Purity->SetLineColor(kRed);
    gr_Purity->SetLineWidth(2);
    gr_Purity->Draw("C");
    
    TLegend *legend1 = new TLegend(.80,.50,.95,.65);
    legend1->AddEntry(gr_Efficiency,"Efficiency", "l");
    legend1->AddEntry(gr_Purity,"Purity", "l");
    legend1->Draw("same");
    
    c2->cd(2);
    gr_Max->SetLineColor(kBlack);
    gr_Max->SetLineWidth(2);
    gr_Max->SetTitle("Efficiency*Purity");
    gr_Max->GetXaxis()->SetTitle("BDT response");
    gr_Max->Draw("AC");
    
    //    if (Use["MLP"]) {
    //
    //        //CALCOLO TAGLIO MLP
    //
    //        float cut_MLP[500]=0;
    //        float tau_integral_MLP[500]={-9999};
    //        float charm_integral_MLP[500]={-9999};
    //        float tau_Efficiency_MLP[500]={-9999};
    //        float tau_Purity_MLP[500]={-9999};
    //        float err_Eff_MLP[500]={0};
    //        float err_Pur_MLP[500]={0};
    //
    //        cout << "start: " << h_MLP_S->GetBinLowEdge(0) << endl;
    //
    //        int nbinjj_=h_MLP_S->FindLastBinAbove(0,1);
    //        bmax = h_MLP_S->GetNbinsX();//FindLastBinAbove(0,1);
    //       TAxis *axis = h_MLP_S->GetXaxis();
    
    //        for (int jj=0; jj<nbinjj_; jj++) {
    //            bmin = jj;//axis->FindBin(cut[jj]);
    //            cut_MLP[jj]=h_MLP_S->GetBinLowEdge(jj);//+(float(jj)*1.2/h_bdt_S->GetNbinsX());
    //
    //            tau_integral_MLP[jj] = h_MLP_S->Integral(bmin,bmax);
    //            charm_integral_MLP[jj] = h_MLP_B->Integral(bmin,bmax);
    //
    //            tau_Efficiency_MLP[jj] = tau_integral_MLP[jj]/tau_integral_MLP[0];
    //            tau_Purity_MLP[jj] = tau_integral_MLP[jj]/(tau_integral_MLP[jj]+charm_integral_MLP[jj]);
    //
    //            //cout << "MLP: " << jj << "\t\t" << tau_Efficiency_MLP[jj] << "\t" << tau_Purity_MLP[jj] << " (" << tau_integral_MLP[jj] << ")/" << (tau_integral_MLP[jj]+charm_integral_MLP[jj]) << ")\t" << endl;
    //
    //            err_Eff_MLP[jj]=0;//TMath::Sqrt((tau_Efficiency[jj]*(1-tau_Efficiency[jj]))/(tau_integral[0]));
    //            err_Pur_MLP[jj]=0;//TMath::Sqrt((tau_Purity[jj]*(1-tau_Purity[jj]))/(tau_integral[0]));
    //
    //        }
    //
    //        Float_t EP_MLP[500];
    //        float temp_=0;
    //        int ind_=0;
    //        float suggestedcut_MLP=0;
    //        for (int jjj=0; jjj<nbinjj_; jjj++) {
    //            EP_MLP[jjj]=tau_Efficiency_MLP[jjj]*tau_Purity_MLP[jjj];
    //            if (EP_MLP[jjj]>=temp_) {
    //                suggestedcut_MLP=cut_MLP[jjj];
    //                temp_=EP_MLP[jjj];
    //                ind_=jjj;
    //            }
    //        }
    //
    //        TGraphErrors *gr_Efficiency_MLP = new TGraphErrors(nbinjj_, cut_MLP, tau_Efficiency_MLP, zero, err_Eff_MLP);
    //        TGraphErrors *gr_Purity_MLP = new TGraphErrors(nbinjj_, cut_MLP, tau_Purity_MLP, zero, err_Pur_MLP);
    //
    //        gr_Efficiency_MLP->SetTitle("Efficiency and Purity vs cut MLP");
    //        gr_Efficiency_MLP->GetXaxis()->SetTitle("MLP response");
    //        gr_Purity_MLP->GetXaxis()->SetTitle("MLP response");
    //
    //        TGraphErrors *gr_Max_MLP = new TGraphErrors(nbinjj_, cut_MLP, EP_MLP, zero, zero);
    //
    //        c2MLP->cd(1);
    //        //    gr_Efficiency_MLP->SetMarkerColor(kBlue);
    //        //    gr_Efficiency_MLP->SetMarkerStyle(22);
    //        gr_Efficiency_MLP->SetLineColor(kBlue);
    //        gr_Efficiency_MLP->SetLineWidth(2);
    //        gr_Efficiency_MLP->Draw("AC");
    //
    //        //    gr_Purity_MLP->SetMarkerColor(kRed);
    //        //    gr_Purity_MLP->SetMarkerStyle(23);
    //        gr_Purity_MLP->SetLineColor(kRed);
    //        gr_Purity_MLP->SetLineWidth(2);
    //        gr_Purity_MLP->Draw("C");
    //
    //        legend1->Draw("same");
    //
    //        c2MLP->cd(2);
    //        gr_Max_MLP->SetLineColor(kBlack);
    //        gr_Max_MLP->SetLineWidth(2);
    //        gr_Max_MLP->SetTitle("Efficiency*Purity");
    //        gr_Max_MLP->GetXaxis()->SetTitle("MLP response");
    //        gr_Max_MLP->Draw("AC");
    //
    //
    //        cout << "TAGLIO SUGGERITO MLP: " << suggestedcut_MLP << " sig: " << tau_Efficiency_MLP[ind] << "%, bkg: " << 1-(charm_integral_MLP[ind]/charm_integral_MLP[0]) << "%" << endl;
    //
    //    }
    //
    //
    
    //
    //    if (Use["MLPBFGS"]) {
    //
    //        //CALCOLO TAGLIO MLPBFGS
    //
    //        float cut_MLPBFGS[100]=0;
    //        float tau_integral_MLPBFGS[100]={-9999};
    //        float charm_integral_MLPBFGS[100]={-9999};
    //        float tau_Efficiency_MLPBFGS[100]={-9999};
    //        float tau_Purity_MLPBFGS[100]={-9999};
    //        float err_Eff_MLPBFGS[100]={0};
    //        float err_Pur_MLPBFGS[100]={0};
    //
    //        cout << "start: " << h_MLPBFGS_S->GetBinLowEdge(0) << endl;
    //
    //        int nbinjj_=h_MLPBFGS_S->FindLastBinAbove(0,1);
    //        bmax = h_MLPBFGS_S->GetNbinsX();//FindLastBinAbove(0,1);
    //            TAxis *axis = h_MLPBFGS_S->GetXaxis();
    //        for (int jj=0; jj<nbinjj_; jj++) {
    //
    //            bmin = jj;//axis->FindBin(cut[jj]);
    //            cut_MLPBFGS[jj]=h_MLPBFGS_S->GetBinLowEdge(jj);//+(float(jj)*1.2/h_bdt_S->GetNbinsX());
    //
    //            tau_integral_MLPBFGS[jj] = h_MLPBFGS_S->Integral(bmin,bmax);
    //            charm_integral_MLPBFGS[jj] = h_MLPBFGS_B->Integral(bmin,bmax);
    //
    //            tau_Efficiency_MLPBFGS[jj] = tau_integral_MLPBFGS[jj]/tau_integral_MLPBFGS[0];
    //            tau_Purity_MLPBFGS[jj] = tau_integral_MLPBFGS[jj]/(tau_integral_MLPBFGS[jj]+charm_integral_MLPBFGS[jj]);
    //
    //            cout << "MLPBFGS: " << jj << "\t\t" << tau_Efficiency_MLPBFGS[jj] << "\t" << tau_Purity_MLPBFGS[jj] << " (" << tau_integral_MLPBFGS[jj] << ")/" << (tau_integral_MLPBFGS[jj]+charm_integral_MLPBFGS[jj]) << ")\t" << endl;
    //
    //            err_Eff_MLPBFGS[jj]=0;//TMath::Sqrt((tau_Efficiency[jj]*(1-tau_Efficiency[jj]))/(tau_integral[0]));
    //            err_Pur_MLPBFGS[jj]=0;//TMath::Sqrt((tau_Purity[jj]*(1-tau_Purity[jj]))/(tau_integral[0]));
    //
    //        }
    //
    //        Float_t EP_MLPBFGS[100];
    //        float temp_=0;
    //        int ind_=0;
    //        float suggestedcut_MLPBFGS=0;
    //        for (int jjj=0; jjj<nbinjj_; jjj++) {
    //            EP_MLPBFGS[jjj]=tau_Efficiency_MLPBFGS[jjj]*tau_Purity_MLPBFGS[jjj];
    //            if (EP_MLPBFGS[jjj]>=temp_) {
    //                suggestedcut_MLPBFGS=cut_MLPBFGS[jjj];
    //                temp_=EP_MLPBFGS[jjj];
    //                ind_=jjj;
    //            }
    //        }
    //
    //        cout << "TAGLIO SUGGERITO MLPBFGS: " << suggestedcut_MLPBFGS << " sig: " << tau_Efficiency_MLPBFGS[ind] << "%, bkg: " << 1-(charm_integral_MLPBFGS[ind]/charm_integral_MLPBFGS[0]) << "%" << endl;
    //
    //
    //        TGraphErrors *gr_Efficiency_MLPBFGS = new TGraphErrors(nbinjj_, cut_MLPBFGS, tau_Efficiency_MLPBFGS, zero, err_Eff_MLPBFGS);
    //        TGraphErrors *gr_Purity_MLPBFGS = new TGraphErrors(nbinjj_, cut_MLPBFGS, tau_Purity_MLPBFGS, zero, err_Pur_MLPBFGS);
    //
    //        gr_Efficiency_MLPBFGS->SetTitle("Efficiency and Purity vs cut MLPBFGS");
    //        gr_Efficiency_MLPBFGS->GetXaxis()->SetTitle("MLPBFGS response");
    //        gr_Purity_MLPBFGS->GetXaxis()->SetTitle("MLPBFGS response");
    //
    //
    //        c2MLPBFGS->cd(1);
    //        //    gr_Efficiency_MLPBFGS->SetMarkerColor(kBlue);
    //        //    gr_Efficiency_MLPBFGS->SetMarkerStyle(22);
    //        gr_Efficiency_MLPBFGS->SetLineColor(kBlue);
    //        gr_Efficiency_MLPBFGS->SetLineWidth(2);
    //        gr_Efficiency_MLPBFGS->Draw("AC");
    //
    //        //    gr_Purity_MLPBFGS->SetMarkerColor(kRed);
    //        //    gr_Purity_MLPBFGS->SetMarkerStyle(23);
    //        gr_Purity_MLPBFGS->SetLineColor(kRed);
    //        gr_Purity_MLPBFGS->SetLineWidth(2);
    //        gr_Purity_MLPBFGS->Draw("C");
    //
    //        legend1->Draw("same");
    //
    //        TGraphErrors *gr_Max_MLPBFGS = new TGraphErrors(nbinjj_, cut_MLPBFGS, EP_MLPBFGS, zero, zero);
    //
    //        c2MLPBFGS->cd(2);
    //        gr_Max_MLPBFGS->SetLineColor(kBlack);
    //        gr_Max_MLPBFGS->SetLineWidth(2);
    //        gr_Max_MLPBFGS->SetTitle("Efficiency*Purity");
    //        gr_Max_MLPBFGS->GetXaxis()->SetTitle("MLPBFGS response");
    //        gr_Max_MLPBFGS->Draw("AC");
    //
    //    }
    //
    
    //int nbinjj_ =0;
    //    if (Use["TMlpANN"]) {
    //
    //        //CALCOLO TAGLIO TMlpANN
    //
    //        float cut_TMlpANN[100]=0;
    //        float tau_integral_TMlpANN[100]={-9999};
    //        float charm_integral_TMlpANN[100]={-9999};
    //        float tau_Efficiency_TMlpANN[100]={-9999};
    //        float tau_Purity_TMlpANN[100]={-9999};
    //        float err_Eff_TMlpANN[100]={0};
    //        float err_Pur_TMlpANN[100]={0};
    //
    //        cout << "start: " << h_TMlpANN_S->GetBinLowEdge(0) << endl;
    //
    //        nbinjj_=h_TMlpANN_S->FindLastBinAbove(0,1);
    //        bmax = h_TMlpANN_S->GetNbinsX();//FindLastBinAbove(0,1);
    //            TAxis *axis = h_TMlpANN_S->GetXaxis();
    
    //        for (int jj=0; jj<nbinjj_; jj++) {
    //
    //            bmin = jj;//axis->FindBin(cut[jj]);
    //            cut_TMlpANN[jj]=h_TMlpANN_S->GetBinLowEdge(jj);//+(float(jj)*1.2/h_bdt_S->GetNbinsX());
    //
    //            tau_integral_TMlpANN[jj] = h_TMlpANN_S->Integral(bmin,bmax);
    //            charm_integral_TMlpANN[jj] = h_TMlpANN_B->Integral(bmin,bmax);
    //
    //            tau_Efficiency_TMlpANN[jj] = tau_integral_TMlpANN[jj]/tau_integral_TMlpANN[0];
    //            tau_Purity_TMlpANN[jj] = tau_integral_TMlpANN[jj]/(tau_integral_TMlpANN[jj]+charm_integral_TMlpANN[jj]);
    //
    //            err_Eff_TMlpANN[jj]=0;//TMath::Sqrt((tau_Efficiency[jj]*(1-tau_Efficiency[jj]))/(tau_integral[0]));
    //            err_Pur_TMlpANN[jj]=0;//TMath::Sqrt((tau_Purity[jj]*(1-tau_Purity[jj]))/(tau_integral[0]));
    //
    //        }
    //
    //
    //        Float_t EP_TMlpANN[100];
    //        float temp_=0;
    //        int ind_=0;
    //        float suggestedcut_TMlpANN=0;
    //        for (int jjj=0; jjj<nbinjj_; jjj++) {
    //            EP_TMlpANN[jjj]=tau_Efficiency_TMlpANN[jjj]*tau_Purity_TMlpANN[jjj];
    //            if (EP_TMlpANN[jjj]>=temp_) {
    //                suggestedcut_TMlpANN=cut_TMlpANN[jjj];
    //                temp_=EP_TMlpANN[jjj];
    //                ind_=jjj;
    //            }
    //        }
    //
    //        cout << "TAGLIO SUGGERITO TMlpANN: " << suggestedcut_TMlpANN << " sig: " << tau_Efficiency_TMlpANN[ind] << "%, bkg: " << 1-(charm_integral_TMlpANN[ind]/charm_integral_TMlpANN[0]) << "%" << endl;
    //
    //        TGraphErrors *gr_Efficiency_TMlpANN = new TGraphErrors(nbinjj_, cut_TMlpANN, tau_Efficiency_TMlpANN, zero, err_Eff_TMlpANN);
    //        TGraphErrors *gr_Purity_TMlpANN = new TGraphErrors(nbinjj_, cut_TMlpANN, tau_Purity_TMlpANN, zero, err_Pur_TMlpANN);
    //
    //        gr_Efficiency_TMlpANN->SetTitle("Efficiency and Purity vs cut TMlpANN");
    //        gr_Efficiency_TMlpANN->GetXaxis()->SetTitle("TMlpANN response");
    //        gr_Purity_TMlpANN->GetXaxis()->SetTitle("TMlpANN response");
    //
    //
    //        c2TMlpANN->cd(1);
    //        //    gr_Efficiency_TMlpANN->SetMarkerColor(kBlue);
    //        //    gr_Efficiency_TMlpANN->SetMarkerStyle(22);
    //        gr_Efficiency_TMlpANN->SetLineColor(kBlue);
    //        gr_Efficiency_TMlpANN->SetLineWidth(2);
    //        gr_Efficiency_TMlpANN->Draw("AC");
    //
    //        //    gr_Purity_TMlpANN->SetMarkerColor(kRed);
    //        //    gr_Purity_TMlpANN->SetMarkerStyle(23);
    //        gr_Purity_TMlpANN->SetLineColor(kRed);
    //        gr_Purity_TMlpANN->SetLineWidth(2);
    //        gr_Purity_TMlpANN->Draw("C");
    //
    //        legend1->Draw("same");
    //
    //        TGraphErrors *gr_Max_TMlpANN = new TGraphErrors(nbinjj_, cut_TMlpANN, EP_TMlpANN, zero, zero);
    //
    //
    //        c2TMlpANN->cd(2);
    //        gr_Max_TMlpANN->SetLineColor(kBlack);
    //        gr_Max_TMlpANN->SetLineWidth(2);
    //        gr_Max_TMlpANN->SetTitle("Efficiency*Purity");
    //        gr_Max_TMlpANN->GetXaxis()->SetTitle("TMlpANN response");
    //        gr_Max_TMlpANN->Draw("AC");
    //
    //
    //    }
    //
    
    c->cd();
    
    
    TLine *cutline = new TLine(suggestedcut,c->GetUymin(),suggestedcut,c->GetUymax());
    cutline->SetLineWidth(2);
    cutline->SetLineColor(kBlue-8);
    if(channel!=4)cutline->Draw();
    
    TArrow *freccia= new TArrow(suggestedcut, (c->GetUymax()/2), suggestedcut+0.07, (c->GetUymax()/2), 0.009, ">");
    freccia->SetLineColor(kBlue-8);
    freccia->SetLineWidth(2);
    if(channel!=4)freccia->Draw();
    
    
    TLine *event1, *event2, *event3, *event4, *event5, *eventBER, *eventBARI, *eventPDBO, *eventNAG2, *eventNAG4;
    if (channel==1) {
        event1 = new TLine(bdtev1,c->GetUymin(),bdtev1,(c->GetUymax()/2.5));
        event1->SetLineWidth(2);
        if (goldensilver==1) event1->SetLineColor(kYellow+1);
        else event1->SetLineColor(kGreen+2);
        event1->Draw();
        event4 = new TLine(bdtev4,c->GetUymin(),bdtev4,(c->GetUymax()/2.5));
        event4->SetLineWidth(2);
        if (goldensilver==1) event4->SetLineColor(kYellow+1);
        else event4->SetLineColor(kGreen+3);
        event4->Draw();
        event5 = new TLine(bdtev5,c->GetUymin(),bdtev5,(c->GetUymax()/2.5));
        event5->SetLineWidth(2);
        if (goldensilver==1) event5->SetLineColor(kYellow+1);
        else event5->SetLineColor(kGreen+4);
        event5->Draw();
        eventBER = new TLine(bdtevBER,c->GetUymin(),bdtevBER,(c->GetUymax()/2.5));
        eventBER->SetLineWidth(2);
        if (goldensilver==1) eventBER->SetLineColor(kGray+1);
        else eventBER->SetLineColor(kYellow+1);
        eventBER->Draw();
        eventPDBO = new TLine(bdtevPDBO,c->GetUymin(),bdtevPDBO,(c->GetUymax()/2.5));
        eventPDBO->SetLineWidth(2);
        if (goldensilver==1) eventPDBO->SetLineColor(kGray+1);
        else eventPDBO->SetLineColor(kOrange+1);
        eventPDBO->Draw();
        //        eventNAG1 = new TLine(bdtevNAG1,c->GetUymin(),bdtevNAG1,(c->GetUymax()/2.5));
        //        eventNAG1->SetLineWidth(2);
        //        eventNAG1->SetLineColor(kBlue-4);
        //        eventNAG1->Draw();
        eventNAG2 = new TLine(bdtevNAG2,c->GetUymin(),bdtevNAG2,(c->GetUymax()/2.5));
        eventNAG2->SetLineWidth(2);
        if (goldensilver==1) eventNAG2->SetLineColor(kGray+1);
        else eventNAG2->SetLineColor(kRed);
        eventNAG2->Draw();
        
        cout << "BDT response: bdtev1 " << bdtev1 << "\tbdtev4 " << bdtev4 << "\tbdtev5 " << bdtev5 << "\tbdtevBER " << bdtevBER << "\tbdtevPDBO " << bdtevPDBO << "\tbdtevNAG2 " << bdtevNAG2 << endl;
        
    }
    if (channel==3) {
        event2 = new TLine(bdtev2,c->GetUymin(),bdtev2,(c->GetUymax()/2.5));
        event2->SetLineWidth(2);
        if (goldensilver==1) event2->SetLineColor(kYellow+1);
        else event2->SetLineColor(kGreen+2);
        event2->Draw();
        eventBARI = new TLine(bdtevBARI,c->GetUymin(),bdtevBARI,(c->GetUymax()/2.5));
        eventBARI->SetLineWidth(2);
        if (goldensilver==1) eventBARI->SetLineColor(kGray+1);
        else eventBARI->SetLineColor(kMagenta+1);
        eventBARI->Draw();
        //        eventNAG3 = new TLine(bdtevNAG3,c->GetUymin(),bdtevNAG3,(c->GetUymax()/2.5));
        //        eventNAG3->SetLineWidth(2);
        //        eventNAG3->SetLineColor(kYellow+1);
        //        eventNAG3->Draw();
        eventNAG4 = new TLine(bdtevNAG4,c->GetUymin(),bdtevNAG4,(c->GetUymax()/2.5));
        eventNAG4->SetLineWidth(2);
        if (goldensilver==1) eventNAG4->SetLineColor(kGray+1);
        else eventNAG4->SetLineColor(kRed-3);
        eventNAG4->Draw();
        
        cout << "BDT response: bdtev2 " << bdtev2 << "\tbdtevBARI " << bdtevBARI << "\tbdtevNAG4 " << bdtevNAG4 << endl;
        
    }
    if (channel==2) {
        event3 = new TLine(bdtev3,c->GetUymin(),bdtev3,(c->GetUymax()/2.5));
        event3->SetLineWidth(2);
        if (goldensilver==1) event3->SetLineColor(kYellow+1);
        else event3->SetLineColor(kGreen+2);
        event3->Draw();
        
        cout << "BDT response: bdtev3 " << bdtev3 << endl;
    }
    
    TLegend *legend_k = new TLegend(.25,.85,.35,.70);
    legend_k->AddEntry(h_bdt_S,"#nu_{#tau}","f");
    legend_k->AddEntry(h_bdt_B,"bkg","f");
    //if(channel==1)legend_k->AddEntry(event1,"\"golden\" candidates","l");
    if(channel==2)legend_k->AddEntry(event3,"\"golden\" candidate","l");
    //if(channel==3)legend_k->AddEntry(event2,"\"golden\" candidates","l");
    
    //if(channel=1)legend_k->AddEntry(eventBER,"\"silver\" candidates","l");
    //if(channel==3)legend_k->AddEntry(eventBARI,"\"silver\" candidates","l");
    
    legend_k->Draw("same");
    //
    
    
    
    //   TLine *cutline_MLP;
    //TArrow *freccia_MLP ;
    //    if (Use["MLP"]) {
    //
    //        cMLP->cd();
    //
    //
    //        cutline_MLP = new TLine(suggestedcut_MLP,cMLP->GetUymin(),suggestedcut_MLP,cMLP->GetUymax());
    //        cutline_MLP->SetLineWidth(2);
    //        cutline_MLP->SetLineColor(kBlue-8);
    //        if(channel!=4)cutline_MLP->Draw();
    //
    //        freccia_MLP= new TArrow(suggestedcut_MLP, (cMLP->GetUymax()/2), suggestedcut_MLP+0.07, (cMLP->GetUymax()/2), 0.009, ">");
    //        freccia_MLP->SetLineColor(kBlue-8);
    //        freccia_MLP->SetLineWidth(2);
    //        if(channel!=4)freccia_MLP->Draw();
    //
    //
    //        if (channel==1) {
    //            TLine *event1_MLP = new TLine(MLPev1,cMLP->GetUymin(),MLPev1,(cMLP->GetUymax()/2.5));
    //            event1_MLP->SetLineWidth(2);
    //            if (goldensilver==1) event1_MLP->SetLineColor(kYellow+1);
    //            else event1_MLP->SetLineColor(kGreen+2);
    //            event1_MLP->Draw();
    //            TLine *event4_MLP = new TLine(MLPev4,cMLP->GetUymin(),MLPev4,(cMLP->GetUymax()/2.5));
    //            event4_MLP->SetLineWidth(2);
    //            if (goldensilver==1) event4_MLP->SetLineColor(kYellow+1);
    //            else event4_MLP->SetLineColor(kGreen+3);
    //            event4_MLP->Draw();
    //            TLine *event5_MLP = new TLine(MLPev5,cMLP->GetUymin(),MLPev5,(cMLP->GetUymax()/2.5));
    //            event5_MLP->SetLineWidth(2);
    //            if (goldensilver==1) event5_MLP->SetLineColor(kYellow+1);
    //            else event5_MLP->SetLineColor(kGreen+4);
    //            event5_MLP->Draw();
    //            TLine *eventBER_MLP = new TLine(MLPevBER,cMLP->GetUymin(),MLPevBER,(cMLP->GetUymax()/2.5));
    //            eventBER_MLP->SetLineWidth(2);
    //            if (goldensilver==1) eventBER_MLP->SetLineColor(kGray+1);
    //            else eventBER_MLP->SetLineColor(kYellow+1);
    //            eventBER_MLP->Draw();
    //            TLine *eventPDBO_MLP = new TLine(MLPevPDBO,cMLP->GetUymin(),MLPevPDBO,(cMLP->GetUymax()/2.5));
    //            eventPDBO_MLP->SetLineWidth(2);
    //            if (goldensilver==1) eventPDBO_MLP->SetLineColor(kGray+1);
    //            else eventPDBO_MLP->SetLineColor(kOrange+1);
    //            eventPDBO_MLP->Draw();
    //            //        TLine *eventNAG1 = new TLine(MLPevNAG1,cMLP->GetUymin(),MLPevNAG1,(cMLP->GetUymax()/2.5));
    //            //        eventNAG1->SetLineWidth(2);
    //            //        eventNAG1->SetLineColor(kBlue-4);
    //            //        eventNAG1->Draw();
    //            TLine *eventNAG2_MLP = new TLine(MLPevNAG2,cMLP->GetUymin(),MLPevNAG2,(cMLP->GetUymax()/2.5));
    //            eventNAG2_MLP->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG2_MLP->SetLineColor(kGray+1);
    //            else eventNAG2_MLP->SetLineColor(kRed);
    //            eventNAG2_MLP->Draw();
    //
    //        }
    //        if (channel==3) {
    //            TLine *event2_MLP = new TLine(MLPev2,cMLP->GetUymin(),MLPev2,(cMLP->GetUymax()/2.5));
    //            event2_MLP->SetLineWidth(2);
    //            if (goldensilver==1) event2_MLP->SetLineColor(kYellow+1);
    //            else event2_MLP->SetLineColor(kGreen+2);
    //            event2_MLP->Draw();
    //            TLine *eventBARI_MLP = new TLine(MLPevBARI,cMLP->GetUymin(),MLPevBARI,(cMLP->GetUymax()/2.5));
    //            eventBARI_MLP->SetLineWidth(2);
    //            if (goldensilver==1) eventBARI_MLP->SetLineColor(kGray+1);
    //            else eventBARI_MLP->SetLineColor(kMagenta+1);
    //            eventBARI_MLP->Draw();
    //            //        TLine *eventNAG3 = new TLine(MLPevNAG3,cMLP->GetUymin(),MLPevNAG3,(cMLP->GetUymax()/2.5));
    //            //        eventNAG3->SetLineWidth(2);
    //            //        eventNAG3->SetLineColor(kYellow+1);
    //            //        eventNAG3->Draw();
    //            TLine *eventNAG4_MLP = new TLine(MLPevNAG4,cMLP->GetUymin(),MLPevNAG4,(cMLP->GetUymax()/2.5));
    //            eventNAG4_MLP->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG4_MLP->SetLineColor(kGray+1);
    //            else eventNAG4_MLP->SetLineColor(kRed-3);
    //            eventNAG4_MLP->Draw();
    //
    //            //        cout << "BDT response: MLPev2 " << MLPev2 << "\tMLPevBARI " << MLPevBARI << "\tMLPevNAG4 " << MLPevNAG4 << endl;
    //
    //        }
    //        if (channel==2) {
    //            TLine *event3_MLP = new TLine(MLPev3,cMLP->GetUymin(),MLPev3,(cMLP->GetUymax()/2.5));
    //            event3_MLP->SetLineWidth(2);
    //            if (goldensilver==1) event3_MLP->SetLineColor(kYellow+1);
    //            else event3_MLP->SetLineColor(kGreen+2);
    //            event3_MLP->Draw();
    //
    //            //cout << "BDT response: MLPev3 " << MLPev3 << endl;
    //        }
    //    }
    //
    //
    //    if (Use["MLPBFGS"]) {
    //
    //        cMLPBFGS->cd();
    //
    //
    //        TLine *cutline_MLPBFGS = new TLine(suggestedcut_MLPBFGS,cMLPBFGS->GetUymin(),suggestedcut_MLPBFGS,cMLPBFGS->GetUymax());
    //        cutline_MLPBFGS->SetLineWidth(2);
    //        cutline_MLPBFGS->SetLineColor(kBlue-8);
    //        if(channel!=4)cutline_MLPBFGS->Draw();
    //
    //        TArrow *freccia_MLPBFGS= new TArrow(suggestedcut_MLPBFGS, (cMLPBFGS->GetUymax()/2), suggestedcut_MLPBFGS+0.07, (cMLPBFGS->GetUymax()/2), 0.009, ">");
    //        freccia_MLPBFGS->SetLineColor(kBlue-8);
    //        freccia_MLPBFGS->SetLineWidth(2);
    //        if(channel!=4)freccia_MLPBFGS->Draw();
    //
    //        if (channel==1) {
    //            TLine *event1_MLPBFGS = new TLine(MLPBFGSev1,cMLPBFGS->GetUymin(),MLPBFGSev1,(cMLPBFGS->GetUymax()/2.5));
    //            event1_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) event1_MLPBFGS->SetLineColor(kYellow+1);
    //            else event1_MLPBFGS->SetLineColor(kGreen+2);
    //            event1_MLPBFGS->Draw();
    //            TLine *event4_MLPBFGS = new TLine(MLPBFGSev4,cMLPBFGS->GetUymin(),MLPBFGSev4,(cMLPBFGS->GetUymax()/2.5));
    //            event4_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) event4_MLPBFGS->SetLineColor(kYellow+1);
    //            else event4_MLPBFGS->SetLineColor(kGreen+3);
    //            event4_MLPBFGS->Draw();
    //            TLine *event5_MLPBFGS = new TLine(MLPBFGSev5,cMLPBFGS->GetUymin(),MLPBFGSev5,(cMLPBFGS->GetUymax()/2.5));
    //            event5_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) event5_MLPBFGS->SetLineColor(kYellow+1);
    //            else event5_MLPBFGS->SetLineColor(kGreen+4);
    //            event5_MLPBFGS->Draw();
    //            TLine *eventBER_MLPBFGS = new TLine(MLPBFGSevBER,cMLPBFGS->GetUymin(),MLPBFGSevBER,(cMLPBFGS->GetUymax()/2.5));
    //            eventBER_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) eventBER_MLPBFGS->SetLineColor(kGray+1);
    //            else eventBER_MLPBFGS->SetLineColor(kYellow+1);
    //            eventBER_MLPBFGS->Draw();
    //            TLine *eventPDBO_MLPBFGS = new TLine(MLPBFGSevPDBO,cMLPBFGS->GetUymin(),MLPBFGSevPDBO,(cMLPBFGS->GetUymax()/2.5));
    //            eventPDBO_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) eventPDBO_MLPBFGS->SetLineColor(kGray+1);
    //            else eventPDBO_MLPBFGS->SetLineColor(kOrange+1);
    //            eventPDBO_MLPBFGS->Draw();
    //            //        TLine *eventNAG1 = new TLine(MLPBFGSevNAG1,cMLPBFGS->GetUymin(),MLPBFGSevNAG1,(cMLPBFGS->GetUymax()/2.5));
    //            //        eventNAG1->SetLineWidth(2);
    //            //        eventNAG1->SetLineColor(kBlue-4);
    //            //        eventNAG1->Draw();
    //            TLine *eventNAG2_MLPBFGS = new TLine(MLPBFGSevNAG2,cMLPBFGS->GetUymin(),MLPBFGSevNAG2,(cMLPBFGS->GetUymax()/2.5));
    //            eventNAG2_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG2_MLPBFGS->SetLineColor(kGray+1);
    //            else eventNAG2_MLPBFGS->SetLineColor(kRed);
    //            eventNAG2_MLPBFGS->Draw();
    //
    //        }
    //        if (channel==3) {
    //            TLine *event2_MLPBFGS = new TLine(MLPBFGSev2,cMLPBFGS->GetUymin(),MLPBFGSev2,(cMLPBFGS->GetUymax()/2.5));
    //            event2_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) event2_MLPBFGS->SetLineColor(kYellow+1);
    //            else event2_MLPBFGS->SetLineColor(kGreen+2);
    //            event2_MLPBFGS->Draw();
    //            TLine *eventBARI_MLPBFGS = new TLine(MLPBFGSevBARI,cMLPBFGS->GetUymin(),MLPBFGSevBARI,(cMLPBFGS->GetUymax()/2.5));
    //            eventBARI_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) eventBARI_MLPBFGS->SetLineColor(kGray+1);
    //            else eventBARI_MLPBFGS->SetLineColor(kMagenta+1);
    //            eventBARI_MLPBFGS->Draw();
    //            //        TLine *eventNAG3 = new TLine(MLPBFGSevNAG3,cMLPBFGS->GetUymin(),MLPBFGSevNAG3,(cMLPBFGS->GetUymax()/2.5));
    //            //        eventNAG3->SetLineWidth(2);
    //            //        eventNAG3->SetLineColor(kYellow+1);
    //            //        eventNAG3->Draw();
    //            TLine *eventNAG4_MLPBFGS = new TLine(MLPBFGSevNAG4,cMLPBFGS->GetUymin(),MLPBFGSevNAG4,(cMLPBFGS->GetUymax()/2.5));
    //            eventNAG4_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG4_MLPBFGS->SetLineColor(kGray+1);
    //            else eventNAG4_MLPBFGS->SetLineColor(kRed-3);
    //            eventNAG4_MLPBFGS->Draw();
    //
    //            //        cout << "BDT response: MLPBFGSev2 " << MLPBFGSev2 << "\tMLPBFGSevBARI " << MLPBFGSevBARI << "\tMLPBFGSevNAG4 " << MLPBFGSevNAG4 << endl;
    //
    //        }
    //        if (channel==2) {
    //            TLine *event3_MLPBFGS = new TLine(MLPBFGSev3,cMLPBFGS->GetUymin(),MLPBFGSev3,(cMLPBFGS->GetUymax()/2.5));
    //            event3_MLPBFGS->SetLineWidth(2);
    //            if (goldensilver==1) event3_MLPBFGS->SetLineColor(kYellow+1);
    //            else event3_MLPBFGS->SetLineColor(kGreen+2);
    //            event3_MLPBFGS->Draw();
    //
    //            //cout << "BDT response: MLPBFGSev3 " << MLPBFGSev3 << endl;
    //        }
    //    }
    //
    //    if (Use["TMlpANN"]){
    //        cTMlpANN->cd();
    //
    //        TLine *cutline_TMlpANN = new TLine(suggestedcut_TMlpANN,cTMlpANN->GetUymin(),suggestedcut_TMlpANN,cTMlpANN->GetUymax());
    //        cutline_TMlpANN->SetLineWidth(2);
    //        cutline_TMlpANN->SetLineColor(kBlue-8);
    //        if(channel!=4)cutline_TMlpANN->Draw();
    //
    //        TArrow *freccia_TMlpANN= new TArrow(suggestedcut_TMlpANN, (cTMlpANN->GetUymax()/2), suggestedcut_TMlpANN+0.07, (cTMlpANN->GetUymax()/2), 0.009, ">");
    //        freccia_TMlpANN->SetLineColor(kBlue-8);
    //        freccia_TMlpANN->SetLineWidth(2);
    //        if(channel!=4)freccia_TMlpANN->Draw();
    //
    //        //    /*TLine *event = new TLine(bdtev,cTMlpANN->GetUymin(),bdtev,(cTMlpANN->GetUymax()/2));
    //        //     event->SetLineWidth(2);
    //        //     event->SetLineColor(kGreen+2);
    //        //     event->Draw();
    //        //     */
    //
    //        if (channel==1) {
    //            TLine *event1_TMlpANN = new TLine(TMlpANNev1,cTMlpANN->GetUymin(),TMlpANNev1,(cTMlpANN->GetUymax()/2.5));
    //            event1_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) event1_TMlpANN->SetLineColor(kYellow+1);
    //            else event1_TMlpANN->SetLineColor(kGreen+2);
    //            event1_TMlpANN->Draw();
    //            TLine *event4_TMlpANN = new TLine(TMlpANNev4,cTMlpANN->GetUymin(),TMlpANNev4,(cTMlpANN->GetUymax()/2.5));
    //            event4_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) event4_TMlpANN->SetLineColor(kYellow+1);
    //            else event4_TMlpANN->SetLineColor(kGreen+3);
    //            event4_TMlpANN->Draw();
    //            TLine *event5_TMlpANN = new TLine(TMlpANNev5,cTMlpANN->GetUymin(),TMlpANNev5,(cTMlpANN->GetUymax()/2.5));
    //            event5_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) event5_TMlpANN->SetLineColor(kYellow+1);
    //            else event5_TMlpANN->SetLineColor(kGreen+4);
    //            event5_TMlpANN->Draw();
    //            TLine *eventBER_TMlpANN = new TLine(TMlpANNevBER,cTMlpANN->GetUymin(),TMlpANNevBER,(cTMlpANN->GetUymax()/2.5));
    //            eventBER_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) eventBER_TMlpANN->SetLineColor(kGray+1);
    //            else eventBER_TMlpANN->SetLineColor(kYellow+1);
    //            eventBER_TMlpANN->Draw();
    //            TLine *eventPDBO_TMlpANN = new TLine(TMlpANNevPDBO,cTMlpANN->GetUymin(),TMlpANNevPDBO,(cTMlpANN->GetUymax()/2.5));
    //            eventPDBO_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) eventPDBO_TMlpANN->SetLineColor(kGray+1);
    //            else eventPDBO_TMlpANN->SetLineColor(kOrange+1);
    //            eventPDBO_TMlpANN->Draw();
    //            //        TLine *eventNAG1 = new TLine(TMlpANNevNAG1,cTMlpANN->GetUymin(),TMlpANNevNAG1,(cTMlpANN->GetUymax()/2.5));
    //            //        eventNAG1->SetLineWidth(2);
    //            //        eventNAG1->SetLineColor(kBlue-4);
    //            //        eventNAG1->Draw();
    //            TLine *eventNAG2_TMlpANN = new TLine(TMlpANNevNAG2,cTMlpANN->GetUymin(),TMlpANNevNAG2,(cTMlpANN->GetUymax()/2.5));
    //            eventNAG2_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG2_TMlpANN->SetLineColor(kGray+1);
    //            else eventNAG2_TMlpANN->SetLineColor(kRed);
    //            eventNAG2_TMlpANN->Draw();
    //
    //        }
    //        if (channel==3) {
    //            TLine *event2_TMlpANN = new TLine(TMlpANNev2,cTMlpANN->GetUymin(),TMlpANNev2,(cTMlpANN->GetUymax()/2.5));
    //            event2_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) event2_TMlpANN->SetLineColor(kYellow+1);
    //            else event2_TMlpANN->SetLineColor(kGreen+2);
    //            event2_TMlpANN->Draw();
    //            TLine *eventBARI_TMlpANN = new TLine(TMlpANNevBARI,cTMlpANN->GetUymin(),TMlpANNevBARI,(cTMlpANN->GetUymax()/2.5));
    //            eventBARI_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) eventBARI_TMlpANN->SetLineColor(kGray+1);
    //            else eventBARI_TMlpANN->SetLineColor(kMagenta+1);
    //            eventBARI_TMlpANN->Draw();
    //            //        TLine *eventNAG3 = new TLine(TMlpANNevNAG3,cTMlpANN->GetUymin(),TMlpANNevNAG3,(cTMlpANN->GetUymax()/2.5));
    //            //        eventNAG3->SetLineWidth(2);
    //            //        eventNAG3->SetLineColor(kYellow+1);
    //            //        eventNAG3->Draw();
    //            TLine *eventNAG4_TMlpANN = new TLine(TMlpANNevNAG4,cTMlpANN->GetUymin(),TMlpANNevNAG4,(cTMlpANN->GetUymax()/2.5));
    //            eventNAG4_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) eventNAG4_TMlpANN->SetLineColor(kGray+1);
    //            else eventNAG4_TMlpANN->SetLineColor(kRed-3);
    //            eventNAG4_TMlpANN->Draw();
    //
    //            //        cout << "BDT response: TMlpANNev2 " << TMlpANNev2 << "\tTMlpANNevBARI " << TMlpANNevBARI << "\tTMlpANNevNAG4 " << TMlpANNevNAG4 << endl;
    //
    //        }
    //        if (channel==2) {
    //            TLine *event3_TMlpANN = new TLine(TMlpANNev3,cTMlpANN->GetUymin(),TMlpANNev3,(cTMlpANN->GetUymax()/2.5));
    //            event3_TMlpANN->SetLineWidth(2);
    //            if (goldensilver==1) event3_TMlpANN->SetLineColor(kYellow+1);
    //            else event3_TMlpANN->SetLineColor(kGreen+2);
    //            event3_TMlpANN->Draw();
    //
    //            //cout << "BDT response: TMlpANNev3 " << TMlpANNev3 << endl;
    //        }
    //
    
    
    
    
    //SALVO PLOT
    char outputplot[50];
    
    sprintf (outputplot,"./plot/BDTplotweighted_%d.pdf", channel);
    c->SaveAs(outputplot);
    
    //    if (Use["MLP"]){
    //        sprintf (outputplot,"./plot/MLPplotweighted_%d.pdf", channel);
    //        cMLP->SaveAs(outputplot);
    //    }
    //
    //    if (Use["MLPBFGS"]){
    //        sprintf (outputplot,"./plot/MLPBFGSplotweighted_%d.pdf", channel);
    //        cMLPBFGS->SaveAs(outputplot);
    //    }
    //
    //    if (Use["TMlpANN"]){
    //        sprintf (outputplot,"./plot/TMlpANNplotweighted_%d.pdf", channel);
    //        cTMlpANN->SaveAs(outputplot);
    //    }
    
    sprintf (outputplot,"./plot/KinVar_%d.pdf", channel);
    c1->SaveAs(outputplot);
    
    sprintf (outputplot,"./plot/EffPur_%d.pdf", channel);
    c2->SaveAs(outputplot);
    
    //    sprintf (outputplot,"./plot/EffperPur_%d.pdf", channel);
    //    c3->SaveAs(outputplot);
    
    
    
    delete factory;
    delete dataloader;
    // Launch the GUI for the root macros
    if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );
    
    return 0;
}

int main( int argc, char** argv )
{
    // Select methods (don't look at this code - not of interest)
    TString methodList;
    for (int i=1; i<argc; i++) {
        TString regMethod(argv[i]);
        if(regMethod=="-b" || regMethod=="--batch") continue;
        if (!methodList.IsNull()) methodList += TString(",");
        methodList += regMethod;
    }
    return TMVAClassification(methodList);
}
