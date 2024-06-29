#include "Alignment.h"

#include <fun4all/Fun4AllReturnCodes.h>

#include <ffaobjects/EventHeader.h>
#include <phool/getClass.h>
#include <phool/PHCompositeNode.h>
#include <phool/PHNodeIterator.h>
#include <phool/recoConsts.h>

#include <boost/format.hpp>
#include <boost/math/special_functions/sign.hpp>

#include "YAlignment.cxx"
#include "YMLPParallel.h"

struct sEventIndex {
   int event;
   int ntracks;
};

// Reference: https://github.com/sPHENIX-Collaboration/analysis/blob/master/TPC/DAQ/macros/prelimEvtDisplay/TPCEventDisplay.C
//____________________________________________________________________________..
Alignment::Alignment(const std::string &name):
 SubsysReco(name)
{
}

//____________________________________________________________________________..
Alignment::~Alignment()
{
}

//____________________________________________________________________________..
int Alignment::Init(PHCompositeNode *topNode)
{
  PHNodeIterator dstiter(topNode);

  recoConsts *rc = recoConsts::instance();
  m_runNumber = rc->get_IntFlag("RUNNUMBER");

  PHCompositeNode* dstNode = dynamic_cast<PHCompositeNode *>(dstiter.findFirst("PHCompositeNode", "DST"));
  if (!dstNode)
  {
    std::cout << __FILE__ << "::" << __func__ << " - DST Node missing, doing nothing." << std::endl;
    exit(1);
  }

  trkrHitSetContainer = findNode::getClass<TrkrHitSetContainerv1>(dstNode, "TRKR_HITSET");
  if (!trkrHitSetContainer)
  {
    std::cout << __FILE__ << "::" << __func__ << " - TRKR_HITSET  missing, doing nothing." << std::endl;
    exit(1);
  }

  actsGeom = findNode::getClass<ActsGeometry>(topNode, "ActsGeometry");
  if (!actsGeom)
  {
    std::cout << __FILE__ << "::" << __func__ << " - ActsGeometry missing, doing nothing." << std::endl;
    exit(1);
  }

  geantGeom_mvtx = findNode::getClass<PHG4CylinderGeomContainer>(topNode, "CYLINDERGEOM_MVTX");
  if (!geantGeom_mvtx)
  {
    std::cout << __FILE__ << "::" << __func__ << " - CYLINDERGEOM_MVTX missing, doing nothing." << std::endl;
    exit(1);
  }

  geantGeom_intt = findNode::getClass<PHG4CylinderGeomContainer>(topNode, "CYLINDERGEOM_INTT");
  if (!geantGeom_intt)
  {
    std::cout << __FILE__ << "::" << __func__ << " - CYLINDERGEOM_INTT missing, doing nothing." << std::endl;
    exit(1);
  }

  YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kBatch; //kSteepestDescent kStochastic kBatch
 
  YAlignment* yalign = new YAlignment();
  
  yalign->LoadNetworkUpdateList(m_load_network_update_list);
  //yalign->SetNetworkUpdateListLayerStave(NUlayer, NUstave);
  yalign->SetSourceDataName(m_source_data_name);
  yalign->SetSourceTreeName(m_source_tree_name); //mlp_Input_S
  
  yalign->SetEpoch(m_epoch);
  yalign->SetStep(m_step); //step

  yalign->SetActsGeom(actsGeom);
  yalign->SetGeantGeomMVTX(geantGeom_mvtx);
  yalign->SetGeantGeomINTT(geantGeom_intt);

  vector<int> fhiddenlayer;
  fhiddenlayer.push_back(0); 
  yalign->SetHiddenLayer(fhiddenlayer);
  yalign->PrepareData(m_ndata,m_core);
  
  std::cout<<"Start TrainMLP !"<<std::endl;
  if(m_step==0){
     TString prevWeightSet = "";
     yalign->TrainMLP(method);
  }else{
     yalign->SetPrevUSL(Form("./MLPTrain_Step%d/UpdateSensorsList.txt",m_step-1)); 
     yalign->SetPrevWeight(Form("./MLPTrain_Step%d/weights/weights.txt",m_step-1)); 
     yalign->SetPrevWeightDU(Form("./MLPTrain_Step%d/weights/weightsDU.txt",m_step-1)); 
     yalign->TrainMLP(method);
  }
  gSystem->Exec(Form("mv UpdateSensorsList.txt MLPTrain/UpdateSensorsList.txt"));

  gSystem->Exec(Form("mkdir MLPTrain/Res_Monitor"));
  
  for(int l = 0; l < nLAYER; l++){
     gSystem->Exec(Form("mv ResMean_Layer%d_Epoch*.gif MLPTrain/Res_Monitor/",l));      
  }
  
  gSystem->Exec(Form("mkdir MLPTrain/Residual"));
  
  gSystem->Exec(Form("mv Residual_Epoch_At_*.root MLPTrain/Residual/"));
  gSystem->Exec(Form("mv Residual_Monitor_Epoch_At_*.root MLPTrain/Residual/"));
  gSystem->Exec(Form("mkdir MLPTrain/Track_Monitor"));
  
  gSystem->Exec(Form("mv Tracks*Layer*.gif MLPTrain/Track_Monitor/"));

  if(nEPOCH>0) {
     gSystem->Exec(Form("mkdir MLPTrain/Cost_Gradient"));

     gSystem->Exec(Form("mv CostGradient_Epoch_At_*.txt MLPTrain/Cost_Gradient/"));

     gSystem->Exec(Form("tar -zxvf TrendingNetwork.tgz"));

     gSystem->Exec(Form("cp o2sim_geometry.root TrendingNetwork/o2sim_geometry.root"));
     gSystem->Exec(Form("cp MLPTrain/weights/* TrendingNetwork/weights/"));
     gSystem->Exec(Form("cp MLPTrain/Cost_Gradient/* TrendingNetwork/Cost_Gradient/"));
  
     gSystem->Exec(Form("echo \"#NTracks by Sensor\" > NTracksBySensor_Epoch-1.txt"));
     gSystem->Exec(Form("sed 's/Chip//g' MLPTrain/Cost_Gradient/CostGradient_Epoch_At_0.txt | awk '{print $3\" \"$4}' >> NTracksBySensor_Epoch-1.txt"));
  
     gSystem->Exec(Form("mv NTracksBySensor_Epoch-1.txt TrendingNetwork/NTracks_Profile/NTracksBySensor_Epoch-1.txt"));
  
     //gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b run_macros.C'(-1,%d)';cd ..",nEPOCH-1)); 
     gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b checkCostGradients.C'(-1,%d)';cd ..",nEPOCH-1)); 
     gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b checkWeightGradients_extended.C'(-1,%d)';cd ..",nEPOCH-1)); 
     
     gSystem->Exec(Form("rm TrendingNetwork/o2sim_geometry.root"));
     gSystem->Exec(Form("rm TrendingNetwork/weights/*"));
     gSystem->Exec(Form("rm TrendingNetwork/Cost_Gradient/*"));

  }
  gSystem->Exec(Form("mv TrendingNetwork MLPTrain/"));
  //
  gSystem->Exec(Form("mv MLPTrain MLPTrain_Step%d",m_step));

  return Fun4AllReturnCodes::EVENT_OK;
}
//____________________________________________________________________________..
int Alignment::process_event(PHCompositeNode *topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}


//____________________________________________________________________________..
int Alignment::End(PHCompositeNode *topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int Alignment::Reset(PHCompositeNode *topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
void Alignment::Print(const std::string &what) const
{
}

