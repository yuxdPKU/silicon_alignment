#include "mvtx_standalone_cluster.h"

#include <fun4all/Fun4AllReturnCodes.h>

#include <ffaobjects/EventHeader.h>
#include <phool/getClass.h>
#include <phool/PHCompositeNode.h>
#include <phool/PHNodeIterator.h>
#include <phool/recoConsts.h>

#include <boost/format.hpp>
#include <boost/math/special_functions/sign.hpp>

// Reference: https://github.com/sPHENIX-Collaboration/analysis/blob/master/TPC/DAQ/macros/prelimEvtDisplay/TPCEventDisplay.C
//____________________________________________________________________________..
mvtx_standalone_cluster::mvtx_standalone_cluster(const std::string &name):
 SubsysReco(name)
{
}

//____________________________________________________________________________..
mvtx_standalone_cluster::~mvtx_standalone_cluster()
{
}

//____________________________________________________________________________..
int mvtx_standalone_cluster::Init(PHCompositeNode *topNode)
{
  outFile = new TFile(outFileName.c_str(), "RECREATE");
  outTree = new TTree("Hits", "Hits");
  //outTree = new TTree("Clusters", "Clusters");
  outTree->OptimizeBaskets();
  outTree->SetAutoSave(-5e6);

  outTree->Branch("event", &event, "event/I");
  outTree->Branch("strobe_BCOs", &strobe_BCOs);
  outTree->Branch("L1_BCOs", &L1_BCOs);
  outTree->Branch("numberL1s", &numberL1s, "numberL1s/I");
  outTree->Branch("layer", &layer, "layer/I");
  outTree->Branch("stave", &stave, "stave/I");
  outTree->Branch("chip", &chip, "chip/I");
  outTree->Branch("row", &row);
  outTree->Branch("col", &col);
  outTree->Branch("localX", &localX);
  outTree->Branch("localY", &localY);
  outTree->Branch("globalX", &globalX);
  outTree->Branch("globalY", &globalY);
  outTree->Branch("globalZ", &globalZ);
  //outTree->Branch("clusZSize", &clusZ);
  //outTree->Branch("clusPhiSize", &clusPhi);
  //outTree->Branch("clusSize", &clusSize);
  outTree->Branch("chip_occupancy", &chip_occupancy, "chip_occupancy/F");
  outTree->Branch("chip_hits", &chip_hits, "chip_hits/I");

  return Fun4AllReturnCodes::EVENT_OK;
}
//____________________________________________________________________________..
int mvtx_standalone_cluster::process_event(PHCompositeNode *topNode)
{
  bool occAbove0p3 = false;

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

  //trktClusterContainer = findNode::getClass<TrkrClusterContainer>(dstNode, "TRKR_CLUSTER");
  //if (!trktClusterContainer)
  //{
  //  std::cout << __FILE__ << "::" << __func__ << " - TRKR_CLUSTER missing, doing nothing." << std::endl;
  //  exit(1);
  //}

  actsGeom = findNode::getClass<ActsGeometry>(topNode, "ActsGeometry");
  if (!actsGeom)
  {
    std::cout << __FILE__ << "::" << __func__ << " - ActsGeometry missing, doing nothing." << std::endl;
    exit(1);
  }

  geantGeom = findNode::getClass<PHG4CylinderGeomContainer>(topNode, "CYLINDERGEOM_MVTX");
  if (!geantGeom)
  {
    std::cout << __FILE__ << "::" << __func__ << " - CYLINDERGEOM_MVTX missing, doing nothing." << std::endl;
    exit(1);
  }

  mvtx_event_header = findNode::getClass<MvtxEventInfov2>(topNode, "MVTXEVENTHEADER");
  if (!mvtx_event_header)
  {
    std::cout << __FILE__ << "::" << __func__ << " - MVTXEVENTHEADER missing, doing nothing." << std::endl;
    exit(1);
  }

  std::set<uint64_t> strobeList = mvtx_event_header->get_strobe_BCOs();
  for (auto iterStrobe = strobeList.begin(); iterStrobe != strobeList.end(); ++iterStrobe)
  {
    strobe_BCOs.push_back(*iterStrobe); 
    std::set<uint64_t> l1List = mvtx_event_header->get_L1_BCO_from_strobe_BCO(*iterStrobe);
    for (auto iterL1 = l1List.begin(); iterL1 != l1List.end(); ++iterL1)
    {
      L1_BCOs.push_back(*iterL1); 
    }
  }

  event = f4aCounter;
  numberL1s = mvtx_event_header->get_number_L1s();
  layer = 0;
  stave = 0;
  chip = 0;
  row.clear();
  col.clear();
  localX.clear();
  localY.clear();
  globalX.clear();
  globalY.clear();
  globalZ.clear();
  clusZ.clear();
  clusPhi.clear();
  clusSize.clear();
  chip_occupancy = 0.;
  chip_hits = 0.;

  //Set up the event display writer
  std::ofstream outFile;
  bool firstHits = true;
  float minX = 0., minY = 0., minZ = 0., maxX = 0., maxY = 0., maxZ = 0.;
  if (m_write_evt_display)// && trktClusterContainer->size() >= m_min_clusters)
  {
    outFile.open(m_evt_display_path + "/EvtDisplay_" + m_runNumber + "_" + L1_BCOs[0] + ".json");
    event_file_start(outFile, m_run_date, m_runNumber, L1_BCOs[0]); 
  }

  TrkrHitSetContainer::ConstRange hitsetrange = trkrHitSetContainer->getHitSets(TrkrDefs::TrkrId::mvtxId);

  for (TrkrHitSetContainer::ConstIterator hitsetitr = hitsetrange.first; hitsetitr != hitsetrange.second; ++hitsetitr)
  {
    TrkrHitSet *hitset = hitsetitr->second;
    auto hitsetkey = hitset->getHitSetKey();

    layer = TrkrDefs::getLayer(hitsetkey);
    stave = MvtxDefs::getStaveId(hitsetkey);
    chip = MvtxDefs::getChipId(hitsetkey);

    //TrkrClusterContainer::ConstRange clusterrange = trktClusterContainer->getClusters(hitsetkey);

    auto surface = actsGeom->maps().getSiliconSurface(hitsetkey);
    auto layergeom = dynamic_cast<CylinderGeom_Mvtx *>(geantGeom->GetLayerGeom(layer));
    TVector2 LocalUse;

    ++nChips;
    chip_hits = hitsetitr->second->size();
    chip_occupancy = (float) chip_hits / (512*1024);
    if (chip_occupancy*100 >= -1.) //Take all clusters
    {
      ++nChipsWOccGreaterThan0p3;
      if (!occAbove0p3)
      {
        occAbove0p3 = true;
        ++nEventsWOccGreaterThan0p3;
        if (numberL1s != 0) ++nEventsWTriggerAndOccGreaterThan0p3;
      }
      //for (TrkrClusterContainer::ConstIterator clusteritr = clusterrange.first; clusteritr != clusterrange.second; ++clusteritr)
      //{
      TrkrHitSet::ConstRange hit_range = hitsetitr->second->getHits();
      for (TrkrHitSet::ConstIterator hit_iter = hit_range.first; hit_iter != hit_range.second; ++hit_iter)
      {
        //TrkrCluster *cluster = clusteritr->second;
        TrkrDefs::hitkey hitKey = hit_iter->first;

        row.push_back(MvtxDefs::getRow(hitKey));
        col.push_back(MvtxDefs::getCol(hitKey));
std::cout<<"layer = "<<layer<<" , stave = "<<stave<<" , chip = "<<chip<<std::endl;
std::cout<<"row = "<<MvtxDefs::getRow(hitKey)<<" , col = "<<MvtxDefs::getCol(hitKey)<<std::endl;
        TVector3 local_coords = layergeom->get_local_coords_from_pixel(MvtxDefs::getRow(hitKey), MvtxDefs::getCol(hitKey)); // cm, sPHENIX unit
std::cout<<"local_coords = ("<<local_coords.x()<<","<<local_coords.y()<<","<<local_coords.z()<<")"<<std::endl;
        localX.push_back(local_coords.x());
        localY.push_back(local_coords.z());
        //localX.push_back(cluster->getLocalX());
        //localY.push_back(cluster->getLocalY());
        //clusZ.push_back(cluster->getZSize());
        //clusPhi.push_back(cluster->getPhiSize());
        //clusSize.push_back(cluster->getAdc());

        LocalUse.SetX(local_coords.x());
        LocalUse.SetY(local_coords.z());
        //LocalUse.SetX(cluster->getLocalX());
        //LocalUse.SetY(cluster->getLocalY());
        TVector3 ClusterWorld = layergeom->get_world_from_local_coords(surface, actsGeom, LocalUse); // cm, sPHENIX unit
std::cout<<"global_coords = ("<<ClusterWorld.x()<<","<<ClusterWorld.y()<<","<<ClusterWorld.z()<<")"<<std::endl;
        globalX.push_back(ClusterWorld.X());
        globalY.push_back(ClusterWorld.Y());
        globalZ.push_back(ClusterWorld.Z());

//center of the surface corresponding to that thitsetkey
Acts::Vector3 center = surface->center(actsGeom->geometry().getGeoContext()); // mm, 10x cm
std::cout<<"Joe's method, center = ("<<center.x()<<","<<center.y()<<","<<center.z()<<")"<<std::endl;

double world_coords[3];
layergeom->find_sensor_center(surface, actsGeom, world_coords); // cm, sPHENIX unit
std::cout<<"Use find_sensor_center, center = ("<<world_coords[0]<<","<<world_coords[1]<<","<<world_coords[2]<<")"<<std::endl;

        if (outFile)
        {
          std::ostringstream spts;
          if (firstHits)
          {
            firstHits = false;
            minX = maxX = ClusterWorld.X();
            minY = maxY = ClusterWorld.Y();
            minZ = maxZ = ClusterWorld.Z();
          }
          else
          {
            spts << ",";
            if (ClusterWorld.Y() < minY)
            {
              minX = ClusterWorld.X();
              minY = ClusterWorld.Y();
              minZ = ClusterWorld.Z();
            }
            if (ClusterWorld.Y() > maxY)
            {
              maxX = ClusterWorld.X();
              maxY = ClusterWorld.Y();
              maxZ = ClusterWorld.Z();
            }
          }

          spts << "{ \"x\": ";
          spts << ClusterWorld.X();
          spts << ", \"y\": ";
          spts << ClusterWorld.Y();
          spts << ", \"z\": ";
          spts << ClusterWorld.Z();
          spts << ", \"e\": 0}";

          outFile << (boost::format("%1%") % spts.str());
          spts.clear();
          spts.str("");
        }
      }

      outTree->Fill();

      row.clear();
      col.clear();
      localX.clear();
      localY.clear();
      globalX.clear();
      globalY.clear();
      globalZ.clear();
      clusZ.clear();
      clusPhi.clear();
      clusSize.clear();
    }
  }

  if (outFile)
  {
    event_file_trailer(outFile, minX, minY, minZ, maxX, maxY, maxZ);
    outFile.close();
  }

  strobe_BCOs.clear();
  L1_BCOs.clear();
  ++f4aCounter;

  return Fun4AllReturnCodes::EVENT_OK;
}


//____________________________________________________________________________..
int mvtx_standalone_cluster::End(PHCompositeNode *topNode)
{
  outFile->Write();
  outFile->Close();
  delete outFile;

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int mvtx_standalone_cluster::Reset(PHCompositeNode *topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
void mvtx_standalone_cluster::Print(const std::string &what) const
{
}

