#include "intt_standalone_cluster.h"

#include <fun4all/Fun4AllReturnCodes.h>

#include <ffaobjects/EventHeader.h>
#include <phool/getClass.h>
#include <phool/PHCompositeNode.h>
#include <phool/PHNodeIterator.h>
#include <phool/recoConsts.h>

#include <boost/format.hpp>
#include <boost/math/special_functions/sign.hpp>

int nEvents_INTT = 0;
int nChips_INTT = 0;

// Reference: https://github.com/sPHENIX-Collaboration/analysis/blob/master/TPC/DAQ/macros/prelimEvtDisplay/TPCEventDisplay.C
//____________________________________________________________________________..
intt_standalone_cluster::intt_standalone_cluster(const std::string &name):
 SubsysReco(name)
{
}

//____________________________________________________________________________..
intt_standalone_cluster::~intt_standalone_cluster()
{
}

//____________________________________________________________________________..
int intt_standalone_cluster::Init(PHCompositeNode *topNode)
{
  outFile = new TFile(outFileName.c_str(), "RECREATE");
  outTree = new TTree("Hits", "Hits");
  //outTree = new TTree("Clusters", "Clusters");
  outTree->OptimizeBaskets();
  outTree->SetAutoSave(-5e6);

  outTree->Branch("event", &event, "event/I");
  outTree->Branch("bco_full", &bco_full);
  outTree->Branch("layer", &layer, "layer/I");
  outTree->Branch("ladderzid", &ladderzid, "ladderzid/I");
  outTree->Branch("ladderphiid", &ladderphiid, "ladderphiid/I");
  outTree->Branch("timebucketid", &timebucketid, "timebucketid/I");
  outTree->Branch("nhit_bco", &nhit_bco, "nhit_bco/I");
  outTree->Branch("nhit", &nhit, "nhit/I");
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

  return Fun4AllReturnCodes::EVENT_OK;
}
//____________________________________________________________________________..
int intt_standalone_cluster::process_event(PHCompositeNode *topNode)
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

  geantGeom = findNode::getClass<PHG4CylinderGeomContainer>(topNode, "CYLINDERGEOM_INTT");
  if (!geantGeom)
  {
    std::cout << __FILE__ << "::" << __func__ << " - CYLINDERGEOM_INTT missing, doing nothing." << std::endl;
    exit(1);
  }

  //intt_event_header = findNode::getClass<InttEventInfov1>(topNode, "INTTEVENTHEADER");
  //if (!intt_event_header)
  //{
  //  std::cout << __FILE__ << "::" << __func__ << " - INTTEVENTHEADER missing, doing nothing." << std::endl;
  //  exit(1);
  //}

  inttcont = findNode::getClass<InttRawHitContainerv2>(topNode, "INTTRAWHIT");
  if (!inttcont)
  {
    std::cout << __FILE__ << "::" << __func__ << " - INTTRAWHIT missing, doing nothing." << std::endl;
    exit(1);
  }

  nhit_bco = inttcont->get_nhits();

  //all hits for this bco is the same, because it has been syncronized
  //for (int ihit=0; ihit<nhit_bco; ihit++)
  //{
  //  InttRawHit* inttrawhit = inttcont->get_hit(ihit);
  //  bco_full = inttrawhit->get_bco();
  //  std::cout<<"hit "<<ihit<<" , bco_full = "<<bco_full<<std::endl;
  //}

  //get the bco of the first hit as the representation of all the hits
  if (nhit_bco>0)
  {
    InttRawHit* inttrawhit = inttcont->get_hit(0);
    bco_full = inttrawhit->get_bco();
  }
  else
  {
    bco_full = -1;
  }

  //bco_full = intt_event_header->get_bco_full();
  event = f4aCounter;
  nhit = 0;
  layer = 0;
  ladderzid = 0;
  ladderphiid = 0;
  timebucketid = 0;
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

  ++nEvents_INTT;

  //Set up the event display writer
  //std::ofstream outFile;
  //bool firstHits = true;
  //float minX = 0., minY = 0., minZ = 0., maxX = 0., maxY = 0., maxZ = 0.;
  //if (m_write_evt_display)// && trktClusterContainer->size() >= m_min_clusters)
  //{
  //  outFile.open(m_evt_display_path + "/EvtDisplay_" + m_runNumber + "_" + L1_BCOs[0] + ".json");
  //  event_file_start(outFile, m_run_date, m_runNumber, L1_BCOs[0]); 
  //}

  TrkrHitSetContainer::ConstRange hitsetrange = trkrHitSetContainer->getHitSets(TrkrDefs::TrkrId::inttId);

  for (TrkrHitSetContainer::ConstIterator hitsetitr = hitsetrange.first; hitsetitr != hitsetrange.second; ++hitsetitr)
  {
    TrkrHitSet *hitset = hitsetitr->second;
    auto hitsetkey = hitset->getHitSetKey();

    layer = TrkrDefs::getLayer(hitsetkey);
    ladderzid = InttDefs::getLadderZId(hitsetkey);
    ladderphiid = InttDefs::getLadderPhiId(hitsetkey);
    timebucketid = InttDefs::getTimeBucketId(hitsetkey);
//std::cout<<"layer = "<<layer<<" , ladderzid = "<<ladderzid<<" , ladderphiid = "<<ladderphiid<<" , timebucketid = "<<timebucketid<<std::endl;

    //TrkrClusterContainer::ConstRange clusterrange = trktClusterContainer->getClusters(hitsetkey);

    auto surface = actsGeom->maps().getSiliconSurface(hitsetkey);
    auto layergeom = dynamic_cast<CylinderGeomIntt *>(geantGeom->GetLayerGeom(layer));

    TrkrHitSet::ConstRange hit_range = hitsetitr->second->getHits();
    for (TrkrHitSet::ConstIterator hit_iter = hit_range.first; hit_iter != hit_range.second; ++hit_iter)
    {
      //TrkrCluster *cluster = clusteritr->second;
      TrkrDefs::hitkey hitKey = hit_iter->first;

      int rowNow = InttDefs::getRow(hitKey);
      int colNow = InttDefs::getCol(hitKey);

      row.push_back(rowNow);
      col.push_back(colNow);
//std::cout<<"row = "<<rowNow<<" , col = "<<colNow<<std::endl;

      ++nChips_INTT;
      double local_hit_loc[3] = {0, 0, 0};
      layergeom->find_strip_center_localcoords(ladderzid, rowNow, colNow, local_hit_loc); // cm, sPHENIX unit
      TVector2 local;
      local.SetX(local_hit_loc[1]);
      local.SetY(local_hit_loc[2]);
std::cout<<"local = ("<<local.X()<<","<<local.Y()<<")"<<std::endl;

      localX.push_back(local.X());
      localY.push_back(local.Y());
      //localX.push_back(cluster->getLocalX());
      //localY.push_back(cluster->getLocalY());
      //clusZ.push_back(cluster->getZSize());
      //clusPhi.push_back(cluster->getPhiSize());
      //clusSize.push_back(cluster->getAdc());

      TVector3 glob = layergeom->get_world_from_local_coords(surface, actsGeom, local); // cm, sPHENIX unit
std::cout<<"glob = ("<<glob.x()<<","<<glob.y()<<","<<glob.z()<<")"<<std::endl;
      globalX.push_back(glob.X());
      globalY.push_back(glob.Y());
      globalZ.push_back(glob.Z());

//center of the surface corresponding to that thitsetkey
//Acts::Vector3 center = surface->center(actsGeom->geometry().getGeoContext()); // mm, 10x cm
//std::cout<<"Joe's method, center = ("<<center.x()<<","<<center.y()<<","<<center.z()<<")"<<std::endl;

//double world_coords[3];
//layergeom->find_segment_center(surface, actsGeom, world_coords); // cm, sPHENIX unit
//std::cout<<"Use find_sensor_center, center = ("<<world_coords[0]<<","<<world_coords[1]<<","<<world_coords[2]<<")"<<std::endl;

    }
    nhit = row.size();

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

  //if (outFile)
  //{
  //  event_file_trailer(outFile, minX, minY, minZ, maxX, maxY, maxZ);
  //  outFile.close();
  //}

  ++f4aCounter;

  return Fun4AllReturnCodes::EVENT_OK;
}


//____________________________________________________________________________..
int intt_standalone_cluster::End(PHCompositeNode *topNode)
{
  outFile->Write();
  outFile->Close();
  delete outFile;


std::cout << "INTT hit analyser end" << std::endl;
std::cout << "Number of events analysed: " << nEvents_INTT << std::endl;
std::cout << "Number of chips with hits: " << nChips_INTT << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int intt_standalone_cluster::Reset(PHCompositeNode *topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
void intt_standalone_cluster::Print(const std::string &what) const
{
}


void intt_standalone_cluster::event_file_start(std::ofstream &jason_file_header, std::string date, int runid, int bco)
{
  jason_file_header << "{" << std::endl;
  jason_file_header << "    \"EVENT\": {" << std::endl;
  jason_file_header << "        \"runid\":" << runid << ", " << std::endl;
  jason_file_header << "        \"evtid\": 1, " << std::endl;
  jason_file_header << "        \"time\": 0, " << std::endl;
  jason_file_header << "       \"type\": \"Collision\", " << std::endl;
  jason_file_header << "        \"s_nn\": 0, " << std::endl;
  jason_file_header << "        \"B\": 3.0," << std::endl;
  jason_file_header << "        \"pv\": [0,0,0]," << std::endl;
  jason_file_header << " \"runstats\": [\"sPHENIX Internal\"," << std::endl;
  jason_file_header << "\"Cosmic\"," << std::endl;
  jason_file_header << "\"" << date << ", Run " << runid << "\"," << std::endl;
  jason_file_header << "\"BCO:" << bco <<"\"]  " << std::endl;
  jason_file_header << "    }," << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << "    \"META\": {" << std::endl;
  jason_file_header << "       \"HITS\": {" << std::endl;
  jason_file_header << "          \"INNERTRACKER\": {" << std::endl;
  jason_file_header << "              \"type\": \"3D\"," << std::endl;
  jason_file_header << "              \"options\": {" << std::endl;
  jason_file_header << "              \"size\": 5," << std::endl;
  jason_file_header << "              \"color\": 16777215" << std::endl;
  jason_file_header << "              } " << std::endl;
  jason_file_header << "          }," << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << "          \"TRACKHITS\": {" << std::endl;
  jason_file_header << "              \"type\": \"3D\"," << std::endl;
  jason_file_header << "              \"options\": {" << std::endl;
  jason_file_header << "              \"size\": 0.5," << std::endl;
  jason_file_header << "              \"transparent\": 0.5," << std::endl;
  jason_file_header << "              \"color\": 16073282" << std::endl;
  jason_file_header << "              } " << std::endl;
  jason_file_header << "          }," << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << "    \"JETS\": {" << std::endl;
  jason_file_header << "        \"type\": \"JET\"," << std::endl;
  jason_file_header << "        \"options\": {" << std::endl;
  jason_file_header << "            \"rmin\": 0," << std::endl;
  jason_file_header << "            \"rmax\": 78," << std::endl;
  jason_file_header << "            \"emin\": 0," << std::endl;
  jason_file_header << "            \"emax\": 30," << std::endl;
  jason_file_header << "            \"color\": 16777215," << std::endl;
  jason_file_header << "            \"transparent\": 0.5 " << std::endl;
  jason_file_header << "        }" << std::endl;
  jason_file_header << "    }" << std::endl;
  jason_file_header << "        }" << std::endl;
  jason_file_header << "    }" << std::endl;
  jason_file_header << "," << std::endl;
  jason_file_header << "    \"HITS\": {" << std::endl;
  jason_file_header << "        \"CEMC\":[{\"eta\": 0, \"phi\": 0, \"e\": 0}" << std::endl;
  jason_file_header << "            ]," << std::endl;
  jason_file_header << "        \"HCALIN\": [{\"eta\": 0, \"phi\": 0, \"e\": 0}" << std::endl;
  jason_file_header << "            ]," << std::endl;
  jason_file_header << "        \"HCALOUT\": [{\"eta\": 0, \"phi\": 0, \"e\": 0}" << std::endl;
  jason_file_header << " " << std::endl;
  jason_file_header << "            ]," << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << "    \"TRACKHITS\": [" << std::endl;
  jason_file_header << "" << std::endl;
  jason_file_header << " ";
}

void intt_standalone_cluster::event_file_trailer(std::ofstream &json_file_trailer, float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
  float deltaX = minX - maxX;
  float deltaY = minY - maxY;
  float deltaZ = minZ - maxZ;
  float length = sqrt(pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2));

  json_file_trailer << "]," << std::endl;
  json_file_trailer << "    \"JETS\": [" << std::endl;
  json_file_trailer << "         ]" << std::endl;
  json_file_trailer << "    }," << std::endl;
  json_file_trailer << "\"TRACKS\": {" << std::endl;
  json_file_trailer << "   \"B\": 0.000014," << std::endl;
  json_file_trailer << "   \"TRACKHITS\": [" << std::endl;
  json_file_trailer << "     {" << std::endl;
  json_file_trailer << "       \"color\": 16777215," << std::endl;
  json_file_trailer << "       \"l\": " << length << "," << std::endl;
  json_file_trailer << "       \"nh\": 6," << std::endl;
  json_file_trailer << "       \"pxyz\": [" << std::endl;
  json_file_trailer << "         " << deltaX << "," << std::endl;
  json_file_trailer << "        " << deltaY << "," << std::endl;
  json_file_trailer << "         " << deltaZ << std::endl;
  json_file_trailer << "       ]," << std::endl;
  json_file_trailer << "       \"q\": 1," << std::endl;
  json_file_trailer << "       \"xyz\": [" << std::endl;
  json_file_trailer << "         " << maxX << ", " << std::endl;
  json_file_trailer << "          " << maxY << ", " << std::endl;
  json_file_trailer << "         " << maxZ << std::endl;
  json_file_trailer << "       ]" << std::endl;
  json_file_trailer << "     }" << std::endl;
  json_file_trailer << "   ]" << std::endl;
  json_file_trailer << "}" << std::endl;
  json_file_trailer << "}" << std::endl;
}

