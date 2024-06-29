// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <fun4all/SubsysReco.h>

#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TVector2.h>
#include <TVector3.h>
#include <TObject.h>
#include <TString.h>

#include <mvtx/CylinderGeom_Mvtx.h>
#include <intt/CylinderGeomIntt.h>
#include <g4detectors/PHG4CylinderGeomContainer.h>

#include <trackbase/MvtxDefs.h>
#include <trackbase/TrkrHitSetContainerv1.h>
#include <trackbase/TrkrHitv2.h>
#include <trackbase/TrkrHitSet.h>
#include <trackbase/TrkrClusterContainerv4.h>
#include <trackbase/TrkrClusterHitAssocv3.h>
#include <trackbase/TrkrClusterv4.h>
#include <trackbase/ActsGeometry.h>

class PHCompositeNode;

class Alignment : public SubsysReco
{
 public:

  Alignment(const std::string &name = "Alignment");

  ~Alignment() override;

  /** Called during initialization.
      Typically this is where you can book histograms, and e.g.
      register them to Fun4AllServer (so they can be output to file
      using Fun4AllServer::dumpHistos() method).
   */
  int Init(PHCompositeNode *topNode) override;

  /** Called for first event when run number is known.
      Typically this is where you may want to fetch data from
      database, because you know the run number. A place
      to book histograms which have to know the run number.
   */
  //int InitRun(PHCompositeNode *topNode) override;

  /** Called for each event.
      This is where you do the real work.
   */
  int process_event(PHCompositeNode *topNode) override;

  /// Clean up internals after each event.
  //int ResetEvent(PHCompositeNode *topNode) override;

  /// Called at the end of each run.
  //int EndRun(const int runnumber) override;

  /// Called at the end of all processing.
  int End(PHCompositeNode *topNode) override;

  /// Reset
  int Reset(PHCompositeNode * /*topNode*/) override;

  void Print(const std::string &what = "ALL") const override;

  // alignment
  void SetEpoch(int epoch) {m_epoch = epoch};
  int  GetEpoch() {return m_epoch;}

  void SetStep(int step) {m_step = step;};
  int  GetStep() {return m_step;}

  void SetNData(int n) {m_ndata = n;}
  int  GetNData() {return m_ndata}

  void SetCore(int i) {m_core = i;}
  int GetCore() {return m_core;}

  void SetLoadNetworkUpdateList(bool userdefined) {m_load_network_update_list = userdefined};
  bool GetLoadNetworkUpdateList() {return m_load_network_update_list;}

  void SetSourceDataName(TString name) {m_source_data_name = name;}
  TString GetSourceDataName() {return m_source_data_name;}

  void SetSourceTreeName(TString name) {m_source_tree_name = name};
  TString GetSourceTreeName() {return m_source_tree_name;}

 private:
  TrkrHitSetContainerv1 *trkrHitSetContainer = nullptr;
  TrkrClusterContainer *trktClusterContainer = nullptr;
  ActsGeometry *actsGeom = nullptr;
  PHG4CylinderGeomContainer *geantGeom_mvtx;
  PHG4CylinderGeomContainer *geantGeom_intt;

  //alignment
  int m_epoch;
  int m_step;
  int m_ndata;
  int m_core;
  bool m_load_network_update_list;
  TString m_source_data_name;
  TString m_source_tree_name;
};

#endif // ALIGNMENT_H
