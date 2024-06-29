// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_YDetectorGeometry
#define ROOT_YDetectorGeometry

#include "TObject.h"
#include "TString.h"
#include <vector>

#if !defined(__CLING__) || defined(__ROOTCLING__)
//#define ENABLE_UPGRADES
#include <TGeoMatrix.h> // for TGeoHMatrix
#include "DetectorsCommonDataFormats/DetMatrixCache.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/CcdbApi.h"
#include "ITSBase/GeometryTGeo.h"
#include <TRandom.h>
#include <TFile.h>
#include <vector>
#include <fmt/format.h>
#endif
//____________________________________________________________________
//
// YDetectorGeometry
//
// Description
//
//--------------------------------------------------------------------
class YDetectorGeometry;

namespace GEOM {
namespace Internal {
   class YDetectorGeometryAllocator;

   YDetectorGeometry *GetGEOM2();

} 
} // End ROOT::Internal


class YDetectorGeometry {

friend YDetectorGeometry *GEOM::Internal::GetGEOM2();

private:
   YDetectorGeometry(const YDetectorGeometry&);                   			//Not implemented
   YDetectorGeometry& operator=(const YDetectorGeometry&);        			//Not implemented

protected:
   YDetectorGeometry(const char *name, const char *title);              				  
   friend class ::GEOM::Internal::YDetectorGeometryAllocator;

public:
   YDetectorGeometry();
   virtual ~YDetectorGeometry() {};  
      
   //o2::its::GeometryTGeo* GetGeom() { return geom;}

   TVector3 GToS(int chipID, double gx, double gy, double gz);
   TVector3 GToL(int chipID, double gx, double gy, double gz);
   TVector3 SToL(int chipID, double s1, double s2, double s3);
   TVector3 SToG(int chipID, double s1, double s2, double s3);
   TVector3 LToG(int chipID, float lx, float ly);
   TVector3 LToS(int chipID, float lx, float ly); //not implemented LToG -> GToS
   TVector3 NormalVector(int chipID);				// Stave plane 
   TVector3 NormalVector_Deformation(int chipID, double ds1_0, double ds2_0, double *ds3);   			
   TVector3 Function_Deformation(int chipID, float lx, float ly, double ds1_0, double ds2_0, double *ds3);

   //void load_matrix(TString files="geom_sPHENIX.txt");
   //const o2::math_utils::Transform3D& getMatrixL2G(int sensID) const { return mL2G.getMatrix(sensID); };

   int GetLastChipIndex(int lay) const { return mLastChipIndex[lay]; }
   int GetFirstChipIndex(int lay) const { return (lay == 0) ? 0 : mLastChipIndex[lay - 1] + 1; }
  
   int GetLayer(int index) const;    /// Get chip layer, from 0
   int GetHalfBarrel(int index) const;   /// Get chip half barrel, from 0
   int GetStave(int index) const;   /// Get chip stave, from 0
   int GetHalfStave(int index) const;   /// Get chip substave id in stave, from 0
   int GetModule(int index) const;   /// Get chip module id in substave, from 0
   int GetChipIdInLayer(int index) const;   /// Get chip number within layer, from 0
   int GetChipIdInStave(int index) const;   /// Get chip number within stave, from 0
   int GetChipIdInHalfStave(int index) const;   /// Get chip number within stave, from 0
   int GetChipIdInModule(int index) const;   /// Get chip number within module, from 0
   
   bool GetChipId(int index, int& lay, int& hba, int& sta, int& mod, int& chip) const;
   int GetChipIndex(int lay, int sta, int chipInStave) const;
     
private:
   //o2::its::GeometryTGeo* geom;

public:
   static constexpr int NCols = 1024;
   static constexpr int NRows = 512;
   static constexpr int NPixels = NRows * NCols;
   static constexpr float PitchCol = 29.24e-4;
   static constexpr float PitchRow = 26.88e-4;
   static constexpr float PassiveEdgeReadOut = 0.12f;              // width of the readout edge (Passive bottom)
   static constexpr float PassiveEdgeTop = 37.44e-4;               // Passive area on top
   static constexpr float PassiveEdgeSide = 29.12e-4;              // width of Passive area on left/right of the sensor
   static constexpr float ActiveMatrixSizeCols = PitchCol * NCols; // Active size along columns
   static constexpr float ActiveMatrixSizeRows = PitchRow * NRows; // Active size along rows

   // effective thickness of sensitive layer, accounting for charge collection non-unifoemity, https://alice.its.cern.ch/jira/browse/AOC-46
   static constexpr float SensorLayerThicknessEff = 28.e-4;
   static constexpr float SensorLayerThickness = 30.e-4;                                               // physical thickness of sensitive part
   static constexpr float SensorSizeCols = ActiveMatrixSizeCols + PassiveEdgeSide + PassiveEdgeSide;   // SensorSize along columns
   static constexpr float SensorSizeRows = ActiveMatrixSizeRows + PassiveEdgeTop + PassiveEdgeReadOut; // SensorSize along rows

   static constexpr int MAXLAYERS = 15; ///< max number of active layers

   Int_t mNumberOfLayers;                        ///< number of layers
   Int_t mNumberOfHalfBarrels;                   ///< number of halfbarrels
   std::vector<int> mNumberOfStaves;             ///< number of staves/layer(layer)
   std::vector<int> mNumberOfHalfStaves;         ///< the number of substaves/stave(layer)
   std::vector<int> mNumberOfModules;            ///< number of modules/substave(layer)
   std::vector<int> mNumberOfChipsPerModule;     ///< number of chips per module (group of chips on substaves)
   std::vector<int> mNumberOfChipRowsPerModule;  ///< number of chips rows per module (relevant for OB modules)
   std::vector<int> mNumberOfChipsPerHalfStave;  ///< number of chips per substave
   std::vector<int> mNumberOfChipsPerStave;      ///< number of chips per stave
   std::vector<int> mNumberOfChipsPerHalfBarrel; ///< number of chips per halfbarrel
   std::vector<int> mNumberOfChipsPerLayer;      ///< number of chips per stave
   std::vector<int> mLastChipIndex;              ///< max ID of the detctor in the layer

   
   //mvtx
   double unit_cm = 10.0; //default value defined in mm unit
   
   double loc_sensor_in_chip_mvtx[3] = {0.058128, -0.0005, 0.0};  // mvtx_stave_v1.gdml

   //detector information

   static constexpr int NLayer = 7;   //layer number in ITS detector
   static constexpr int NLayerIB = 3;

   static constexpr int NSubStave2[NLayer] = { 1, 1, 1, 1, 1, 1, 1 };
   const int NSubStave[NLayer] = { 1, 1, 1, 1, 1, 1, 1 };
   const int NStaves[NLayer] = { 12, 16, 20, 12, 12, 16, 16 };
   const int nHicPerStave[NLayer] = { 1, 1, 1, 1, 1, 1, 1 };
   const int nChipsPerHic[NLayer] = { 9, 9, 9, 4, 4, 4, 4 };
   const int ChipBoundary[NLayer + 1] = { 0, 108, 252, 432, 480, 528, 592, 656 };
   const int StaveBoundary[NLayer + 1] = { 0, 12, 28, 48, 60, 72, 88, 104 };

   //o2::detectors::MatrixCache<o2::math_utils::Transform3D> mL2G;    ///< Local to Global matrices

   ClassDef(YDetectorGeometry,0)  							//Top level (or geom) structure for all classes   
};

namespace GEOM {
   YDetectorGeometry *GetGEOM();
   namespace Internal {
   
      R__EXTERN YDetectorGeometry *yGEOMLocal;
   }
}

#define yGEOM (GEOM::GetGEOM())

#endif
