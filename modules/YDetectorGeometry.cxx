// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//
// YDetectorGeometry
//
//
///////////////////////////////////////////////////////////////////////////

#include "YDetectorGeometry.h"

#include "SegmentationAlpide.h"

#define PIX_PRECISION 1e-2

static void at_exit_of_YDetectorGeometry() {
   if (GEOM::Internal::yGEOMLocal)
      GEOM::Internal::yGEOMLocal->~YDetectorGeometry();
}

// This local static object initializes the GEOM system
namespace GEOM {
namespace Internal {
   class YDetectorGeometryAllocator {

      char fHolder[sizeof(YDetectorGeometry)];
   public:
      YDetectorGeometryAllocator() {
         new(&(fHolder[0])) YDetectorGeometry("geom","The GEOM of EVERYTHING");
      }

      ~YDetectorGeometryAllocator() {
         if (yGEOMLocal) {
            yGEOMLocal->~YDetectorGeometry();
         }
      }
   };

   extern YDetectorGeometry *yGEOMLocal;

   YDetectorGeometry *GetGEOM1() {
      if (yGEOMLocal)
         return yGEOMLocal;
      static YDetectorGeometryAllocator alloc;
      return yGEOMLocal;
   }

   YDetectorGeometry *GetGEOM2() {
      static Bool_t initInterpreter = kFALSE;
      if (!initInterpreter) {
         initInterpreter = kTRUE;
      }
      return yGEOMLocal;
   }
   typedef YDetectorGeometry *(*GetGEOMFun_t)();

   static GetGEOMFun_t yGetGEOM = &GetGEOM1;


} // end of Internal sub namespace
// back to GEOM namespace

   YDetectorGeometry *GetGEOM() {
      return (*Internal::yGetGEOM)();
   }
}

YDetectorGeometry *GEOM::Internal::yGEOMLocal = GEOM::GetGEOM();

ClassImp(YDetectorGeometry);

////////////////////////////////////////////////////////////////////////////////
/// Default Constructor YSensorSet

YDetectorGeometry::YDetectorGeometry(const char *name, const char *title) 
{
   std::cout<<"Default Constructor YDetectorGeometry "<<std::endl;
   //These are supposed to be defined as constant header.

   //mL2G.setSize(656);
   load_matrix("geom_sPHENIX.txt");
   //o2::base::GeometryManager::loadGeometry("", false, false);
/*
   TFile file("ITSAlignment.root");
   std::vector<o2::detectors::AlignParam>* aliPars;
   file.GetObject("ccdb_object", aliPars);
   //file.GetObject("alignment", aliPars);   
   o2::base::GeometryManager::applyAlignment(*aliPars);
   std::cout<<" LOAD ITSaligment constants"<< std::endl;    
   for(int ich = 0; ich < (*aliPars).size(); ich ++) (*aliPars)[ich].print();
      
   //delete aliPars;   
*/   
   //o2::detectors::DetID detITS("ITS");
   //geom = o2::its::GeometryTGeo::Instance();

   //geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::L2G));


   //Int_t mNumberOfLayers;                        ///< number of layers
   mNumberOfLayers = 7;
   //Int_t mNumberOfHalfBarrels;                   ///< number of halfbarrels
   mNumberOfHalfBarrels = 2;
   
   //std::vector<int> mNumberOfStaves;             ///< number of staves/layer(layer)
   mNumberOfStaves.push_back(12);
   mNumberOfStaves.push_back(16);
   mNumberOfStaves.push_back(20);
   mNumberOfStaves.push_back(12);
   mNumberOfStaves.push_back(12);
   mNumberOfStaves.push_back(16);
   mNumberOfStaves.push_back(16);
   
   //std::vector<int> mNumberOfHalfStaves;         ///< the number of substaves/stave(layer)
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);
   mNumberOfHalfStaves.push_back(1);

   //std::vector<int> mNumberOfModules;            ///< number of modules/substave(layer)
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   mNumberOfModules.push_back(1);
   
   //std::vector<int> mNumberOfChipsPerModule;     ///< number of chips per module (group of chips on substaves)
   mNumberOfChipsPerModule.push_back(9);
   mNumberOfChipsPerModule.push_back(9);
   mNumberOfChipsPerModule.push_back(9);
   mNumberOfChipsPerModule.push_back(4);
   mNumberOfChipsPerModule.push_back(4);
   mNumberOfChipsPerModule.push_back(4);
   mNumberOfChipsPerModule.push_back(4);
   
   //std::vector<int> mNumberOfChipRowsPerModule;  ///< number of chips rows per module (relevant for OB modules)
   //std::vector<int> mNumberOfChipsPerHalfStave;  ///< number of chips per substave
   mNumberOfChipsPerHalfStave.push_back(9);
   mNumberOfChipsPerHalfStave.push_back(9);
   mNumberOfChipsPerHalfStave.push_back(9);
   mNumberOfChipsPerHalfStave.push_back(4);
   mNumberOfChipsPerHalfStave.push_back(4);
   mNumberOfChipsPerHalfStave.push_back(4);
   mNumberOfChipsPerHalfStave.push_back(4);   
      
   //std::vector<int> mNumberOfChipsPerStave;      ///< number of chips per stave
   mNumberOfChipsPerStave.push_back(9);
   mNumberOfChipsPerStave.push_back(9);
   mNumberOfChipsPerStave.push_back(9);
   mNumberOfChipsPerStave.push_back(4);
   mNumberOfChipsPerStave.push_back(4);
   mNumberOfChipsPerStave.push_back(4);
   mNumberOfChipsPerStave.push_back(4);   
   
   //std::vector<int> mNumberOfChipsPerHalfBarrel; ///< number of chips per halfbarrel
   mNumberOfChipsPerHalfBarrel.push_back(9*6);
   mNumberOfChipsPerHalfBarrel.push_back(9*8);
   mNumberOfChipsPerHalfBarrel.push_back(9*10);
   mNumberOfChipsPerHalfBarrel.push_back(4*6);
   mNumberOfChipsPerHalfBarrel.push_back(4*6);
   mNumberOfChipsPerHalfBarrel.push_back(4*8);
   mNumberOfChipsPerHalfBarrel.push_back(4*8);      
      
   //std::vector<int> mNumberOfChipsPerLayer;      ///< number of chips per layer
   mNumberOfChipsPerLayer.push_back(9*12);
   mNumberOfChipsPerLayer.push_back(9*16);
   mNumberOfChipsPerLayer.push_back(9*20);
   mNumberOfChipsPerLayer.push_back(4*12);
   mNumberOfChipsPerLayer.push_back(4*12);
   mNumberOfChipsPerLayer.push_back(4*16);
   mNumberOfChipsPerLayer.push_back(4*16);  
    
   //std::vector<int> mLastChipIndex;              ///< max ID of the detctor in the layer
   mLastChipIndex.push_back(107);
   mLastChipIndex.push_back(251);   
   mLastChipIndex.push_back(431);   
   mLastChipIndex.push_back(479);   
   mLastChipIndex.push_back(527);   
   mLastChipIndex.push_back(591);   
   mLastChipIndex.push_back(655);      
   
   GEOM::Internal::yGEOMLocal = this;
   GEOM::Internal::yGetGEOM = &GEOM::Internal::GetGEOM2;   
}


void YDetectorGeometry::load_matrix(TString files="geom_sPHENIX.txt")
{

   FILE *fInput_in;
   fInput_in = fopen(files,"read");
   int size_tvalue = 2000;
   char tvalue[size_tvalue];
   for(int a = 0; a < size_tvalue; a++){
      tvalue[a] = '\0';
   }
  
   std::cout<<"File Load : "<<files<<std::endl;

   while(fgets(tvalue,sizeof(tvalue),fInput_in) != NULL){
      TString fTarget = tvalue; 
      fTarget = TString(fTarget(0,fTarget.Length()-1));  
      fTarget += " ";

      TString  value[20];  

      int beg = 0;
      int end = fTarget.Index(" ", beg + 1);
      int cnt = 0;
      while (end != -1) {       
         value[cnt] = TString(fTarget(beg,end-beg));    
         value[cnt].ReplaceAll(" ","");
         beg = end + 1;
         end = fTarget.Index(" ", beg + 1);
         cnt++;
      }   

      if(value[0]=="sensorID") continue;
      else {

         int sensorID = value[0].TString::Atoi();      
         //o2::math_utils::Transform3D mat;
         //sensorID layer stave chip Tx Ty Tz Rxx Rxy Rxz Ryx Ryy Ryz Rzx Rzy Rzz 
         double Tx = (double) std::stod((const char*)value[4]);
         double Ty = (double) std::stod((const char*)value[5]);
         double Tz = (double) std::stod((const char*)value[6]);
 
         double matT[3] = {Tx, Ty, Tz};
          
         double Rxx = (double) std::stod((const char*)value[7]);
         double Rxy = (double) std::stod((const char*)value[8]);
         double Rxz = (double) std::stod((const char*)value[9]);
         double Ryx = (double) std::stod((const char*)value[10]);
         double Ryy = (double) std::stod((const char*)value[11]);
         double Ryz = (double) std::stod((const char*)value[12]);
         double Rzx = (double) std::stod((const char*)value[13]);
         double Rzy = (double) std::stod((const char*)value[14]);
         double Rzz = (double) std::stod((const char*)value[15]);         
/*
         std::cout<<sensorID<<" "<<Tx <<" "<<Ty <<" "<<Tz <<" "
                                 <<Rxx<<" "<<Rxy<<" "<<Rxz<<" "
                                 <<Ryx<<" "<<Ryy<<" "<<Ryz<<" "
                                 <<Rzx<<" "<<Rzy<<" "<<Rzz<<std::endl;
*/
         double matR[9] = {Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz};  
         TGeoHMatrix ghmat(Form("Chip%d",sensorID));
         ghmat.SetTranslation(matT);
         ghmat.SetRotation(matR);
         
         //mat.set(ghmat);
         //mL2G.setMatrix(mat, sensorID);
         mL2G[sensorID] = ghmat;

      }
    
   }
   fclose(fInput_in);
}

////////////////////////////////////////////////////////////////////////////////
/// GToS // mvtx ok + intt ok

//global(x,y,z) -> local (x,y) 
TVector3 YDetectorGeometry::GToS(int chipID, double gx, double gy, double gz) 
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;     
   }
   
   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){
      //o2::math_utils::Point3D<float> gloC(gx*unit_cm, gy*unit_cm, gz*unit_cm);
      //auto locC = getMatrixL2G(chipID) ^ gloC; // convert global coordinates to local.
      float gloC[3] = {gx*unit_cm, gy*unit_cm, gz*unit_cm};
      float locC[3];
      mL2G[chipID].MasterToLocal(gloC, locC);
      //std::cout<<"[GToS MVTX STEP 0] ChipID "<<chipID<<" gloC "<<gx<<" "<<gy<<" "<<gz<<" locC "<<locC.X()<<" "<<locC.Y()<<" "<<locC.Z()<<std::endl;   
      float l1= locC.X()/unit_cm; //xrow s2
      float l2= locC.Y()/unit_cm; //zcol s1
      float l3= locC.Z()/unit_cm; //s3
      //std::cout<<"[GToS MVTX STEP 1] ChipID "<<chipID<<" gloC "<<gx<<" "<<gy<<" "<<gz<<" locC "<<l1<<" "<<l2<<" "<<l3<<std::endl;
   
      TVector3 v3(-l2,-l1,l3);   
      return v3;
   } else {
      //o2::math_utils::Point3D<float> gloC(gx*unit_cm, gy*unit_cm, gz*unit_cm);
      //auto locC = getMatrixL2G(chipID) ^ gloC; // convert global coordinates to local.
      float gloC[3] = {gx*unit_cm, gy*unit_cm, gz*unit_cm};
      float locC[3];
      mL2G[chipID].MasterToLocal(gloC, locC);
      //std::cout<<"[GToS INTT STEP 0] ChipID "<<chipID<<" gloC "<<gx<<" "<<gy<<" "<<gz<<" locC "<<locC.X()<<" "<<locC.Y()<<" "<<locC.Z()<<std::endl;   
      float l1= locC.X()/unit_cm; //xrow s2
      float l2= locC.Y()/unit_cm; //zcol s1
      float l3= locC.Z()/unit_cm; //s3
      //std::cout<<"[GToS INTT STEP 1] ChipID "<<chipID<<" gloC "<<gx<<" "<<gy<<" "<<gz<<" locC "<<l1<<" "<<l2<<" "<<l3<<std::endl;
      
      TVector3 v3(-l2,-l1,l3);   
      return v3;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// GToL // mvtx ok + intt ok

//global(x,y,z) -> pixel (row, col) 
TVector3 YDetectorGeometry::GToL(int chipID, double gx, double gy, double gz) 
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;     
   }
 
   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){   
      TVector3 gtos = GToS(chipID, gx, gy, gz);

      float l1= -gtos(1) + loc_sensor_in_chip_mvtx[0];  //row
      float l2= +gtos(2) + loc_sensor_in_chip_mvtx[1];  //norm
      float l3= -gtos(0) + loc_sensor_in_chip_mvtx[2];  //col
      //TVector3 v3(l3,l1,l2);
      float frow, fcol;
      //o2::itsmft::SegmentationAlpide::localToDetector(l1, l3 ,row, col);
      //
      // convert to row/col w/o over/underflow check
      l1 = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - l1; // coordinate wrt top edge of Active matrix
      l3 += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
      frow = float(l1 / PitchRow) - 0.5;
      fcol = float(l3 / PitchCol) - 0.5;
      if (l1 < 0) {
        frow -= 1;
      }
      if (l3 < 0) {
        fcol -= 1;
      }

      TVector3 v3(std::round(frow/PIX_PRECISION)*PIX_PRECISION,std::round(fcol/PIX_PRECISION)*PIX_PRECISION,0);
      return v3;
   } else {
      //implementation : void CylinderGeomIntt::find_strip_index_values(const int segment_z_bin, const double yin, const double zin, int &strip_y_index, int &strip_z_index)

      //int m_Layer;
      int m_NStripsPhiCell = 256;
      int m_NStripsZSensor[2];
      //double m_StripX = 0.032;
      double m_StripY = 0.0078; 
      //double m_SensorRadius;
      //double m_StripXOffset;
      //double m_OffsetPhi;
      //double m_OffsetRot;
      double m_dPhi;      //m_dPhi = 2. * TMath::Pi() / nladders_layer;
      m_dPhi = 2. * TMath::Pi() / mNumberOfStaves[Layer];

      double m_StripZ[2];
      //double m_LadderZ[2];

      // Type-A
      m_StripZ[0] = 1.6;//strip_z0;
      //m_LadderZ[0] = ladder_z0;
      m_NStripsZSensor[0] = 8;//nstrips_z_sensor0;

      // Type-B
      m_StripZ[1] = 2.0;//strip_z1;
      //m_LadderZ[1] = ladder_z1;
      m_NStripsZSensor[1] = 5;//nstrips_z_sensor1;

/*
      CylinderGeomIntt *mygeom = new CylinderGeomIntt(
          sphxlayer,					     		const int layer,			m_Layer(layer)
          params->get_double_param("strip_x"),		     		const double strip_x,			m_StripX(strip_x)	
          params->get_double_param("strip_y"),		     		const double strip_y,			m_StripY(strip_y)
          params->get_double_param("strip_z_0"),			const double strip_z0,	
          params->get_double_param("strip_z_1"),			const double strip_z1,
          params->get_int_param("nstrips_z_sensor_0"),			const int nstrips_z_sensor0,
          params->get_int_param("nstrips_z_sensor_1"),			const int nstrips_z_sensor1,
          params->get_int_param("nstrips_phi_sensor"),			const int nstrips_phi_cell,		m_NStripsPhiCell(nstrips_phi_cell)
          params_layer->get_int_param("nladder"),			const int nladders_layer,
          m_PosZ[ilayer][0] / cm,  					const double ladder_z0,
          m_PosZ[ilayer][1] / cm,					const double ladder_z1,
          m_SensorRadius[ilayer] / cm,					const double sensor_radius,		m_SensorRadius(sensor_radius)
          0.0,								const double strip_x_offset,		m_StripXOffset(strip_x_offset)
          params_layer->get_double_param("offsetphi") * deg / rad,  	const double offsetphi,			m_OffsetPhi(offsetphi)
          params_layer->get_double_param("offsetrot") * deg / rad);	const double offsetrot)			m_OffsetRot(offsetrot)
*/      
      
      int layer(-1), hb(-1), stv(-1), md(-1), mchip(-1);
      GetChipId(chipID, layer, hb, stv, md, mchip);
      int segment_z_bin = mchip;
      
      TVector3 gtos = GToS(chipID, gx, gy, gz);

      float l1= -gtos(1);  //row
      float l2= +gtos(2);  //norm
      float l3= -gtos(0);  //col
     
      double yin = l1;
      double zin = l3;
      
      // Given the location in y and z in sensor local coordinates, find the strip y and z index values

      // find the sensor type (inner or outer) from the segment_z_bin (location of sensor on ladder)
      const int itype = segment_z_bin % 2;
      if (itype != 0 && itype != 1)
      {
         cout << "Problem: itype = " << itype << endl;
         TVector3 v3(-9999,-9999,-9999);
         return v3; 
      }

      // expect cm
      double zpos = zin;
      double ypos = yin;

      const double strip_z = m_StripZ[itype];
      const int nstrips_z_sensor = m_NStripsZSensor[itype];
      const int nstrips_y_sensor = m_NStripsPhiCell;

      // get the strip z index
      double zup = (double) nstrips_z_sensor * strip_z / 2.0 + zpos;
      //int strip_z_index = (int) (zup / strip_z);
      float strip_z_index = (float) (zup / strip_z) - 0.5; 
      
      // get the strip y index
      double yup = (double) nstrips_y_sensor * m_StripY / 2.0 + ypos;
      //int strip_y_index = (int) (yup / m_StripY);
      float strip_y_index = (float) (yup / m_StripY) - 0.5;
      
      TVector3 v3(std::round(strip_y_index/PIX_PRECISION)*PIX_PRECISION,std::round(strip_z_index/PIX_PRECISION)*PIX_PRECISION,0);            
      return v3;   
   }
}

////////////////////////////////////////////////////////////////////////////////
/// SToL // mvtx ok + intt ok

//local (x,y) -> pixel (row, col)
TVector3 YDetectorGeometry::SToL(int chipID, double s1, double s2, double s3) 
{ 

   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;     
   }

   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){
      //mvtx sensor 0 : row, 1: norm, 2: col
      s1 -= loc_sensor_in_chip_mvtx[2];
      s2 -= loc_sensor_in_chip_mvtx[0];
      s3 -= loc_sensor_in_chip_mvtx[1];    

      s1 = -s1;
      s2 = -s2;
      
      s2 = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - s2;   // coordinate wrt top edge of Active matrix
      s1 += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
      float frow = float(s2 / PitchRow) - 0.5;
      float fcol = float(s1 / PitchCol) - 0.5;
      //TVector3 v3(frow,fcol,0);
      TVector3 v3(std::round(frow/PIX_PRECISION)*PIX_PRECISION,std::round(fcol/PIX_PRECISION)*PIX_PRECISION,0);
      return v3;      
   } else {

      //implementation : void CylinderGeomIntt::find_strip_index_values(const int segment_z_bin, const double yin, const double zin, int &strip_y_index, int &strip_z_index)

      //int m_Layer;
      int m_NStripsPhiCell = 256;
      int m_NStripsZSensor[2];
      //double m_StripX = 0.032;
      double m_StripY = 0.0078; 
      //double m_SensorRadius;
      //double m_StripXOffset;
      //double m_OffsetPhi;
      //double m_OffsetRot;
      double m_dPhi;      //m_dPhi = 2. * TMath::Pi() / nladders_layer;
      m_dPhi = 2. * TMath::Pi() / mNumberOfStaves[Layer];

      double m_StripZ[2];
      //double m_LadderZ[2];

      // Type-A
      m_StripZ[0] = 1.6;//strip_z0;
      //m_LadderZ[0] = ladder_z0;
      m_NStripsZSensor[0] = 8;//nstrips_z_sensor0;

      // Type-B
      m_StripZ[1] = 2.0;//strip_z1;
      //m_LadderZ[1] = ladder_z1;
      m_NStripsZSensor[1] = 5;//nstrips_z_sensor1;

/*
      CylinderGeomIntt *mygeom = new CylinderGeomIntt(
          sphxlayer,					     		const int layer,			m_Layer(layer)
          params->get_double_param("strip_x"),		     		const double strip_x,			m_StripX(strip_x)	
          params->get_double_param("strip_y"),		     		const double strip_y,			m_StripY(strip_y)
          params->get_double_param("strip_z_0"),			const double strip_z0,	
          params->get_double_param("strip_z_1"),			const double strip_z1,
          params->get_int_param("nstrips_z_sensor_0"),			const int nstrips_z_sensor0,
          params->get_int_param("nstrips_z_sensor_1"),			const int nstrips_z_sensor1,
          params->get_int_param("nstrips_phi_sensor"),			const int nstrips_phi_cell,		m_NStripsPhiCell(nstrips_phi_cell)
          params_layer->get_int_param("nladder"),			const int nladders_layer,
          m_PosZ[ilayer][0] / cm,  					const double ladder_z0,
          m_PosZ[ilayer][1] / cm,					const double ladder_z1,
          m_SensorRadius[ilayer] / cm,					const double sensor_radius,		m_SensorRadius(sensor_radius)
          0.0,								const double strip_x_offset,		m_StripXOffset(strip_x_offset)
          params_layer->get_double_param("offsetphi") * deg / rad,  	const double offsetphi,			m_OffsetPhi(offsetphi)
          params_layer->get_double_param("offsetrot") * deg / rad);	const double offsetrot)			m_OffsetRot(offsetrot)
*/      
      
      int layer(-1), hb(-1), stv(-1), md(-1), mchip(-1);
      GetChipId(chipID, layer, hb, stv, md, mchip);
      int segment_z_bin = mchip;

      float l1= -s2;  //row
      float l2= +s3;  //norm
      float l3= -s1;  //col
     
      double yin = l1;
      double zin = l3;    

      // Given the location in y and z in sensor local coordinates, find the strip y and z index values

      // find the sensor type (inner or outer) from the segment_z_bin (location of sensor on ladder)
      const int itype = segment_z_bin % 2;
      if (itype != 0 && itype != 1)
      {
         cout << "Problem: itype = " << itype << endl;
         TVector3 v3(-9999,-9999,-9999);
         return v3; 
      }

      // expect cm
      double zpos = zin;
      double ypos = yin;

      const double strip_z = m_StripZ[itype];
      const int nstrips_z_sensor = m_NStripsZSensor[itype];
      const int nstrips_y_sensor = m_NStripsPhiCell;

      // get the strip z index
      double zup = (double) nstrips_z_sensor * strip_z / 2.0 + zpos;
      //int strip_z_index = (int) (zup / strip_z);
      float strip_z_index = (float) (zup / strip_z) - 0.5;
      
      // get the strip y index
      double yup = (double) nstrips_y_sensor * m_StripY / 2.0 + ypos;
      //int strip_y_index = (int) (yup / m_StripY);
      float strip_y_index = (float) (yup / m_StripY) - 0.5;
      
      TVector3 v3(std::round(strip_y_index/PIX_PRECISION)*PIX_PRECISION,std::round(strip_z_index/PIX_PRECISION)*PIX_PRECISION,0);      
      return v3; 

   }
}

////////////////////////////////////////////////////////////////////////////////
/// SToG

//local (x,y) -> global(x,y,z)
TVector3 YDetectorGeometry::SToG(int chipID, double s1, double s2, double s3)
{

   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;     
   }
   
   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){
      
      //std::cout<<"[SToG MVTX STEP 0] ChipID "<<chipID<<" senC "<<s1<<" "<<s2<<" "<<s3<<std::endl;   
      TVector3 stol = SToL(chipID, s1, s2, s3);
      float row = stol(0);
      float col = stol(1);
      //std::cout<<"[SToG MVTX STEP 1] ChipID "<<chipID<<" row "<<row<<" col "<<col<<std::endl;   
      TVector3 ltog = LToG(chipID, row, col);

      //std::cout<<"[SToG MVTX STEP 2] ChipID "<<chipID<<" locC "<<ltog.X()<<" "<<ltog.Y()<<" "<<ltog.Z()<<std::endl;   
      float gx = ltog.X();
      float gy = ltog.Y();
      float gz = ltog.Z();

      TVector3 normV = NormalVector(chipID);
      float dgx = s3*normV(0);
      float dgy = s3*normV(1);

      TVector3 v3(gx+dgx,gy+dgy,gz);
      return v3;
   } else {

      //std::cout<<"[SToG INTT STEP 0] ChipID "<<chipID<<" senC "<<s1<<" "<<s2<<" "<<s3<<std::endl;   
      TVector3 stol = SToL(chipID, s1, s2, s3);
      float row = stol(0);
      float col = stol(1);
      //std::cout<<"[SToG INTT STEP 1] ChipID "<<chipID<<" row "<<row<<" col "<<col<<std::endl;   
      TVector3 ltog = LToG(chipID, row, col);

      //std::cout<<"[SToG INTT STEP 2] ChipID "<<chipID<<" locC "<<ltog.X()<<" "<<ltog.Y()<<" "<<ltog.Z()<<std::endl;   
      float gx = ltog.X();
      float gy = ltog.Y();
      float gz = ltog.Z();

      TVector3 normV = NormalVector(chipID);
      float dgx = s3*normV(0);
      float dgy = s3*normV(1);

      TVector3 v3(gx+dgx,gy+dgy,gz);
      return v3;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// LToG // mvtx ok + intt ok

//pixel (row, col) -> global(x,y,z)
TVector3 YDetectorGeometry::LToG(int chipID, float row, float col) 
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;     
   }

   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){

      //o2::math_utils::Point3D<float> locC;
      TVector3 locC;
      //o2::itsmft::SegmentationAlpide::detectorToLocal(row, col, locC); // local coordinates
      SegmentationAlpide::detectorToLocalUnchecked(row, col, locC); // local coordinates
   
      //std::cout<<"[LToG MVTX STEP 0] ChipID "<<chipID<<" row "<<row<<" col "<<col<<" locC "<<locC.X()<<" "<<locC.Y()<<" "<<locC.Z()<<std::endl;
      double lx = locC.X() - loc_sensor_in_chip_mvtx[0];
      double ly = locC.Y() - loc_sensor_in_chip_mvtx[1]; 
      double lz = locC.Z() - loc_sensor_in_chip_mvtx[2];
      //std::cout<<"[LToG MVTX STEP 1] ChipID "<<chipID<<" row "<<row<<" col "<<col<<" locC "<<lx<<" "<<ly<<" "<<lz<<std::endl;
      
      locC.SetXYZ(lx*unit_cm, lz*unit_cm, 0);
     
      //auto gloC = getMatrixL2G(chipID) * locC;
      float locC_used[3] = {lx*unit_cm, lz*unit_cm, 0};
      float gloC_used[3];
      mL2G[chipID].LocalToMaster(locC_used, gloC_used);
      TVector3 gloC;
      gloC.SetXYZ(gloC_used[0], gloC_used[1], gloC_used[2]);

      float gx = gloC.X()/unit_cm;
      float gy = gloC.Y()/unit_cm;
      float gz = gloC.Z()/unit_cm;

      TVector3 v3(gx,gy,gz);
      return v3;
   } else {

      //implementation : void CylinderGeomIntt::find_strip_center_localcoords(const int segment_z_bin, const int strip_y_index, const int strip_z_index, double location[])

      //int m_Layer;
      int m_NStripsPhiCell = 256;
      int m_NStripsZSensor[2];
      //double m_StripX = 0.032;
      double m_StripY = 0.0078; 
      //double m_SensorRadius;
      //double m_StripXOffset;
      //double m_OffsetPhi;
      //double m_OffsetRot;
      double m_dPhi;      //m_dPhi = 2. * TMath::Pi() / nladders_layer;
      m_dPhi = 2. * TMath::Pi() / mNumberOfStaves[Layer];

      double m_StripZ[2];
      //double m_LadderZ[2];

      // Type-A
      m_StripZ[0] = 1.6;//strip_z0;
      //m_LadderZ[0] = ladder_z0;
      m_NStripsZSensor[0] = 8;//nstrips_z_sensor0;

      // Type-B
      m_StripZ[1] = 2.0;//strip_z1;
      //m_LadderZ[1] = ladder_z1;
      m_NStripsZSensor[1] = 5;//nstrips_z_sensor1;

/*
      CylinderGeomIntt *mygeom = new CylinderGeomIntt(
          sphxlayer,					     		const int layer,			m_Layer(layer)
          params->get_double_param("strip_x"),		     		const double strip_x,			m_StripX(strip_x)	
          params->get_double_param("strip_y"),		     		const double strip_y,			m_StripY(strip_y)
          params->get_double_param("strip_z_0"),			const double strip_z0,	
          params->get_double_param("strip_z_1"),			const double strip_z1,
          params->get_int_param("nstrips_z_sensor_0"),			const int nstrips_z_sensor0,
          params->get_int_param("nstrips_z_sensor_1"),			const int nstrips_z_sensor1,
          params->get_int_param("nstrips_phi_sensor"),			const int nstrips_phi_cell,		m_NStripsPhiCell(nstrips_phi_cell)
          params_layer->get_int_param("nladder"),			const int nladders_layer,
          m_PosZ[ilayer][0] / cm,  					const double ladder_z0,
          m_PosZ[ilayer][1] / cm,					const double ladder_z1,
          m_SensorRadius[ilayer] / cm,					const double sensor_radius,		m_SensorRadius(sensor_radius)
          0.0,								const double strip_x_offset,		m_StripXOffset(strip_x_offset)
          params_layer->get_double_param("offsetphi") * deg / rad,  	const double offsetphi,			m_OffsetPhi(offsetphi)
          params_layer->get_double_param("offsetrot") * deg / rad);	const double offsetrot)			m_OffsetRot(offsetrot)
*/      
      
      int layer(-1), hb(-1), stv(-1), md(-1), mchip(-1);
      GetChipId(chipID, layer, hb, stv, md, mchip);
      int segment_z_bin = mchip;
      //int strip_y_index = row;
      //int strip_z_index = col;
      float strip_y_index = row;
      float strip_z_index = col;

      // find the sensor type (inner or outer) from the segment_z_bin (location of sensor on ladder)
      const int itype = segment_z_bin % 2;
      if (itype != 0 && itype != 1)
      {
         cout << "Problem: itype = " << itype << endl;
         
         TVector3 v3(-9999,-9999,-9999);
         return v3;  
      }

      const double strip_z = m_StripZ[itype];
      const int nstrips_z_sensor = m_NStripsZSensor[itype];
      const int nstrips_y_sensor = m_NStripsPhiCell;

      // center of strip in y
      double ypos = (double) strip_y_index * m_StripY + m_StripY / 2.0 - (double) nstrips_y_sensor * m_StripY / 2.0;

      // center of strip in z
      double zpos = (double) strip_z_index * strip_z + strip_z / 2.0 - (double) nstrips_z_sensor * strip_z / 2.0;

      //o2::math_utils::Point3D<float> locC;

      //locC.SetXYZ(ypos*unit_cm, zpos*unit_cm, 0);
      
      //auto gloC = getMatrixL2G(chipID) * locC;

      float locC_used[3] = {ypos*unit_cm, zpos*unit_cm, 0};
      float gloC_used[3];
      mL2G[chipID].LocalToMaster(locC_used, gloC_used);
      TVector3 gloC;
      gloC.SetXYZ(gloC_used[0], gloC_used[1], gloC_used[2]);

      float gx = gloC.X()/unit_cm;
      float gy = gloC.Y()/unit_cm;
      float gz = gloC.Z()/unit_cm;

      TVector3 v3(gx,gy,gz);
      return v3; 
   }
}

TVector3 YDetectorGeometry::NormalVector(int chipID)
{ // Stave plane 				
   if(chipID<0){
      TVector3 e3(-9999,-9999,-9999);
      return e3;     
   }
         
   TVector3 v1(LToG(chipID,256,512).X()-LToG(chipID,256,0).X(),
               LToG(chipID,256,512).Y()-LToG(chipID,256,0).Y(),
               LToG(chipID,256,512).Z()-LToG(chipID,256,0).Z());
   TVector3 v2(LToG(chipID,0,0).X()-LToG(chipID,256,0).X(),
               LToG(chipID,0,0).Y()-LToG(chipID,256,0).Y(),
               LToG(chipID,0,0).Z()-LToG(chipID,256,0).Z());           
   TVector3 v2v1 = v2.Cross(v1);
   double m2m1 = TMath::Sqrt((v2v1.X()*v2v1.X())+(v2v1.Y()*v2v1.Y())+(v2v1.Z()*v2v1.Z()));
   TVector3 v3(v2v1.X()/m2m1,v2v1.Y()/m2m1,v2v1.Z()/m2m1);
   return v3;

}

int YDetectorGeometry::GetLayer(int index) const 
{
   if(index<0) return -9999;
   
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   return lay;   
}

int YDetectorGeometry::GetHalfBarrel(int index) const 
{
   if(index<0) return -9999;
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   int quadrant = index / (mNumberOfChipsPerHalfBarrel[lay]/2.0);
   int halfbarrel = 0;
   
   if(quadrant==0 || quadrant==3) halfbarrel = 0; // WEST
   if(quadrant==1 || quadrant==2) halfbarrel = 1; // EAST
   return halfbarrel;
}

int YDetectorGeometry::GetStave(int index) const 
{
   if(index<0) return -9999;

   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   return index / mNumberOfChipsPerStave[lay];
}

int YDetectorGeometry::GetHalfStave(int index) const 
{
   if(index<0) return -9999;
   return 0;
}

int YDetectorGeometry::GetModule(int index) const 
{
   if(index<0) return -9999;
   return 0;
}

int YDetectorGeometry::GetChipIdInLayer(int index) const 
{
   if(index<0) return -9999;
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   return index;
}

int YDetectorGeometry::GetChipIdInStave(int index) const 
{
   if(index<0) return -9999;
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   return index % mNumberOfChipsPerStave[lay];
}

int YDetectorGeometry::GetChipIdInHalfStave(int index) const 
{
   if(index<0) return -9999;
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   return index % mNumberOfChipsPerHalfStave[lay];
}

int YDetectorGeometry::GetChipIdInModule(int index) const 
{
   if(index<0) return -9999;
   int lay = 0;
   while (index > mLastChipIndex[lay]) {
      lay++;
   }
   index -= GetFirstChipIndex(lay);
   return index % mNumberOfChipsPerModule[lay];
}

bool YDetectorGeometry::GetChipId(int index, int& lay, int& hba, int& sta, int& mod, int& chip) const
{
  lay = GetLayer(index);
  //hb
  index -= GetFirstChipIndex(lay);
  hba = mNumberOfHalfBarrels > 0 ? index / mNumberOfChipsPerHalfBarrel[lay] : -1;
  //sta
  index %= mNumberOfChipsPerHalfBarrel[lay];
  sta = index / mNumberOfChipsPerStave[lay];
  //mod
  index %= mNumberOfChipsPerHalfStave[lay];
  mod = mNumberOfModules[lay] > 0 ? index / mNumberOfChipsPerModule[lay] : -1;
  //mchip
  chip = index % mNumberOfChipsPerModule[lay];

  return kTRUE;
}

int YDetectorGeometry::GetChipIndex(int lay, int sta, int chipInStave) const
{
  return GetFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInStave;
}

