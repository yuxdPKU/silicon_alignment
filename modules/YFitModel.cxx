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
// YFitModel
//
//
///////////////////////////////////////////////////////////////////////////

#include "YFitModel.h"

static void at_exit_of_YFitModel() {
   if (FIT::Internal::yFITLocal)
      FIT::Internal::yFITLocal->~YFitModel();
}

// This local static object initializes the FIT system
namespace FIT {
namespace Internal {
   class YFitModelAllocator {

      char fHolder[sizeof(YFitModel)];
   public:
      YFitModelAllocator() {
         new(&(fHolder[0])) YFitModel();
      }

      ~YFitModelAllocator() {
         if (yFITLocal) {
            yFITLocal->~YFitModel();
         }
      }
   };

   extern YFitModel *yFITLocal;

   YFitModel *GetFIT1() {
      if (yFITLocal)
         return yFITLocal;
      static YFitModelAllocator alloc;
      return yFITLocal;
   }

   YFitModel *GetFIT2() {
      static Bool_t initInterpreter = kFALSE;
      if (!initInterpreter) {
         initInterpreter = kTRUE;
      }
      return yFITLocal;
   }
   typedef YFitModel *(*GetFITFun_t)();

   static GetFITFun_t yGetFIT = &GetFIT1;


} // end of Internal sub namespace
// back to FIT namespace

   YFitModel *GetFIT() {
      return (*Internal::yGetFIT)();
   }
}

YFitModel *FIT::Internal::yFITLocal = FIT::GetFIT();

ClassImp(YFitModel);

////////////////////////////////////////////////////////////////////////////////
/// Default Constructor YFitModel

YFitModel::YFitModel() {
   std::cout<<"Default Constructor YFitModel "<<std::endl;
   //These are supposed to be defined as constant header.
   fB     = 0;

   FIT::Internal::yFITLocal = this;
   FIT::Internal::yGetFIT = &FIT::Internal::GetFIT2;   
}

////////////////////////////////////////////////////////////////////////////////
/// Fit by selected model

void YFitModel::Fit(double* input, double* par, double &MSEvalue, int hitentries=3, YFitModel::EModel model = YFitModel::kLine){

   switch (model) {
      case YFitModel::kLine: {
         line3Dfit(input, par, MSEvalue, hitentries);
         break;
      }
      case YFitModel::kCircle: {
         //circle3Dfit(input, par, MSEvalue, hitentries);
         break;
      }      
      default:
      ;
   } 
} 
  
////////////////////////////////////////////////////////////////////////////////
/// Vertex Estimator

void YFitModel::EstimateVertex(double** input, double* kappa, double* vgz, int hitentries=3, int trackentries=2, YFitModel::EModel model = YFitModel::kLine){

   switch (model) {
      case YFitModel::kLine: {
         std::cout<<"Estimated Vertex by Straight Line Tracks"<<std::endl; 
         break;
      }
      case YFitModel::kCircle: {
         std::cout<<"Estimated Vertex by Circle(Helix) Tracks under B-Field"<<std::endl;      
         double fitpar[trackentries][5];
         double MSEvalue[trackentries];
         for(int t=0; t<trackentries; t++){                     
            std::cout<<"input : ";         
            for(int i=0; i<hitentries*3; i++){
               std::cout<<input[t][i]<<" ";            
            }
            std::cout<<std::endl;
         
            for(int p=0; p<5; p++){
               fitpar[t][p] = 0;
            }
            MSEvalue[t] = 0;
            circleVertex(input[t], fitpar[t], MSEvalue[t], 3);
 
///     
            TVector3 vecX[hitentries+1];
            TVector3 vecXmid[hitentries];
            double InvSlope[hitentries];  
            for(int layer = 0; layer < hitentries; layer++){   
               vecX[layer].SetXYZ(input[t][(3*layer)+0],input[t][(3*layer)+1],input[t][(3*layer)+2]);  
            }
            vecX[hitentries].SetXYZ(0,0,0);  

            double pos_1[2],pos_2[2],pos_3[2];
            double est_1[2],est_2[2],est_3[2]; 
            std::cout<<"YFitModel::EstimateVertex kappa thetaB zeta1 zeta2 thetaR ";
            for(int p = 0; p < 5; p++){
               std::cout<<fitpar[t][p]<<" ";
            }            
            std::cout<<std::endl;

            double RecRadius = fitpar[t][0]>0 ? std::abs(1/(CM2M*(fitpar[t][0] + MinRes))) : std::abs(1/(CM2M*(fitpar[t][0] - MinRes)));
            double CircleXc  = fitpar[t][0]>0 ? RecRadius*std::cos(fitpar[t][1]+fitpar[t][4] + 0.5*TMath::Pi()) : RecRadius*std::cos(fitpar[t][1]+fitpar[t][4] - 0.5*TMath::Pi());
            double CircleYc  = fitpar[t][0]>0 ? RecRadius*std::sin(fitpar[t][1]+fitpar[t][4] + 0.5*TMath::Pi()) : RecRadius*std::sin(fitpar[t][1]+fitpar[t][4] - 0.5*TMath::Pi()); 
   
            std::cout<<"YFitModel::EstimateVertex RecRadius Xc Yc : "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<std::endl;
            CircleXc += input[t][3*(0)+0]; 
            CircleYc += input[t][3*(0)+1];  
            std::cout<<"YFitModel::EstimateVertex RecRadius Xc Yc : "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<std::endl;              
            TVector3 vecXc(CircleXc, CircleYc, 0);
            TVector3 dirXr[hitentries+1];
            for(int a=0; a<hitentries+1;a++){
               dirXr[a] = vecX[a] - vecXc;
               dirXr[a].SetZ(0);
               dirXr[a].Print();
            }

            double beta[hitentries+1];
            for(int l = 0; l < hitentries+1; l++){    
                beta[l] = std::atan2(dirXr[l].Y(), dirXr[l].X()) > 0 ? std::atan2(dirXr[l].Y(), dirXr[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXr[l].Y(), dirXr[l].X());            
               std::cout<<" YFitModel::EstimateVertex beta "<<l<<" "<<std::atan2(dirXr[l].Y(), dirXr[l].X())<<" "<<beta[l]<<std::endl;                              
            } 
            TMatrixD MatA(3,2);
            MatA[0]  = { beta[0],	1};  
            MatA[1]  = { beta[1],	1}; 
            MatA[2]  = { beta[2],	1}; 

            TMatrixD MatAT(2,3);
            MatAT[0] = {beta[0], beta[1], beta[2]};
            MatAT[1] = {      1,       1,	1};  

            TMatrixD MatAT0(2,3);
            MatAT0[0] = {beta[0], beta[1], beta[2]};
            MatAT0[1] = {      1,       1,	 1}; 

            TMatrixD MatZ(3,1);
            MatZ[0]  = { vecX[0].Z()};  
            MatZ[1]  = { vecX[1].Z()}; 
            MatZ[2]  = { vecX[2].Z()};  
    
            TMatrixD MatATAInv(2,2);
            double DetATA = 3*(beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2]) - (beta[0] + beta[1] + beta[2])*(beta[0] + beta[1] + beta[2]);
            MatATAInv[0] = {3/DetATA, 					    -(beta[2] + beta[0] + beta[1])/DetATA};
            MatATAInv[1] = {-(beta[0] + beta[1] + beta[2])/DetATA, (beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2])/DetATA};
     
            TMatrixD MatP(2,1);
             
            MatP = MatATAInv * MatAT0 * MatZ;

            double parz[2];
  
            parz[0] = MatP[0][0];
            parz[1] = MatP[1][0];  

            std::cout<<" YFitModel::EstimateVertex parz : "<<parz[0]<<" "<<parz[1]<<std::endl;         
            std::cout<<" YFitModel::EstimateVertex Circle(Xc, Yc, R) = "<<CircleXc<<" "<<CircleYc<<" "<<RecRadius<<std::endl;             
            std::cout<<" YFitModel::EstimateVertex thetaB(Est, 0) = "<<fitpar[t][1] + fitpar[t][4]<<" "<<std::atan2(input[t][3*(0)+1],input[t][3*(0)+0])<<std::endl;             
            for(int layer = 0; layer < hitentries; layer++){           
               //"evno:layer:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE"); 
               
               //corrected
               pos_1[0] = input[t][(3*layer)+0]; //alpha
               pos_2[0] = input[t][(3*layer)+1]; //beta
               pos_3[0] = input[t][(3*layer)+2]; //gamma      
              
               est_1[0] = RecRadius*std::cos(beta[layer]) + CircleXc;
               est_2[0] = RecRadius*std::sin(beta[layer]) + CircleYc;
               est_3[0] = (parz[0])*(beta[layer]) + (parz[1]); 
               std::cout<<" YFitModel::EstimateVertex :: layer pos1 est1 "<<pos_1[0]<<" "<<est_1[0]<<std::endl;
               std::cout<<" YFitModel::EstimateVertex :: layer pos2 est2 "<<pos_2[0]<<" "<<est_2[0]<<std::endl;
               std::cout<<" YFitModel::EstimateVertex :: layer pos3 est3 "<<pos_3[0]<<" "<<est_3[0]<<std::endl;                              
               std::cout<<" YFitModel::EstimateVertex :: layer ["<<layer<<"] beta= "<<beta[layer]<<std::endl;
            }        
   
            //"evno:layer:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE"); 
               
            //corrected
            pos_1[0] = 0; //alpha
            pos_2[0] = 0; //beta
            pos_3[0] = 0; //gamma      
               
            est_1[0] = RecRadius*std::cos(beta[hitentries]) + CircleXc;
            est_2[0] = RecRadius*std::sin(beta[hitentries]) + CircleYc;
            est_3[0] = (parz[0])*(beta[hitentries]) + (parz[1]); 
            std::cout<<" YFitModel::EstimateVertex :: layer pos1 est1 "<<pos_1[0]<<" "<<est_1[0]<<std::endl;
            std::cout<<" YFitModel::EstimateVertex :: layer pos2 est2 "<<pos_2[0]<<" "<<est_2[0]<<std::endl;
            std::cout<<" YFitModel::EstimateVertex :: layer pos3 est3 "<<pos_3[0]<<" "<<est_3[0]<<std::endl;                              
            std::cout<<" YFitModel::EstimateVertex :: Beam  [ ] beta = "<<beta[hitentries]<<std::endl;
            
            kappa[t] = fitpar[t][0];
            vgz[t] = est_3[0];
///         
         }       
         break;
      }      
      default:
      ;
   }
}
////////////////////////////////////////////////////////////////////////////////
/// DeltaZ

void YFitModel::DeltaZ(double* kappa, double* vgz, double* dvgz, int trackentries=2){

   for(int i =0; i < trackentries; i++){
      double vgzmean = 0;
      std::cout<<" YFitModel::DeltaZ "<<i<<" ";
      for(int j =0; j < trackentries; j++){
         if(i==j) continue;
         std::cout<<"["<<j<<"] "<<vgz[j]<<" ";
         vgzmean += (double)vgz[j]/(double)(trackentries-1);
      } 
      std::cout<<"vgz vgzmean "<<vgz[i]<<" "<<vgzmean<<" dvgz "<<dvgz[i]<<std::endl;
      dvgz[i] = vgz[i] - vgzmean;
   }  
}

////////////////////////////////////////////////////////////////////////////////
/// Clustering by DBSCAN

void YFitModel::Clustering(double* vgz, int* index, int trackentries=2, YFitModel::EClusterMethod method = YFitModel::kDBSCAN, double epsilon = 0.1, bool ascending =true){

   //std::cout<<"Clustering"<<std::endl;
   double npmin   = 1;
     
   int tag = 0;
   if(ascending==true){
      for(int i =0; i < trackentries; i++){
      //for(int i =trackentries-1; i >=0; i--){
         std::cout<<"vgz ["<<i<<"] "<<vgz[i]<<std::endl;
         if(index[i]!=-1) continue; 
         vector<int> itrack;
         itrack.push_back(i);
         index[i] = tag;
         for(int j =i+1; j < trackentries; j++){
         //for(int j =i-1; j >=0; j--){ 
            if(index[j]!=-1) continue;       
            itrack.push_back(j);
            double distance = std::abs(vgz[j] - vgz[i]);
            std::cout<<" >> "<<j<<" dist = "<<distance<<std::endl;
            if(distance<epsilon) {
               index[j] = tag;
            }
         }
         if(itrack.size()<npmin){
            for(int k =0; k < itrack.size(); k++){
               index[itrack[k]] = -1;
            } 
            continue;  
         }
         tag++;
      }  
   } else {
      //for(int i =0; i < trackentries; i++){
      for(int i =trackentries-1; i >=0; i--){
         std::cout<<"vgz ["<<i<<"] "<<vgz[i]<<std::endl;
         if(index[i]!=-1) continue; 
         vector<int> itrack;
         itrack.push_back(i);
         index[i] = tag;
         //for(int j =i+1; j < trackentries; j++){
         for(int j =i-1; j >=0; j--){ 
            if(index[j]!=-1) continue;       
            itrack.push_back(j);
            double distance = std::abs(vgz[j] - vgz[i]);
            std::cout<<" >> "<<j<<" dist = "<<distance<<std::endl;
            if(distance<epsilon) {
               index[j] = tag;
            }
         }
         if(itrack.size()<npmin){
            for(int k =0; k < itrack.size(); k++){
               index[itrack[k]] = -1;
            } 
            continue;  
         }
         tag++;
      }    
   }
   std::cout<<"index : ";
   for(int i =0; i < trackentries; i++){
      std::cout<<index[i]<<" ";
   }
   std::cout<<std::endl;
}



